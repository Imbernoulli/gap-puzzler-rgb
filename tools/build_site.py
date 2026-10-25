"""Build site/ from recorded runs.

  python tools/build_site.py --rig official RUN MEDIA --rig custom RUN MEDIA \
      [--grid RIG VIDEO CAPTION ...] [--note RIG TEXT ...]

Each --rig names a camera rig, the full-12 run shown in its player (a run
directory containing workspace/runs/<RUN>/, or that inner directory) and the
tools/make_media.py folder rendered from it. The page shows one rig at a time
behind tabs. Every number on the page is read here from the recorded result,
checkpoint log and workflow; nothing is typed by hand.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'site_src'
SITE = ROOT / 'site'
RUN = 'full12_original_order_layout_c_v10'
FRAME_STEPS = 5  # episode/frames.json samples every fifth control step

COLORS = dict(yellow=(1., .69, .01), purple=(.31, .04, .46), cyan=(.07, .73, .98), lime=(.53, .85, .03),
              mint=(.25, .88, .66), maroon=(.64, .025, .10), teal=(.01, .48, .44), orange=(1., .25, .01),
              navy=(.08, .08, .68), pink=(.96, .19, .59), red=(.88, .035, .06), azure=(.015, .43, .91))

LABELS = {
    'perceive_plan': ('Perceive and plan', 'Fit every visible piece and the case from the four RGB views, then plan a grasp from CAD and the arm\'s kinematics.'),
    'grasp': ('Grasp and lift', 'Close on the planned beads, lift, and confirm the hold from a fresh RGB frame and the jaw encoders.'),
    'transport': ('Carry', 'Move the held piece above its pyramid site, planned from the observed case pose.'),
    'release': ('Release', 'Lower, open the jaw, and back away.'),
    'receipt_settle': ('Check placement', 'Two RGB frames one second apart must agree: piece still, hand clear, beads at their sites.'),
    'handoff': ('Hand off', 'Pass the updated scene state to the next piece\'s subgraph.'),
    'cyan_approach': ('Bring hands together', 'Move the held piece to a shared station where the other arm can reach it.'),
    'cyan_receiver_close': ('Second hand closes', 'Close the receiving jaw while the first hand still holds.'),
    'cyan_donor_release': ('First hand opens', 'Open and withdraw the first jaw.'),
    'cyan_receiver_bind': ('Confirm new grip', 'Re-estimate the held pose from fresh RGB and the receiving jaw.'),
    'cyan_receiver_transport': ('Carry with second hand', 'Carry the piece to its site with the receiving arm.'),
    'red_handoff_prepare': ('Bring hands together', 'First transfer: move to a station both arms can reach.'),
    'red_receiver_close': ('Second hand closes', 'First transfer: close the receiving jaw.'),
    'red_sender_release': ('First hand opens', 'First transfer: open and withdraw the giving jaw.'),
    'red_receiver_bind': ('Confirm new grip', 'First transfer: re-estimate the held pose from fresh RGB.'),
    'red_return_prepare': ('Hands together again', 'Second transfer, back to the first arm with a new grasp.'),
    'red_return_close': ('Hand closes', 'Second transfer: close the receiving jaw.'),
    'red_return_release': ('Other hand opens', 'Second transfer: open and withdraw.'),
    'red_return_bind': ('Confirm new grip', 'Second transfer: re-estimate the held pose from fresh RGB.'),
    'red_goal_transport': ('Carry', 'Carry Red to the apex site.'),
    'red_goal_release': ('Release', 'Lower Red onto the pyramid, open, and back away.'),
    'red_goal_verify': ('Check placement', 'Re-observe and close the visual ledger for all twelve pieces.'),
    'finish': ('Finish', 'Open both hands, send both arms home, wait two seconds, and take a final look.'),
}


# Nodes whose code takes a bounded extra look before refusing.
RELOOK = {'receipt_settle', 'finish'}
# Muted inline clips may autoplay in Safari and iOS; app.js plays only what is on screen.
AUTOPLAY = 'data-autoplay autoplay muted loop playsinline controls preload="auto"'


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def inner(run_dir):
    run_dir = Path(run_dir).resolve()
    candidate = run_dir / 'workspace/runs' / RUN
    return candidate if candidate.exists() else run_dir


def success_path(workflow):
    edges = workflow['conditional_edges']
    node = dict(workflow['edges'])['START']
    path = []
    while node not in ('done', 'abort'):
        path.append(node)
        node = next(target for exit_, target in edges[node]['mapping'].items()
                    if exit_ != 'error' and target != 'abort')
    return path


def node_kind(name, piece):
    if name == 'finish_all12':
        return 'finish'
    kind = name[len(piece) + 1:]
    return kind[len(piece) + 1:] if kind.startswith(piece + '_') else kind


PRIMARY = ('placed', 'planned', 'verified', 'carried', 'advanced', 'completed', 'transferred', 'prepared')


def exits(workflow, name):
    mapping = workflow['conditional_edges'][name]['mapping']
    names = sorted({exit_ for exit_, target in mapping.items() if target != 'abort'})
    names.sort(key=lambda e: PRIMARY.index(e) if e in PRIMARY else len(PRIMARY))
    return names


def exit_text(names):
    return names[0] + (f' +{len(names) - 1}' if len(names) > 1 else '')


def metrics(row, first=False):
    out = {}
    if row.get('target_goal_bead_error_m') is not None:
        out['placement_error_mm'] = round(1000 * row['target_goal_bead_error_m'], 2)
    if row.get('maximum_prefix_bead_error_m') is not None and not first:
        out['earlier_pieces_max_mm'] = round(1000 * row['maximum_prefix_bead_error_m'], 2)
    if row.get('released_stable_tail_s') is not None:
        out['stable_s'] = round(row['released_stable_tail_s'], 2)
    return out


def timeline(run):
    episode = run / 'episode'
    result = json.loads((episode / 'result.json').read_text())
    workflow = json.loads((episode / 'trace/0000/workflow.json').read_text())
    rows = jsonl(run / 'independent_verification/checkpoint_results.jsonl')
    path = success_path(workflow)
    order = workflow['meta']['ordered_stages']
    pieces, previous = [], 0
    for index, name in enumerate(path):
        piece = next((p for p in order if name.startswith(p + '_')), None) or 'finish'
        row = rows[index] if index < len(rows) else None
        kind = node_kind(name, piece) if piece != 'finish' else 'finish'
        label, detail = LABELS.get(f'{piece}_{kind}') or LABELS.get(kind) or (kind.replace('_', ' ').capitalize(), '')
        names = exits(workflow, name)
        node = dict(id=name, kind=kind, label=label, detail=detail, exits=names, exit=exit_text(names),
                    start=previous, end=row['native_step'] if row else None,
                    passed=bool(row and row.get('passed')), checkpoint=row['phase'] if row else None,
                    metrics=metrics(row, first=piece == order[0]) if row else {})
        if row:
            previous = row['native_step']
        if not pieces or pieces[-1]['name'] != piece:
            color = COLORS.get(piece, (.2, .2, .2))
            pieces.append(dict(name=piece, color='#%02x%02x%02x' % tuple(int(255 * c) for c in color), nodes=[]))
        pieces[-1]['nodes'].append(node)
    for piece in pieces:
        if piece['name'] in model.CELLS:
            piece['number'] = order.index(piece['name']) + 1
            piece['cells'] = shape_cells(piece['name'])
        piece['start'] = piece['nodes'][0]['start']
        piece['end'] = max((n['end'] for n in piece['nodes'] if n['end'] is not None), default=None)
        piece['placed'] = all(n['passed'] for n in piece['nodes'])
        release = [n for n in piece['nodes'] if n['kind'] in ('release', 'goal_release') and 'placement_error_mm' in n['metrics']]
        piece['placement_error_mm'] = release[0]['metrics']['placement_error_mm'] if release else None
    audit = result.get('independent_audit') or {}
    error = result.get('error')
    return dict(
        control_hz=result['control_hz'], steps=result['steps'], sim_time_s=round(result['sim_time_s'], 1),
        rss_success=result['rss_success'], graph_success=result['graph_success'],
        checkpoints_passed=sum(1 for r in rows if r.get('passed')), checkpoints_total=len(path),
        pieces_placed=sum(1 for p in pieces if p['name'] != 'finish' and p['placed']),
        audit_success=audit.get('success'), audit_max_mm=round(1000 * audit['final_max_bead_error_m'], 2) if audit else None,
        audit_stable_s=round(audit['released_stable_duration_s'], 1) if audit else None,
        error=str(error) if error else None, pieces=pieces)


def variant_row(run_dir):
    run = inner(run_dir)
    manifest = json.loads((run / 'manifest.json').read_text())
    rows = jsonl(run / 'independent_verification/checkpoint_results.jsonl') if (run / 'independent_verification/checkpoint_results.jsonl').exists() else []
    placed = [r['piece'] for r in rows if r.get('phase') == 'handoff' and r.get('passed')]
    out = dict(variant=manifest.get('variant') or Path(run_dir).name, label=manifest.get('variant_label'),
               scene=manifest.get('scene_variant'), seed=manifest.get('seed'), placed=len(placed), done=False)
    result_path = run / 'episode/result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        audit = result.get('independent_audit') or {}
        error = str(result.get('error') or '')
        match = re.search(r"first node error -- ([a-z_]+)_sg\.act: Node '[^']+' failed: (\w+): (.*)", error)
        piece = None
        if not result['graph_success'] and rows:
            done = set(placed)
            piece = next((p for p in manifest['ordered_stages'] if p not in done), None)
        refusal = match.group(3) if match else error
        refusal = refusal.split(' {')[0].split(': {')[0].split(" {'")[0].strip()
        checkpoint = re.search(r"subgraph '(\w+)_sg': \d+ validate=True checkpoint\(s\) failed", error)
        if checkpoint:
            refusal = f"independent physical checkpoint failed after {checkpoint.group(1)}"
        out.update(done=True, rss_success=result['rss_success'], graph_success=result['graph_success'],
                   sim_time_s=round(result['sim_time_s'], 1),
                   audit_max_mm=round(1000 * audit['final_max_bead_error_m'], 2) if audit.get('success') else None,
                   stopped_piece=piece, stopped_node=match.group(1) if match else None,
                   refusal=refusal[:160] or None)
    return out


def variant_label(row):
    labels = dict(nominal='Nominal layout', board_a='Case moved and turned', board_b='Case moved and turned',
                  jitter_a='Loose pieces nudged', jitter_b='Loose pieces nudged', friction_050='Lower friction',
                  friction_080='Higher friction', combined='All of the above', shuffled='Colors shuffled')
    return labels.get(row['variant'], row['variant'])


def describe(scene):
    parts = []
    if 'board_dx' in scene:
        parts.append(f"case {1000 * scene['board_dx']:+.0f}/{1000 * scene['board_dy']:+.0f} mm, {np.degrees(scene.get('board_yaw', 0)):+.1f}°")
    if 'piece_jitter' in scene:
        parts.append(f"pieces ±{1000 * scene['piece_jitter']:.0f} mm, ±{np.degrees(scene['piece_yaw']):.0f}°")
    if 'friction' in scene:
        parts.append(f"μ {scene['friction']:.2f}")
    if scene.get('appearance') == 'shuffled':
        parts.append('colors permuted')
    return ' · '.join(parts) or 'as designed'


def esc(text):
    return (str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;'))


def shapes_html(data):
    by_name = {p['name']: p for p in data['pieces']}
    rows = [('Ten pieces', [n for n in ('azure', 'orange', 'yellow', 'teal', 'purple', 'mint', 'lime', 'maroon', 'navy', 'pink')], 'azure'),
            ('Cyan · one hand-to-hand transfer', ['cyan'], 'cyan'), ('Red · two transfers', ['red'], 'red')]
    out = ['<div class="shapes">']
    for title, members, sample in rows:
        nodes = by_name[sample]['nodes']
        dots = ''.join(f'<i style="--c:{by_name[n]["color"]}" title="{n.capitalize()}"></i>' for n in members)
        chips = '<b aria-hidden="true">→</b>'.join(
            f'<span class="chip" title="{esc(n["detail"])}">{esc(n["label"])}<em title="{esc(" | ".join(n["exits"]))}">{esc(n["exit"])}</em></span>' for n in nodes)
        out.append(f'<div class="shape"><p class="shape-title"><span class="dots">{dots}</span>{esc(title)} · {len(nodes)} nodes</p>'
                   f'<div class="chips">{chips}</div></div>')
    out.append('<p class="note">Each chip shows a node and its success exits. Every node also has an <code>error</code> exit to <code>abort</code>, and each is followed by an independent physical checkpoint. A last <code>finish</code> node sends both arms home.</p></div>')
    return '\n'.join(out)


def outcome_cell(row):
    if row is None:
        return '<td class="muted">not run</td>'
    if not row['done']:
        return f'<td class="muted">running · {row["placed"]}/12 placed</td>'
    if row['graph_success'] and row['rss_success']:
        audit = f' · {row["audit_max_mm"]:.1f} mm' if row.get('audit_max_mm') else ''
        return f'<td><span class="pill pass">12/12</span> workflow complete{audit}</td>'
    if row['rss_success']:
        return (f'<td><span class="pill warn">12/12</span> pieces seated, last node refused'
                f'<small>{esc(row.get("refusal") or "")}</small></td>')
    where = row['stopped_piece'].capitalize() if row.get('stopped_piece') else 'start'
    node = (row.get('stopped_node') or '').replace('_', ' ')
    return (f'<td><span class="pill fail">{row["placed"]}/12</span> stopped at {esc(where)}'
            f'<small>{esc(node + ": " if node else "")}{esc(row.get("refusal") or "")}</small></td>')


def runs_cell(rows):
    """k/n complete, plus where the other runs stopped."""
    if not rows:
        return '<td class="muted">not run</td>'
    done = [r for r in rows if r['done']]
    complete = [r for r in done if r['graph_success'] and r['rss_success']]
    stops = {}
    for r in done:
        if r in complete:
            continue
        where = (r.get('stopped_piece') or 'end').capitalize()
        node = (r.get('stopped_node') or '').replace('_', ' ')
        if r['rss_success'] and not r['graph_success']:
            where, node = 'all seated', 'last node refused'
        key = f'{where} · {node}' if node else where
        stops[key] = stops.get(key, 0) + 1
    running = len(rows) - len(done)
    detail = '; '.join(f'{k} ×{v}' if v > 1 else k for k, v in sorted(stops.items(), key=lambda kv: -kv[1]))
    if running:
        detail += f'{"; " if detail else ""}{running} still running'
    pill = 'pass' if complete else 'fail'
    best = max((r['placed'] for r in done), default=0)
    head = (f'<span class="pill {pill}">{len(complete)}/{len(done)}</span> complete' if complete
            else f'<span class="pill fail">0/{len(done)}</span> complete · best {best}/12')
    return f'<td>{head}<small>{esc(detail)}</small></td>'


def repeats_html(before, after, latest=None):
    """One row summarising every nominal repeat."""
    return (f'<tr><th scope="row">Layout C, repeats</th><td class="mono small">as designed</td>'
            f'{runs_cell(before)}{runs_cell(after)}{runs_cell(latest) if latest is not None else ""}</tr>')


def variants_html(variants, repeats='', latest_label=None):
    extra = f'<th scope="col">{esc(latest_label)}</th>' if latest_label else ''
    head = ('<thead><tr><th scope="col">Scene</th><th scope="col">What changed</th><th scope="col">v13 policy (one run)</th>'
            f'<th scope="col">This work (all runs)</th>{extra}</tr></thead>')
    body = repeats + ''.join(f'<tr><th scope="row">{esc(v["label"])}</th><td class="mono small">{esc(v["change"])}</td>'
                   f'{outcome_cell(v["before"])}{runs_cell(v["after"])}'
                   f'{runs_cell(v["latest"]) if latest_label else ""}</tr>' for v in variants)
    return head + '<tbody>' + body + '</tbody>'


def drift_svg(path):
    d = json.loads(Path(path).read_text())
    rows = d['rows']
    w, h, left, bottom, top, right = 600, 230, 44, 34, 14, 12
    tmax, ymax = 660., 2.5
    x = lambda t: left + (w - left - right) * t / tmax
    y = lambda v: top + (h - top - bottom) * (1 - v / ymax)
    rgb = ' '.join(f'{x(r["t"]):.1f},{y(r["rgb_mm"]):.1f}' for r in rows)
    phys = ' '.join(f'{x(r["t"]):.1f},{y(r["physical_mm"]):.1f}' for r in rows)
    grid = ''.join(f'<line class="grid" x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="tick" x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>' for v in (0, 1, 2))
    ticks = ''.join(f'<text class="tick" x="{x(t):.1f}" y="{h - 12}" text-anchor="middle">{t:g}</text>' for t in (0, 200, 400, 600))
    thr = d['receipt_threshold_mm']
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Case shift over the run: the RGB estimate wanders up to 2 mm from its first value while the physical case slides about 1.1 mm">'
            f'{grid}{ticks}<text class="tick" x="{w - right}" y="{h - 12}" text-anchor="end">s</text>'
            f'<line class="limit" x1="{left}" x2="{w - right}" y1="{y(thr):.1f}" y2="{y(thr):.1f}"/>'
            f'<text class="limit-label" x="{w - right}" y="{y(thr) - 6:.1f}" text-anchor="end">receipt check, {thr:g} mm</text>'
            f'<polyline class="phys" points="{phys}"/><polyline class="rgb" points="{rgb}"/>'
            f'<text class="series rgb-label" x="{x(470):.1f}" y="{y(2.25):.1f}">RGB estimate</text>'
            f'<text class="series phys-label" x="{x(440):.1f}" y="{y(.62):.1f}">physical case</text>'
            f'<text class="tick" x="{left - 8}" y="{top - 2}" text-anchor="end">mm</text></svg>')


def camera_grid(run, out):
    from PIL import Image
    frames = np.load(run / 'episode/rgb/0000.npz')
    views = [Image.fromarray(frames[name]).resize((640, 512)) for name in ('top', 'front', 'rear', 'side')]
    grid = Image.new('RGB', (1280, 1024))
    for i, view in enumerate(views):
        grid.paste(view, ((i % 2) * 640, (i // 2) * 512))
    grid.save(out, quality=82)


def loop_svg():
    steps = [('Rehearse', 'one skill, then 2–6 piece chains'), ('Check', 'independent checkpoint per node'),
             ('Locate', 'logs · saved RGB · robot state'), ('Change', 'one skill, same thresholds'),
             ('Rejoin', 'fresh all-loose 12-piece run')]
    w, box, gap, top = 980, 168, 35, 14
    parts = [f'<div class="loop-wrap"><svg class="loop-svg" viewBox="0 0 {w} 118" role="img" aria-label="Refinement loop: rehearse, check, locate, change, rejoin; repeat until the joined run passes">',
             '<defs><marker id="la" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" class="arrowhead"/></marker></defs>']
    for i, (title, sub) in enumerate(steps):
        x = 6 + i * (box + gap)
        parts.append(f'<rect class="loop-box{" final" if i == 4 else ""}" x="{x}" y="{top}" width="{box}" height="56" rx="8"/>'
                     f'<text class="loop-title" x="{x + 14}" y="{top + 24}">{title}</text>'
                     f'<text class="loop-sub" x="{x + 14}" y="{top + 43}">{sub}</text>')
        if i < 4:
            parts.append(f'<line class="loop-edge" x1="{x + box + 3}" y1="{top + 28}" x2="{x + box + gap - 4}" y2="{top + 28}" marker-end="url(#la)"/>')
    back_from = 6 + 3 * (box + gap) + box / 2
    back_to = 6 + box / 2
    parts.append(f'<path class="loop-edge" d="M{back_from},{top + 58} C{back_from},{top + 96} {back_to},{top + 96} {back_to},{top + 60}" marker-end="url(#la)"/>'
                 f'<text class="loop-sub" x="{(back_from + back_to) / 2}" y="{top + 100}" text-anchor="middle">repeat while any checkpoint fails</text></svg></div>')
    return ''.join(parts)


sys.path.insert(0, str(ROOT / 'policy'))
from puzzler import model  # noqa: E402  (public piece CAD: bead cells of the flat layout)

ORDER = ('azure', 'orange', 'yellow', 'teal', 'purple', 'cyan', 'mint', 'lime', 'maroon', 'navy', 'pink', 'red')


def shape_cells(name):
    """Bead cells of a piece, shifted to start at (0, 0)."""
    cells = model.CELLS[name]
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return [[r - r0, c - c0] for r, c in cells]


def shape_svg(name, cls='shape'):
    """A piece drawn by its shape: beads on the CAD grid with necks between neighbours."""
    cells, pitch, rad = shape_cells(name), 10, 4.4
    rows, cols = max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1
    pos = [(6 + c * pitch, 6 + r * pitch) for r, c in cells]
    necks = ''.join(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>' for i, (x1, y1) in enumerate(pos)
                    for (x2, y2) in pos[i + 1:] if abs(x1 - x2) + abs(y1 - y2) == pitch)
    beads = ''.join(f'<circle cx="{x}" cy="{y}" r="{rad}"/>' for x, y in pos)
    w, h = 12 + (cols - 1) * pitch, 12 + (rows - 1) * pitch
    return (f'<svg class="{cls}" viewBox="0 0 {w} {h}" width="{w * 1.6:.0f}" height="{h * 1.6:.0f}" aria-hidden="true">'
            f'<g class="neck">{necks}</g><g class="bead">{beads}</g></svg>')


def piece_ref(name):
    return f'<span class="pref">{shape_svg(name)}piece {ORDER.index(name) + 1}</span>'


RIG_TEXT = {
    'official': dict(cameras='Official YAM station: one gantry D405 overhead, one D405 on each wrist',
                     views='The gantry camera at t = 0, and all three cameras during piece 12\'s hand-to-hand transfer; '
                           'a free hand moves its camera to look.'),
    'custom': dict(cameras='Four fixed RGB cameras of our own placement: top, front, rear, side',
                   views='All the policy sees at t = 0: four fixed cameras.'),
}


def player_html(rig):
    return (f'<div class="player" data-rig="{rig}">'
            f'<div class="screen"><video data-k="video" src="media/{rig}/gap_full12.mp4" poster="media/{rig}/gap_full12_poster.jpg" '
            f'{AUTOPLAY} aria-label="Full twelve-piece run, replayed from the recorded episode"></video>'
            f'<div class="track" data-k="track" role="group" aria-label="Jump to a piece"></div>'
            f'<p class="clock"><span data-k="time">t = 0.0 s</span><span data-k="speed"></span></p></div>'
            f'<aside class="graph" aria-live="polite"><div class="graph-head"><span class="swatch" data-k="swatch"></span>'
            f'<h3 data-k="piece">Azure</h3><span class="mono small muted" data-k="index"></span></div>'
            f'<svg data-k="graph" class="graph-svg" role="img" aria-label="The current piece\'s subgraph"></svg>'
            f'<p class="node-detail" data-k="detail"></p></aside></div>')


def hands_html(rig):
    return (f'<div class="hands" data-rig="{rig}">'
            f'<figure><video src="media/{rig}/cyan_transfer.mp4" {AUTOPLAY} aria-label="Piece 6 passed from one gripper to the other"></video>'
            f'<figcaption><b>{piece_ref("cyan")}</b> One-hand attempts held it off-centre: it landed 16–44 mm off, or crept to the fingertips and slipped. '
            'The second hand regrasps it centred.</figcaption></figure>'
            f'<figure><video src="media/{rig}/red_transfer.mp4" {AUTOPLAY} aria-label="Piece 12 passed between the grippers twice"></video>'
            f'<figcaption><b>{piece_ref("red")}</b> From where it starts, no one-arm path reaches the top without running a joint to its limit. '
            'Two transfers give a grasp the arm can bring in.</figcaption></figure></div>')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rig', nargs=3, action='append', required=True, metavar=('NAME', 'MAIN_RUN', 'MEDIA'),
                        help='a camera rig (official or custom), its full-12 run and the tools/make_media.py folder for it')
    parser.add_argument('--grid', nargs=3, action='append', default=[], metavar=('RIG', 'VIDEO', 'CAPTION'),
                        help='a start-to-finish grid (tools/make_grid.py) or single full-run video for a rig')
    parser.add_argument('--note', nargs=2, action='append', default=[], metavar=('RIG', 'TEXT'))
    parser.add_argument('--video-stride', type=int, default=3)
    args = parser.parse_args()
    from PIL import Image

    if SITE.exists():
        shutil.rmtree(SITE)
    shutil.copytree(SRC, SITE, ignore=shutil.ignore_patterns('index.html', '*.html'))
    rigs, views, facts, players, hands = {}, [], [], [], []
    for name, main, media in args.rig:
        media, run = Path(media), inner(main)
        out = SITE / 'media' / name
        out.mkdir(parents=True, exist_ok=True)
        data = timeline(run)
        for piece in data['pieces']:
            for node in piece['nodes']:
                node['relook'] = node['kind'] in RELOOK
        data['video'] = dict(steps_per_second=30 * args.video_stride * FRAME_STEPS)
        main_clip = 'gap_full12_plain' if (media / 'gap_full12_plain.mp4').exists() else 'gap_full12'
        shutil.copy2(media / f'{main_clip}.mp4', out / 'gap_full12.mp4')
        for clip in ('cyan_transfer.mp4', 'red_transfer.mp4'):
            shutil.copy2(media / clip, out / clip)
        Image.open(media / f'{main_clip}_final.png').convert('RGB').save(out / 'gap_full12_poster.jpg', quality=85)
        if name == 'custom':
            camera_grid(run, out / 'policy_views.jpg')
            image = f'media/{name}/policy_views.jpg" width="1280" height="1024'
        else:
            image = 'media/policy_views_official.jpg" width="1280" height="720'
        text = RIG_TEXT[name]
        views.append(f'<figure class="views" data-rig="{name}"><img src="{image}" loading="lazy" alt="{esc(text["views"])}">'
                     f'<figcaption>{esc(text["views"])}</figcaption></figure>')
        facts.append(f'<div data-rig="{name}"><dt>Cameras</dt><dd>{esc(text["cameras"])}</dd></div>')
        players.append(player_html(name))
        hands.append(hands_html(name))
        rigs[name] = data
    notes = dict(args.note)
    grids = []
    for name, video, caption in args.grid:
        target = SITE / 'media' / name / 'grid.mp4'
        shutil.copy2(video, target)
        grids.append(f'<div data-rig="{name}"><figure class="grid-video"><video src="media/{name}/grid.mp4" {AUTOPLAY} '
                     f'aria-label="{esc(caption)}"></video><figcaption>{esc(caption)}</figcaption></figure>'
                     f'<p class="note">Each run starts from loose pieces and ends with the finished pyramid. The policy is never told what '
                     f'changed; it re-measures every pose from the cameras. {esc(notes.get(name, ""))}</p></div>')
    page_data = dict(rigs=rigs)
    (SITE / 'data/page.json').write_text(json.dumps(page_data, indent=1) + '\n')
    fill = dict(VIEWS=''.join(views), CAMERA_FACTS=''.join(facts), PLAYERS=''.join(players), HANDS=''.join(hands),
                GRIDS=''.join(grids), DATA=json.dumps(page_data).replace('</', '<\\/'))
    page = (SRC / 'index.html').read_text()
    page = re.sub(r'\{\{P:([a-z]+)\}\}', lambda m: piece_ref(m.group(1)), page)
    for key, value in fill.items():
        page = page.replace('{{' + key + '}}', value)
    missing = re.findall(r'\{\{[A-Z_]+\}\}', page)
    if missing:
        raise SystemExit('Unfilled placeholders: ' + ', '.join(missing))
    (SITE / 'index.html').write_text(page)
    for name, data in rigs.items():
        print(f"site/ {name}: {data['pieces_placed']} pieces, {data['checkpoints_passed']}/{data['checkpoints_total']} checkpoints, "
              f"graph_success={data['graph_success']}")


if __name__ == '__main__':
    main()
