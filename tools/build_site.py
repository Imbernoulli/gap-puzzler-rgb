"""Build site/ from recorded runs.

  python tools/build_site.py --main runs/nominal_fix1 [--before DIR ...] [--after DIR ...]

--main is the full-12 episode shown in the player. --before/--after name
variant run directories (each containing workspace/runs/<RUN>/ or that inner
run directory itself). Every number on the page is read here from the
recorded result, checkpoint log and workflow; nothing is typed by hand.
"""
import argparse
import json
import re
import shutil
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


def repeats_html(before, after):
    """One row summarising every nominal repeat."""
    return (f'<tr><th scope="row">Layout C, repeats</th><td class="mono small">as designed</td>'
            f'{runs_cell(before)}{runs_cell(after)}</tr>')


def variants_html(variants, repeats=''):
    head = '<thead><tr><th scope="col">Scene</th><th scope="col">What changed</th><th scope="col">v13 policy (one run)</th><th scope="col">This work (all runs)</th></tr></thead>'
    body = repeats + ''.join(f'<tr><th scope="row">{esc(v["label"])}</th><td class="mono small">{esc(v["change"])}</td>'
                   f'{outcome_cell(v["before"])}{runs_cell(v["after"])}</tr>' for v in variants)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--main', required=True)
    parser.add_argument('--media', required=True, help='output folder of tools/make_media.py for --main')
    parser.add_argument('--video-stride', type=int, default=3)
    parser.add_argument('--grid', help='grid video (tools/make_grid.py) for the generalization section')
    parser.add_argument('--grid-caption', default='')
    parser.add_argument('--before', nargs='*', default=[])
    parser.add_argument('--after', nargs='*', default=[])
    parser.add_argument('--repeats-before', nargs='*', default=[])
    parser.add_argument('--note', default='', help='which code produced each column')
    parser.add_argument('--repeats-after', nargs='*', default=[])
    args = parser.parse_args()
    from PIL import Image

    if SITE.exists():
        shutil.rmtree(SITE)
    shutil.copytree(SRC, SITE, ignore=shutil.ignore_patterns('index.html', '*.html'))
    media = Path(args.media)
    main_run = inner(args.main)
    data = timeline(main_run)
    for piece in data['pieces']:
        for node in piece['nodes']:
            node['relook'] = node['kind'] in RELOOK
    data['video'] = dict(src='media/gap_full12.mp4', poster='media/gap_full12_poster.jpg',
                         steps_per_second=30 * args.video_stride * FRAME_STEPS)
    for name in ('gap_full12.mp4', 'cyan_transfer.mp4', 'red_transfer.mp4'):
        shutil.copy2(media / name, SITE / 'media' / name)
    Image.open(media / 'gap_full12_final.png').convert('RGB').save(SITE / 'media/gap_full12_poster.jpg', quality=85)
    Image.open(media / 'stills/real_match.png').convert('RGB').resize((960, 540), Image.LANCZOS).save(
        SITE / 'media/rss_sim.jpg', quality=88)
    camera_grid(main_run, SITE / 'media/policy_views_t0.jpg')
    grid = ''
    if args.grid:
        shutil.copy2(args.grid, SITE / 'media/generalization_grid.mp4')
        grid = (f'<figure class="grid-video"><video src="media/generalization_grid.mp4" muted loop playsinline controls '
                f'preload="metadata" aria-label="{esc(args.grid_caption)}"></video><figcaption>{esc(args.grid_caption)}</figcaption></figure>')

    before = {r['variant']: r for r in map(variant_row, args.before)}
    after = {}
    for row in map(variant_row, args.after):
        after.setdefault(row['variant'], []).append(row)
    names = [n for n in ('nominal', 'board_a', 'board_b', 'jitter_a', 'jitter_b', 'friction_050', 'friction_080',
                         'combined', 'shuffled') if n in before or n in after]
    variants = [dict(variant=n, label=variant_label(before.get(n) or after[n][0]),
                     change=describe((before.get(n) or after[n][0])['scene'] or {}),
                     before=before.get(n), after=after.get(n, [])) for n in names]
    data['variants'] = variants
    (SITE / 'data/page.json').write_text(json.dumps(data, indent=1) + '\n')
    witness = json.loads((SRC / 'data/v13_completion_witness.json').read_text())

    fill = dict(
        PIECES_PLACED=str(data['pieces_placed']), CHECKPOINTS=f"{data['checkpoints_passed']}/{data['checkpoints_total']}",
        SIM_TIME=f"{data['sim_time_s']:.0f}", AUDIT_MAX=f"{data['audit_max_mm']:.1f}" if data['audit_max_mm'] else '—',
        VARIANT_ROWS=variants_html(variants, repeats_html([variant_row(r) for r in args.repeats_before],
                                                          [variant_row(r) for r in args.repeats_after])),
        GRID=grid, LOOP_SVG=loop_svg(), GEN_NOTE=esc(args.note),
        GENERALIZATION_CASES=(SRC / 'generalization_cases.html').read_text() if (SRC / 'generalization_cases.html').exists() else '',
        DRIFT_SVG=drift_svg(SRC / 'data/v13_board_drift.json'),
        WITNESS=f"{witness['complete']['matched']}/{witness['complete']['decisive']} views",
        DATA=json.dumps(data).replace('</', '<\\/'))
    page = (SRC / 'index.html').read_text()
    for key, value in fill.items():
        page = page.replace('{{' + key + '}}', value)
    missing = re.findall(r'\{\{[A-Z_]+\}\}', page)
    if missing:
        raise SystemExit('Unfilled placeholders: ' + ', '.join(missing))
    (SITE / 'index.html').write_text(page)
    print(f"site/: {data['pieces_placed']} pieces, {data['checkpoints_passed']}/{data['checkpoints_total']} checkpoints, "
          f"{len(variants)} variants, graph_success={data['graph_success']}")


if __name__ == '__main__':
    main()
