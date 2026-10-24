"""Render every clip and still the site uses from one recorded full-12 episode.

  python tools/make_media.py RUN_DIR OUT_DIR --gpu N

RUN_DIR is runs/<name> (or the inner run directory). Clip ranges come from the
run's own checkpoint log, so the two-hand clips start and end where the graph's
transfer nodes did. Rendering assigns recorded poses; nothing is re-simulated.
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path('/srv/home/bohanlyu/gap-puzzler-web')
RUN = 'full12_original_order_layout_c_v10'
CAMERAS = {
    'closeup': '0.80,0.0,1.10,0.465,0.0,0.775',
    'cyan_transfer': '1.22,-0.28,1.32,0.62,0.14,0.86',
    'red_transfer': '1.02,-0.58,1.42,0.43,-0.02,0.90',
}
# A viewer camera turned about the case to roughly match the hand-held frame used on the page.
REAL_MATCH = 'real_match:0.649,-0.106,1.114,0.465,0.0,0.775,38'


def inner(run):
    run = Path(run).resolve()
    return run / 'workspace/runs' / RUN if (run / 'workspace/runs' / RUN).exists() else run


def step_of(rows, piece, phase):
    return next(r['native_step'] for r in rows if r.get('piece') == piece and r.get('phase') == phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('out')
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    run = inner(args.run)
    workspace = run.parents[1]
    episode = run / 'episode'
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (run / 'independent_verification/checkpoint_results.jsonl').read_text().splitlines()]

    def render(script, *extra):
        env = dict(__import__('os').environ, RSS_WORKSPACE=str(workspace))
        subprocess.run([str(UPSTREAM / 'tools/rss_gpu.sh'), str(args.gpu), str(ROOT / 'tools' / script), str(episode), *extra],
                       cwd=workspace, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    render('render_gap.py', str(out / 'gap_full12.mp4'), '--stride', '3')
    navy = next(r['native_step'] for r in rows if r.get('piece') == 'navy')
    render('render_gap.py', str(out / 'rss_closeup.mp4'), '--start-step', str(navy), '--stride', '2',
           '--camera', CAMERAS['closeup'], '--plain')
    render('render_gap.py', str(out / 'cyan_transfer.mp4'), '--start-step', str(step_of(rows, 'cyan', 'grasp')),
           '--end-step', str(step_of(rows, 'cyan', 'transport')), '--stride', '3', '--camera', CAMERAS['cyan_transfer'], '--plain')
    render('render_gap.py', str(out / 'red_transfer.mp4'), '--start-step', str(step_of(rows, 'red', 'grasp')),
           '--end-step', str(step_of(rows, 'red', 'return_handoff_receiver_bound')), '--stride', '3',
           '--camera', CAMERAS['red_transfer'], '--plain')
    render('render_still.py', str(out / 'stills'), '-1', REAL_MATCH)
    print(out)


if __name__ == '__main__':
    main()
