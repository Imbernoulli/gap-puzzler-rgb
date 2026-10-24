"""Build an isolated workspace from policy/ and run one fresh full-12 episode.

  python experiments/run.py prepare NAME --variant nominal
  python experiments/run.py launch NAME --gpu 4 --cpu 256
  python experiments/run.py status [NAME ...]

The workspace is a copy of policy/ plus links to the pinned RSS/GaP checkouts.
A scene variant changes only the initial scene (board pose, loose-piece
jitter, friction, colors); the policy never receives those numbers.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / 'policy'
RUNS = REPO / 'runs'
UPSTREAM = Path('/srv/home/bohanlyu/gap-puzzler-web')
RSS_ASSETS = UPSTREAM / 'runs/pyramid_full12_bimanual_layout_c_v13/workspace/third_party/robosimstudio/assets'
RUN = 'full12_original_order_layout_c_v10'  # inner name kept for the runner's snapshot wiring
GRAPH = 'graphs/pyramid_original_order_layout_c_v10'

# Seeds for jittered layouts were drawn by rejection sampling (no interpenetrating
# loose pieces, at least 1.5 mm bead clearance) and are fixed here.
VARIANTS = {
    'nominal': dict(seed=0, scene_variant={}, label='Nominal layout C'),
    'board_a': dict(seed=0, scene_variant=dict(board_dx=.006, board_dy=-.006, board_yaw=.05),
                    label='Case +6 mm x, −6 mm y, +2.9° yaw'),
    'board_b': dict(seed=0, scene_variant=dict(board_dx=-.006, board_dy=.006, board_yaw=-.05),
                    label='Case −6 mm x, +6 mm y, −2.9° yaw'),
    'jitter_a': dict(seed=4, scene_variant=dict(piece_jitter=.005, piece_yaw=.07),
                     label='Loose pieces ±5 mm, ±4° (seed 4)'),
    'jitter_b': dict(seed=104, scene_variant=dict(piece_jitter=.005, piece_yaw=.07),
                     label='Loose pieces ±5 mm, ±4° (seed 104)'),
    'friction_050': dict(seed=0, scene_variant=dict(friction=.50), label='Friction μ = 0.50'),
    'friction_080': dict(seed=0, scene_variant=dict(friction=.80), label='Friction μ = 0.80'),
    'combined': dict(seed=201, scene_variant=dict(board_dx=-.004, board_dy=-.004, board_yaw=.03,
                     piece_jitter=.003, piece_yaw=.035, friction=.60),
                     label='Case, pieces and friction together (seed 201)'),
    'shuffled': dict(seed=0, scene_variant=dict(appearance='shuffled'), label='Piece colors shuffled'),
}
# Loose pieces re-placed anywhere on their own arm's side and turned up to ±45°.
VARIANTS.update({f'scatter_{k}': dict(seed=0, scene_variant=dict(scatter_seed=k), label=f'Scattered layout {k}')
                 for k in range(1, 13)})


def workspace(name):
    return RUNS / name / 'workspace'


def prepare(name, variant):
    spec = VARIANTS[variant]
    ws = workspace(name)
    if ws.exists():
        raise SystemExit(f'{ws} already exists')
    ws.mkdir(parents=True)
    subprocess.run(['rsync', '-a', '--exclude', '__pycache__', f'{POLICY}/', f'{ws}/'], check=True)
    (ws / 'tools/variant_native.py').write_bytes((REPO / 'experiments/variant_native.py').read_bytes())
    third = ws / 'third_party'
    (third / 'robosimstudio').mkdir(parents=True)
    subprocess.run(['rsync', '-a', f'{RSS_ASSETS}/', f'{third}/robosimstudio/assets/'], check=True)
    (third / 'robosimstudio/.venv').symlink_to(UPSTREAM / 'third_party/robosimstudio/.venv')
    for package in ('graph-as-policy', 'graph-as-policy-runtime', 'open-robot-skills'):
        (third / package).symlink_to(UPSTREAM / 'third_party' / package)
    workflow = ws / GRAPH / 'workflow.json'
    document = json.loads(workflow.read_text())
    document['meta']['skill_registry_paths'] = [str(ws / 'registry_modular'), str(third / 'open-robot-skills')]
    workflow.write_text(json.dumps(document, indent=2) + '\n')
    run = ws / 'runs' / RUN
    run.mkdir(parents=True)
    manifest = json.loads((POLICY.parent / 'experiments/manifest_template.json').read_text())
    manifest.update(graph=str(ws / GRAPH), episode=str(run / 'episode'), native_root=str(run / 'native_pool'),
                    jobfile=str(run / 'native_jobs.py'), suite=f'gap_puzzler_rgb_{name}',
                    seed=spec['seed'], scene_variant=spec['scene_variant'], variant=variant,
                    variant_label=spec['label'], native_run_started=False, physical_gpu_mask=None)
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (run / 'native_jobs.py').write_text(
        'from tools.variant_native import jobs\nJOBS = jobs(' + repr(str(run / 'manifest.json')) + ')\n')
    print(f'{name}: prepared {variant} at {ws}')


def launch(name, gpu, cpu):
    ws = workspace(name)
    log = RUNS / name / 'native.log'
    command = (f'taskset -c {cpu}-{cpu + 7} env RSS_WORKSPACE={ws} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 '
               f'{UPSTREAM}/tools/rss_gpu.sh {gpu} -u -m tools.launch_unified run-prepared '
               f'--manifest {ws}/runs/{RUN}/manifest.json --candidate-integration')
    process = subprocess.Popen(['bash', '-c', f'{command} > {log} 2>&1'], start_new_session=True)
    (RUNS / name / 'launch.json').write_text(json.dumps(dict(pid=process.pid, gpu=gpu, cpu=cpu, command=command), indent=2))
    print(f'{name}: gpu {gpu}, cpus {cpu}-{cpu + 7}, pid {process.pid}')


def status(name):
    run = workspace(name) / 'runs' / RUN
    rows = []
    log = run / 'independent_verification/checkpoint_results.jsonl'
    if log.exists():
        rows = [json.loads(line) for line in log.read_text().splitlines()]
    placed = [r['piece'] for r in rows if r.get('phase') == 'handoff' and r.get('passed')]
    out = dict(name=name, checkpoints=len(rows), failed=[f"{r.get('piece')}:{r['phase']}" for r in rows if not r.get('passed')],
               placed=len(placed))
    result = run / 'episode/result.json'
    if result.exists():
        r = json.loads(result.read_text())
        out.update(done=True, rss_success=r['rss_success'], graph_success=r['graph_success'],
                   sim_time_s=round(r['sim_time_s'], 1), error=str(r.get('error'))[-240:] if r.get('error') else None,
                   audit=r.get('independent_audit'))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['prepare', 'launch', 'status'])
    parser.add_argument('names', nargs='*')
    parser.add_argument('--variant', default='nominal', choices=sorted(VARIANTS))
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        for name in args.names:
            prepare(name, args.variant)
    elif args.action == 'launch':
        for i, name in enumerate(args.names):
            launch(name, args.gpu, args.cpu + 8 * i)
    else:
        for name in args.names or sorted(p.name for p in RUNS.iterdir() if p.is_dir()):
            print(json.dumps(status(name)))
