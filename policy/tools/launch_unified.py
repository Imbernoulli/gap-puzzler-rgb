"""Prepare a reviewable fresh native candidate; root controls the actual launch."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / 'graphs/pyramid_original_order_layout_c_v10'


def validate(graph):
    subprocess.run([str(ROOT/'tools/rss.sh'), str(ROOT/'tools/validate_staged_graph.py'), str(graph)],
                   cwd=ROOT, check=True)


def _fresh_graph(graph):
    document = json.loads((graph/'workflow.json').read_text())
    from tools.build_unified_graph import ORDERS
    if (document['meta']['ordered_stages'] != list(ORDERS['original_order'])
            or document['meta'].get('initialized_prefix')
            or not document['meta'].get('full_assembly_eligible')):
        raise ValueError('Full candidate requires fresh original all12 order with no initialized prefix')
    literals = [node['inputs']['state'] for node in document['nodes'].values()
                if isinstance(node.get('inputs', {}).get('state'), dict)
                and '$ref' not in node['inputs']['state']]
    if len(literals) != 1:
        raise ValueError('Exactly one initial state is required')
    first, = literals
    for key in ['executed_placements', 'initialized_prefix', 'execution_tokens', 'rgb_history']:
        if first.get(key):
            raise ValueError('Fresh candidate initial state must have empty '+key)
    if first.get('memory', {}).get('placements'):
        raise ValueError('Fresh candidate cannot begin with an earned receipt')
    return document


def prepare(graph, output, *, seed=0, max_steps=60000, device='cuda:0'):
    if device != 'cuda:0':
        raise ValueError('Explicit cuda:0 is required')
    graph = Path(graph).resolve()
    _fresh_graph(graph)
    validate(graph)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output/'manifest.json'
    readiness = json.loads((ROOT/'integration_status.json').read_text())
    manifest = dict(schema='puzzler.unified_fresh_native.v7',
        initialization='fresh_canonical_all12_loose', graph=str(graph),
        episode=str(output/'episode'), native_root=str(output/'native_pool'),
        jobfile=str(output/'native_jobs.py'), suite='unified_'+output.name,
        seed=seed, max_steps=max_steps, device=device, contact_capacity=4096,
        physical_gpu_mask=None, rgb_renderer='RGBSensor tiled CUDA renderer',
        ordered_stages=json.loads((graph/'params.json').read_text())['assembly_order'],
        piece_phase_count=sum(len(row) for row in _fresh_graph(graph)['meta']['phase_schedule'].values()),
        initial_ledgers_empty=True, initialized_prefix=[], terminate_on_success=False,
        qualification_pending_at_preparation=readiness.get('pending', []),
        candidate_integration=False, native_run_started=False,
        full12_acceptance=False, flat_acceptance=False,
        complete_native_logs_required_for_capacity_audit=True,
        launch_policy='Continuous5–7 and10–12 placement must both qualify before full12 launch. Candidate execution never implies acceptance.')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    Path(manifest['jobfile']).write_text('from tools.unified_native import jobs\nJOBS = jobs('+repr(str(manifest_path))+')\n')
    return manifest_path


def run_prepared(manifest_path, *, candidate_integration=False):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('native_run_started') or Path(manifest['episode']).exists():
        raise RuntimeError('Prepared native candidate has already started; use a new output directory')
    graph = Path(manifest['graph'])
    _fresh_graph(graph)
    validate(graph)
    readiness = json.loads((ROOT/'integration_status.json').read_text())
    segments = readiness.get('required_segment_qualifications', {})
    if not all(segments.get(name) is True for name in ('continuous_5_7_placed', 'continuous_10_12_placed')):
        raise RuntimeError('Full12 remains prepared until continuous5–7 and10–12 both complete placement; warm or local checkpoints do not qualify')
    pending = readiness.get('pending', [])
    if pending and not candidate_integration:
        raise RuntimeError('Components still under qualification; full12 not started: '+'; '.join(pending))
    # This flag is an explicit owner launch choice, not an automatic readiness
    # inference. Root checks the first Cyan physical handoff before invoking it.
    manifest.update(candidate_integration=bool(candidate_integration),
        qualification_pending_at_launch=pending,
        native_run_started=True, physical_gpu_mask=os.environ.get('CUDA_VISIBLE_DEVICES'),
        full12_acceptance=False, flat_acceptance=False)
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    from tools.unified_native import run
    return run(path)


def launch(graph, output, *, seed=0, max_steps=60000, device='cuda:0'):
    path = prepare(graph, output, seed=seed, max_steps=max_steps, device=device)
    return run_prepared(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'prepare', 'run', 'run-prepared'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=60000)
    parser.add_argument('--device', choices=['cuda:0'], default='cuda:0')
    parser.add_argument('--candidate-integration', action='store_true',
        help='Root-owned diagnostic launch after actual Cyan handoff qualification; pending later module qualifications remain explicit')
    args = parser.parse_args()
    if args.action == 'validate':
        validate(GRAPH)
    elif args.action == 'run-prepared':
        if args.manifest is None:
            parser.error('--manifest is required')
        print(run_prepared(args.manifest, candidate_integration=args.candidate_integration))
    else:
        if args.output is None:
            parser.error('--output is required')
        if args.action == 'prepare':
            print(prepare(GRAPH, args.output, seed=args.seed, max_steps=args.max_steps, device=args.device))
        else:
            path = prepare(GRAPH, args.output, seed=args.seed, max_steps=args.max_steps, device=args.device)
            print(run_prepared(path, candidate_integration=args.candidate_integration))
