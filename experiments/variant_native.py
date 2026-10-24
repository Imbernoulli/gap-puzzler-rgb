"""Layout-C full12 job under a declared RSS scene variant.

Copied into each generalization workspace as tools/variant_native.py. The
frozen v13 graph, policy, verifiers and thresholds are unchanged. Only the
episode's initial scene is compiled from a perturbed task.Variant (board pose,
loose-piece jitter, friction or appearance). The policy never receives these
numbers: PooledVisionGraph reads only mode/plan/seed, and every pose comes from
RGB fits.
"""
from dataclasses import asdict
import json
from pathlib import Path

from puzzler import model, task
from tools.layout_c_native import LAYOUT_NAME

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ('board_dx', 'board_dy', 'board_yaw', 'piece_jitter', 'piece_yaw', 'friction')


def variant_from(manifest):
    overrides = dict(manifest['scene_variant'])
    unknown = set(overrides) - {*NUMERIC, 'appearance'}
    if unknown:
        raise ValueError('Unsupported scene variant fields: ' + ', '.join(sorted(unknown)))
    return task.Variant(mode='pyramid', pyramid_plan='pyramid_robot6_plan.json', preset_pieces=(),
                        seed=manifest['seed'], appearance=overrides.pop('appearance', 'original'),
                        **{key: float(value) for key, value in overrides.items()})


def public_staging_c(original):
    stage = {name: (center.copy(), yaw, side) for name, (center, yaw, side) in original().items()}
    if set(stage) != set(model.CELLS):
        raise ValueError('Layout C requires all twelve public loose pieces')
    center, yaw, side = stage['lime']
    center[1] += .070
    stage['lime'] = (center, yaw, side)
    return stage


def jobs(manifest_path):
    from robosimstudio.specs import CaptureSpec, JobSpec
    manifest = json.loads(Path(manifest_path).read_text())
    layout = manifest.get('public_initial_layout', {})
    if (layout.get('name') != LAYOUT_NAME or manifest.get('initialized_prefix') != []
            or manifest.get('initial_ledgers_empty') is not True or manifest.get('device') != 'cuda:0'):
        raise ValueError('A fresh all-loose layout-C manifest on cuda:0 is required')
    variant = variant_from(manifest)
    original_staging = task.staging
    try:
        task.staging = lambda: public_staging_c(original_staging)
        spec = task.task_spec(variant, manifest['max_steps'], suite=manifest['suite'],
                              terminate_on_success=False, contact_capacity=4096)
    finally:
        task.staging = original_staging
    kwargs = dict(graph=manifest['graph'], output_dir=manifest['episode'], variant=asdict(variant),
                  max_steps=manifest['max_steps'], observation='rgb-shape',
                  independent_verifier_path=str(ROOT / 'verification/bimanual_independent.py'))
    return [JobSpec(name='pyramid_unified', task=spec.full_name,
        solution='tools.unified_native:UnifiedPyramidGraph', solution_kwargs=tuple(kwargs.items()),
        episodes=(0, 1), seed=variant.seed, max_steps=manifest['max_steps'], device='cuda:0',
        control_mode='joint_position', renderer='',
        capture=CaptureSpec(enabled=True, every=1, contacts=True, tool_spans=True))]
