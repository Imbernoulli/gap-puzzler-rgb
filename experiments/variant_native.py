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

import numpy as np

from puzzler import model, task
from tools.layout_c_native import LAYOUT_NAME

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ('board_dx', 'board_dy', 'board_yaw', 'piece_jitter', 'piece_yaw', 'friction')


def variant_from(manifest):
    overrides = dict(manifest['scene_variant'])
    overrides.pop('scatter_seed', None)
    overrides.pop('scatter_spacing', None)
    unknown = set(overrides) - {*NUMERIC, 'appearance'}
    if unknown:
        raise ValueError('Unsupported scene variant fields: ' + ', '.join(sorted(unknown)))
    return task.Variant(mode='pyramid', pyramid_plan='pyramid_robot6_plan.json', preset_pieces=(),
                        seed=manifest['seed'], appearance=overrides.pop('appearance', 'original'),
                        **{key: float(value) for key, value in overrides.items()})


def scattered_by_side(original, seed, yaw_range=np.pi / 4, spacing=.030):
    """Re-place every loose piece anywhere on its own arm's side, turned up to ±45°.

    Rejection-sampled like task.scattered_staging: beads stay inside the side
    region and every loose piece starts at least `spacing` from the others.
    """
    rng = np.random.default_rng(seed)
    stage = original()
    for restart in range(40):
        occupied, result = [], {}
        for name in rng.permutation(list(model.CELLS)):
            _, yaw0, side = stage[name]
            for attempt in range(2000):
                yaw = float(yaw0 + rng.uniform(-yaw_range, yaw_range))
                center = np.array([rng.uniform(-.28, .23), rng.uniform(.235, .34) * (1 if side == 0 else -1), model.RADIUS + .001])
                points = model.local_beads(name) @ model.rotation(yaw).T + center
                if (np.abs(points[:, 1]).max() > .39 or np.abs(points[:, 1]).min() < .19
                        or points[:, 0].min() < -.31 or points[:, 0].max() > .29):
                    continue
                if occupied and np.linalg.norm(points[:, None] - np.concatenate(occupied)[None, :], axis=2).min() < spacing:
                    continue
                occupied.append(points)
                result[name] = (center, yaw, side)
                break
            else:
                break
        if len(result) == len(model.CELLS):
            return result
    raise RuntimeError('Could not sample a non-overlapping side-preserving layout')


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
    scatter = manifest['scene_variant'].get('scatter_seed')
    spacing = manifest['scene_variant'].get('scatter_spacing', .030)
    try:
        task.staging = ((lambda: scattered_by_side(original_staging, scatter, spacing=spacing)) if scatter is not None
                        else (lambda: public_staging_c(original_staging)))
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
