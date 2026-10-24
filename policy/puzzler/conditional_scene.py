"""Opt-in conditional CAD occupancy, explicitly separate from RGB measurements.

Local benchmark predecessor names are public preconditions. Their goal-centered
occupancy is expanded by the declared bead-displacement precondition margin;
it is not a pose measurement. Current target, board and every unplaced object
must still be identified from fresh RGB. This does not prove hidden geometry.
"""
from copy import deepcopy
from functools import lru_cache
import json
from typing import Any, TypedDict

import numpy as np
from scipy.spatial.transform import Rotation

from . import model, visual_memory
from . import prefix_compatibility
from .vision_observer import goals


class ConditionalPlanningScene(TypedDict):
    contract: str
    target: str
    measurement: dict[str, Any]
    measured_names: list[str]
    conditional_names: list[str]
    occupancy: dict[str, dict[str, Any]]
    prefix_reference_board: list[float]
    margin_scope: str


class ConditionalSceneRefusal(RuntimeError):
    pass


def _plain(value):
    return json.loads(json.dumps(value, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else float(v), allow_nan=False))


@lru_cache(maxsize=12)
def local_sphere_cover(name):
    """Conservative sphere cover of the same public bead/neck CAD as assets.py."""
    beads = model.local_beads(name)
    centers, radii = [*beads], [model.RADIUS]*len(beads)
    for index, aa in enumerate(beads):
        for bb in beads[index+1:]:
            length = float(np.linalg.norm(bb-aa))
            if abs(length-model.PITCH) > 1.e-7:
                continue
            count = int(np.ceil(length/.0025))+1
            spacing = length/(count-1)
            # Every finite-cylinder point has a sample center within half a
            # spacing axially, and at most the public 4.5mm neck radius.
            radius = float(np.hypot(.0045, spacing/2))
            centers.extend(aa+(bb-aa)*u for u in np.linspace(0., 1., count))
            radii.extend([radius]*count)
    return np.asarray(centers), np.asarray(radii)


def bind_local_prefix(memory, observed, parameters):
    """Bind declared predecessors once to a current RGB board reference."""
    out = deepcopy(memory)
    declared = tuple(parameters.get('conditional_prefix', ()))
    if (not parameters.get('conditional_planning') or not declared
            or len(declared)!=len(set(declared)) or set(declared)-set(model.CELLS)
            or out.get('scope')!='conditional_local_benchmark'
            or set(out['local_prefix_assumptions'])!=set(declared)):
        raise ConditionalSceneRefusal('Conditional planning needs explicit local benchmark prefix assumptions')
    if 'board' not in observed.get('poses', {}):
        raise ConditionalSceneRefusal('Conditional planning requires a fresh RGB board')
    board = visual_memory._pose(observed['poses']['board']).tolist()
    margin=float(parameters.get('conditional_prefix_margin_m',.004))
    if not np.isfinite(margin) or not .004<=margin<=.012:
        raise ConditionalSceneRefusal('The public prefix margin is outside the declared development bounds')
    for name in declared:
        if (name in observed['poses'] and
                visual_memory.goal_error(name,observed,parameters)>=margin):
            record=observed.get('prefix_compatibility',{}).get(name,{})
            if not prefix_compatibility.validate_record(name,record,observed,parameters):
                raise ConditionalSceneRefusal('Current prefix point conflicts with its prior and no valid current-image compatibility witness is available: '+name)
    previous = out.get('local_prefix_reference_board')
    if previous is not None and visual_memory.shape_displacement('board', previous, board) >= .0015:
        raise ConditionalSceneRefusal('Current RGB board contradicts the local prefix reference')
    out.setdefault('local_prefix_reference_board', board)
    return out


def build(observed, memory, parameters, target) -> ConditionalPlanningScene:
    """Create separately attributed occupancy without changing observed poses."""
    prefix = set(parameters.get('conditional_prefix', ()))
    if target not in model.CELLS or target in prefix:
        raise ConditionalSceneRefusal('The active target must be outside the initialized prefix')
    memory = bind_local_prefix(memory, observed, parameters)
    if observed.get('observation_source')!='rgb_shape_geometry':
        raise ConditionalSceneRefusal('Conditional planning requires current shape RGB evidence')
    if not isinstance(observed.get('timestamp'), (int, float)) or not np.isfinite(observed['timestamp']) or not observed.get('frame_sha256'):
        raise ConditionalSceneRefusal('Conditional planning requires sensor time and image provenance')
    poses = observed.get('poses', {})
    # Receipt geometry is deliberately not an obstacle substitute in this
    # minimal phase: disturbance/sweep invalidation is not yet integrated.
    required = {'board', target, *(set(model.CELLS)-prefix)}
    missing = required-set(poses)
    if missing:
        raise ConditionalSceneRefusal('Fresh RGB is required for every unplaced object: '+str(sorted(missing)))
    prefix_margin = float(parameters.get('conditional_prefix_margin_m', .004))
    if not np.isfinite(prefix_margin) or not .004<=prefix_margin<=.012:
        raise ConditionalSceneRefusal('Conditional occupancy margins are outside the declared development bounds')
    board = visual_memory._pose(poses['board'])
    rb = Rotation.from_quat(board[3:]).as_matrix()
    public = goals(parameters.get('mode', 'flat'), parameters.get('pyramid_plan', 'pyramid_robot6_plan.json'))
    occupancy, conditional = {}, []
    for name in model.CELLS:
        if name in prefix:
            assumption = memory['local_prefix_assumptions'][name]
            if assumption['status']!='public_prefix_assumed_local_only':
                raise ConditionalSceneRefusal('The local prefix assumption was invalidated: '+name)
            center, rotation = public[name]
            options = [[*(board[:3]+rb@center), *Rotation.from_matrix(rb@rotation).as_quat()]]
            if name in poses:
                measured=visual_memory._pose(poses[name]).tolist()
                # bind_local_prefix already checked any outside point using
                # the optional independent image witness. Retain that point
                # in the union; never replace it with the public hypothesis.
                if visual_memory.shape_displacement(name,options[0],measured)>1.e-9:
                    options.append(measured)
            source, margin = 'public_prefix_assumed_local_only', prefix_margin
            conditional.append(name)
        elif name in poses:
            options=[visual_memory._pose(poses[name]).tolist()]
            source,margin='current_rgb_measurement',0.
        else:
            raise ConditionalSceneRefusal('Unknown occupancy cannot become free space: '+name)
        occupancy[name] = dict(source=source, pose_hypotheses=options, inflation_m=margin,
                               shape='public_beads_and_4.5mm_connector_sphere_cover')
    return _plain(dict(contract='conditional_local_prefix_occupancy_v1', target=target,
        measurement={key: observed[key] for key in ('poses', 'timestamp', 'frame_sha256', 'observation_source','prefix_compatibility') if key in observed},
        measured_names=sorted(poses), conditional_names=sorted(conditional), occupancy=occupancy,
        prefix_reference_board=memory['local_prefix_reference_board'],
        margin_scope='Prefix 4mm is the declared local scene precondition, externally checked against actual native initialization and protected-prefix tail; current visibility never removes that envelope. Intended contact can disturb it, so this remains conditional, not covariance or a guarantee. Every nonprefix object is fresh; planning from remembered receipts is disabled.'))


def matches_measurement(scene, observed):
    expected = scene['measurement']
    return all(_plain(expected.get(key))==_plain(observed.get(key))
               for key in ('poses', 'timestamp', 'frame_sha256', 'observation_source','prefix_compatibility'))


def spheres(scene, *, exclude=()):
    centers, radii = [], []
    for name, item in scene['occupancy'].items():
        if name in exclude:
            continue
        local, sizes = local_sphere_cover(name)
        for option in item['pose_hypotheses']:
            pose = visual_memory._pose(option)
            centers.extend(local@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3])
            radii.extend(sizes+item['inflation_m'])
    return np.asarray(centers, dtype=float).reshape(-1, 3), np.asarray(radii, dtype=float)


def hand_clearance(position, rotation, width, scene, *, exclude=(), plane_z=None):
    """Analytic hand-vs-union sphere covers plus the existing static CAD query."""
    from .gripper_geometry import clearance, scene_clearance
    centers, radii = spheres(scene, exclude=exclude)
    board = np.asarray(scene['measurement']['poses'].get('board',scene.get('board_for_planning',{}).get('pose')))
    actual = clearance(position, rotation, width, centers, radius=radii)
    fixed = scene_clearance(position, rotation, width, [], table_z=float(board[2]-.003),
                            plane_z=plane_z, board=board)
    return actual if actual['clearance_m']<fixed['clearance_m'] else fixed
