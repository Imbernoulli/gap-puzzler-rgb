"""Explicit visual placement receipts, never a source of fresh measurements.

This experimental ledger is carried in native graph state. It does not hold an
environment, use simulator history, fill missing observed poses, or enable
planning from memory. Persistence while hidden is a conditional label only.
"""
from copy import deepcopy
import json
from typing import Any, TypedDict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from . import model
from .vision_observer import goals


class PlacementReceipt(TypedDict):
    piece: str
    source: str
    before: dict[str, Any]
    after: dict[str, Any]
    released_tcp_position: list[float]
    maximum_displacement_m: float
    goal_error_m: float
    hand_gap_m: float


class ReceiptTrack(TypedDict):
    status: str
    receipt: PlacementReceipt
    last_measured_pose: list[float]
    last_measured_timestamp: float
    last_frame_sha256: dict[str, str]
    invalidation_reason: str


class PrefixAssumption(TypedDict):
    piece: str
    status: str
    origin: str


class VisualMemory(TypedDict):
    schema: str
    scope: str
    placements: dict[str, ReceiptTrack]
    local_prefix_assumptions: dict[str, PrefixAssumption]
    last_timestamp: float
    last_frame_sha256: dict[str, str]
    planning_from_memory_enabled: bool


class MemoryRefusal(RuntimeError):
    pass


def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def new_memory(prefix=(), *, conditional_local_benchmark=False) -> VisualMemory:
    names = tuple(prefix)
    if len(names) != len(set(names)) or set(names)-set(model.CELLS):
        raise MemoryRefusal('Invalid local benchmark predecessor names')
    if names and not conditional_local_benchmark:
        raise MemoryRefusal('Full rollouts must start with an empty visual ledger')
    return dict(schema='puzzler.visual_memory.v1',
        scope='conditional_local_benchmark' if conditional_local_benchmark else 'continuous_visual_rollout',
        placements={}, local_prefix_assumptions={name: dict(piece=name,
            status='public_prefix_assumed_local_only', origin='declared_stage_precondition') for name in names},
        last_timestamp=-1., last_frame_sha256={}, planning_from_memory_enabled=False)


def _pose(value):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (7,) or not np.isfinite(pose).all()
            or abs(float(np.linalg.norm(pose[3:]))-1.) > 1.e-4):
        raise MemoryRefusal('A visual pose must be finite and normalized')
    return pose


def _cloud(name, value):
    pose = _pose(value)
    if name == 'board':
        points = np.array([[.0275+sx*.110, sy*.1575, .013]
                           for sx in (-1, 1) for sy in (-1, 1)])
    else:
        points = model.local_beads(name)
    return points@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]


def shape_displacement(name, before, after):
    aa, bb = _cloud(name, before), _cloud(name, after)
    if name == 'board':
        return float(np.linalg.norm(aa-bb, axis=1).max())
    distances = np.linalg.norm(aa[:, None]-bb[None, :], axis=2)
    ii, jj = linear_sum_assignment(distances)
    return float(distances[ii, jj].max())


def _observation(observed, piece):
    if observed.get('observation_source') not in ('rgb_shape_geometry','calibrated_rgb_cad_seeded'):
        raise MemoryRefusal('Placement receipts require the declared shape RGB observation source')
    timestamp = observed.get('timestamp')
    hashes = observed.get('frame_sha256')
    if (not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp)
            or not isinstance(hashes, dict) or not hashes
            or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in hashes.items())):
        raise MemoryRefusal('Placement receipts require finite RGB time and pixel provenance')
    poses = observed.get('poses', {})
    if not {'board', piece}.issubset(poses):
        raise MemoryRefusal('Receipt target and board must be currently measured')
    return dict(timestamp=float(timestamp), frame_sha256=dict(hashes),
                observation_source=observed['observation_source'],
                poses={name: _pose(poses[name]).tolist() for name in ('board', piece)})


def goal_error(piece, observed, parameters):
    body = _pose(observed['poses'][piece])
    board = _pose(observed['poses']['board'])
    center, rotation = goals(parameters.get('mode', 'flat'),
                             parameters.get('pyramid_plan', 'pyramid_robot6_plan.json'))[piece]
    desired = (model.local_beads(piece)@rotation.T+center)@Rotation.from_quat(board[3:]).as_matrix().T+board[:3]
    distances = np.linalg.norm(_cloud(piece, body)[:, None]-desired[None, :], axis=2)
    ii, jj = linear_sum_assignment(distances)
    return float(distances[ii, jj].max())


def release_receipt(piece, before, after, tcp, parameters) -> PlacementReceipt:
    """Recompute a receipt solely from a real release pair and encoder TCP."""
    if piece not in model.CELLS:
        raise MemoryRefusal('Unknown receipt piece')
    first, second = _observation(before, piece), _observation(after, piece)
    if second['timestamp']-first['timestamp'] < 1.-1.e-6:
        raise MemoryRefusal('Release evidence requires the existing one-second stability interval')
    displacement = max(shape_displacement(n, first['poses'][n], second['poses'][n])
                       for n in ('board', piece))
    encoder = np.asarray(tcp, dtype=float)
    if encoder.shape != (3,) or not np.isfinite(encoder).all():
        raise MemoryRefusal('Release encoder position is invalid')
    gap = float(np.linalg.norm(_cloud(piece, second['poses'][piece])-encoder, axis=1).min())
    error = goal_error(piece, second, parameters)
    if displacement >= .0015 or gap <= .055 or error >= .003:
        raise MemoryRefusal('A stable, separated, visually placed release is required')
    return dict(piece=piece, source='rgb_release_pair', before=first, after=second,
                released_tcp_position=encoder.tolist(), maximum_displacement_m=displacement,
                goal_error_m=error, hand_gap_m=gap)


def mint_placement(memory, receipt, current, parameters) -> VisualMemory:
    """Only the release-success confirmation calls this operation."""
    piece = receipt['piece']
    checked = release_receipt(piece, receipt['before'], receipt['after'],
                              receipt['released_tcp_position'], parameters)
    if _plain(receipt) != _plain(checked) or _observation(current, piece) != checked['after']:
        raise MemoryRefusal('The pending receipt differs from the confirmed RGB release')
    out = deepcopy(memory)
    out['placements'][piece] = dict(status='measured_now', receipt=checked,
        last_measured_pose=checked['after']['poses'][piece],
        last_measured_timestamp=checked['after']['timestamp'],
        last_frame_sha256=checked['after']['frame_sha256'], invalidation_reason='')
    return out


def invalidate_disturbance(memory, pieces, reason):
    """Explicit hook for future conservative swept-volume/support reasoning.

    The current branch keeps all original fresh-scene gates and does not use
    remembered geometry for actions. Before enabling that use, motion/support
    disturbance calls must be integrated; this API alone does not prove them.
    """
    if not reason:
        raise MemoryRefusal('A disturbance needs an explicit reason')
    out = deepcopy(memory)
    for piece in pieces:
        if piece in out['placements']:
            out['placements'][piece].update(status='unknown_or_conflicted', invalidation_reason=reason)
        if piece in out['local_prefix_assumptions']:
            out['local_prefix_assumptions'][piece]['status'] = 'unknown_or_conflicted'
    return out


def observe(memory, observed, parameters) -> VisualMemory:
    """Update receipt status without changing the current observation."""
    out = deepcopy(memory)
    timestamp = observed.get('timestamp')
    hashes = observed.get('frame_sha256', {})
    if not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp) or not hashes:
        raise MemoryRefusal('Visual memory updates need recorded current RGB')
    if timestamp < out['last_timestamp']:
        raise MemoryRefusal('Visual memory cannot move backwards in sensor time')
    if timestamp == out['last_timestamp'] and hashes != out['last_frame_sha256']:
        raise MemoryRefusal('One sensor time cannot name different image evidence')
    poses = observed.get('poses', {})
    for piece, track in out['placements'].items():
        receipt = track['receipt']
        if track['status'] == 'unknown_or_conflicted':
            continue
        if 'board' in poses and shape_displacement('board', receipt['after']['poses']['board'], poses['board']) >= .0015:
            track.update(status='unknown_or_conflicted', invalidation_reason='current RGB board contradicts the release reference')
            continue
        if piece not in poses:
            track['status'] = 'remembered_static_conditional'
            continue
        pose = _pose(poses[piece]).tolist()
        moved = shape_displacement(piece, receipt['after']['poses'][piece], pose)
        outside = 'board' in poses and goal_error(piece, observed, parameters) >= .003
        track.update(last_measured_pose=pose, last_measured_timestamp=float(timestamp), last_frame_sha256=dict(hashes))
        if moved >= .003 or outside:
            track.update(status='unknown_or_conflicted', invalidation_reason='current RGB contradicts the accepted placement')
        else:
            track['status'] = 'measured_now'
    out.update(last_timestamp=float(timestamp), last_frame_sha256=dict(hashes))
    return _plain(out)
