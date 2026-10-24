"""Evidence-only RGB ray/CAD audit for release re-observation.

This module deliberately stops at an evidence report.  It never creates or
updates a placement receipt, visual-memory track, route, or policy pose.  The
report is usable by Cyan/Red release re-observe code, but the existing receipt
gates remain the authority for handoff.

Inputs are policy-visible observations: calibrated RGB detector output,
public CAD fit metadata, and synchronous robot q16/FK snapshots.  There is no
depth channel and no evaluator/object-state access here.
"""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from . import model
from .vision_observer import goals


DEFAULTS = dict(
    min_views=2,
    min_view_support=0.6,
    max_median_residual_m=0.003,
    max_rms_residual_m=0.006,
    min_pair_dt_s=1.0,
    max_pair_displacement_m=0.0015,
    max_q16_delta=0.035,
    min_hand_gap_m=0.055,
    disjoint_bead_gap_m=0.020,
)


def _finite_array(value: Any, shape: tuple[int, ...], dtype=float) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        return None
    if array.shape != shape or not np.isfinite(array).all():
        return None
    return array


def _pose(value: Any) -> np.ndarray | None:
    pose = _finite_array(value, (7,))
    if pose is None or abs(float(np.linalg.norm(pose[3:])) - 1.0) > 1e-4:
        return None
    return pose


def _cloud(name: str, pose: Any) -> np.ndarray | None:
    pose = _pose(pose)
    if pose is None:
        return None
    if name == 'board':
        points = np.array([[.0275 + sx * .110, sy * .1575, .013]
                           for sx in (-1, 1) for sy in (-1, 1)], dtype=float)
    elif name in model.CELLS:
        points = np.asarray(model.local_beads(name), dtype=float)
    else:
        return None
    return points @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3]


def _max_assignment_distance(a: np.ndarray, b: np.ndarray) -> float:
    distances = np.linalg.norm(a[:, None] - b[None, :], axis=2)
    ii, jj = linear_sum_assignment(distances)
    return float(distances[ii, jj].max())


def _q16(observed: Mapping[str, Any]) -> dict[str, Any]:
    provenance = observed.get('robot_geometry_provenance', {})
    values = _finite_array(provenance.get('robot_encoder_values'), (16,), dtype=np.float32)
    expected = provenance.get('robot_encoder_sha256')
    actual = hashlib.sha256(values.tobytes()).hexdigest() if values is not None else None
    valid = values is not None and isinstance(expected, str) and actual == expected
    return dict(
        valid=bool(valid),
        values=values.tolist() if values is not None else None,
        sha256=actual,
        expected_sha256=expected if isinstance(expected, str) else None,
        step=provenance.get('step'),
        timestamp=provenance.get('timestamp'),
        source=provenance.get('source'),
    )


def _fk(observed: Mapping[str, Any], explicit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate an optional synchronous FK snapshot without deriving one.

    Existing observers expose q16 but not a FK snapshot.  In that case this
    function returns ``status='missing'`` rather than inferring TCP state.
    """
    source = explicit
    if source is None:
        for key in ('robot_fk', 'fk', 'fk_snapshot'):
            value = observed.get(key)
            if isinstance(value, Mapping):
                source = value
                break
    if not isinstance(source, Mapping):
        return dict(status='missing', valid=False, source=None, tcp_position_m=None,
                    tcp_quaternion=None)
    position = _finite_array(source.get('tcp_position_m', source.get('position')), (3,))
    quaternion = _finite_array(source.get('tcp_quaternion', source.get('quaternion')), (4,))
    qnorm = float(np.linalg.norm(quaternion[...])) if quaternion is not None else math.nan
    valid = position is not None and quaternion is not None and abs(qnorm - 1.0) <= 1e-4
    return dict(status='valid' if valid else 'invalid', valid=bool(valid),
                source=source.get('source'),
                tcp_position_m=position.tolist() if position is not None else None,
                tcp_quaternion=quaternion.tolist() if quaternion is not None else None)


def _ray_fit(observed: Mapping[str, Any], piece: str, detection: Mapping[str, Any]) -> dict[str, Any]:
    supplied = observed.get('ray_cloud_cad_fit', {}).get(piece, {})
    supplied = supplied if isinstance(supplied, Mapping) else {}
    views = list(supplied.get('views', detection.get('views', [])) or [])
    support = supplied.get('target_support', detection.get('center_support', []))
    try:
        support = [float(x) for x in support]
    except (TypeError, ValueError):
        support = []
    median = supplied.get('median_ray_residual_m', detection.get('median_residual_m'))
    rms = supplied.get('rms_ray_residual_m', detection.get('rms_residual_m'))
    try:
        median = float(median)
        rms = float(rms)
    except (TypeError, ValueError):
        median, rms = math.nan, math.nan
    mask = supplied.get('occlusion_mask_by_view')
    if not isinstance(mask, Mapping):
        mask = {view: (i >= len(support) or support[i] < DEFAULTS['min_view_support'])
                for i, view in enumerate(views)}
    disjoint_tracks = supplied.get('view_track_disjoint', len(views) == len(set(views)))
    foreground = supplied.get('foreground_rays_by_view')
    if foreground is None:
        foreground = observed.get('foreground_rays_by_view')
    foreground_counts = {}
    foreground_valid = False
    if isinstance(foreground, Mapping):
        foreground_valid = True
        for view, value in foreground.items():
            rays = _finite_array(value, (len(value), 3)) if hasattr(value, '__len__') else None
            if rays is None or len(rays) == 0 or not np.allclose(np.linalg.norm(rays, axis=1), 1., atol=1e-3):
                foreground_valid = False
                break
            foreground_counts[str(view)] = int(len(rays))
    return dict(
        source=supplied.get('source', 'calibrated_camera_rays_public_CAD_no_depth'),
        views=views,
        view_track_disjoint=bool(disjoint_tracks),
        target_support=support,
        occlusion_mask_by_view={str(k): bool(v) for k, v in mask.items()},
        median_ray_residual_m=median if math.isfinite(median) else None,
        rms_ray_residual_m=rms if math.isfinite(rms) else None,
        pose_3d=deepcopy(supplied.get('pose_3d', detection.get('pose'))),
        foreground_rays_recorded=bool(foreground_valid),
        foreground_ray_count_by_view=foreground_counts,
        depth_channel_present=False,
    )


def _goal_error(piece: str, observed: Mapping[str, Any], parameters: Mapping[str, Any] | None) -> float | None:
    if not parameters:
        return None
    poses = observed.get('poses', {})
    body, board = _pose(poses.get(piece)), _pose(poses.get('board'))
    if body is None or board is None or piece not in model.CELLS:
        return None
    try:
        center, rotation = goals(parameters.get('mode', 'pyramid'),
                                 parameters.get('pyramid_plan', 'pyramid_robot6_plan.json'))[piece]
    except (KeyError, OSError, ValueError):
        return None
    actual = _cloud(piece, body)
    board_r = Rotation.from_quat(board[3:]).as_matrix()
    desired = (model.local_beads(piece) @ np.asarray(rotation).T + np.asarray(center)) @ board_r.T + board[:3]
    return _max_assignment_distance(actual, desired)


def _disjoint_tracks(piece: str, observed: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    poses = observed.get('poses', {})
    detections = observed.get('detections', {})
    target = _cloud(piece, poses.get(piece))
    conflicts = []
    if target is None:
        return dict(passes=False, target_present=False, minimum_gap_m=None, conflicts=[])
    for other, pose in poses.items():
        if other in (piece, 'board') or other not in model.CELLS:
            continue
        detection = detections.get(other, {})
        # Only accepted, current visual tracks participate in the disjoint
        # test. A missing/ambiguous foreign track remains an UNKNOWN gap.
        if not isinstance(detection, Mapping) or detection.get('visible') is not True:
            continue
        cloud = _cloud(other, pose)
        if cloud is None:
            continue
        minimum = float(np.linalg.norm(target[:, None] - cloud[None, :], axis=2).min())
        if minimum < float(thresholds['disjoint_bead_gap_m']):
            conflicts.append(dict(piece=other,minimum_gap_m=minimum))
    return dict(passes=not conflicts, target_present=True,
                minimum_gap_m=min((x['minimum_gap_m'] for x in conflicts), default=None),
                conflicts=conflicts)


def observe_evidence(observed: Mapping[str, Any], piece: str,
                     *, parameters: Mapping[str, Any] | None = None,
                     fk_snapshot: Mapping[str, Any] | None = None,
                     thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Produce a non-mutating current RGB/CAD/q16/FK evidence report."""
    limits = dict(DEFAULTS)
    if thresholds:
        limits.update(thresholds)
    detection = observed.get('detections', {}).get(piece, {})
    detection = detection if isinstance(detection, Mapping) else {}
    ray = _ray_fit(observed, piece, detection)
    q16 = _q16(observed)
    fk = _fk(observed, fk_snapshot)
    pose_present = _pose(observed.get('poses', {}).get(piece)) is not None
    board_present = _pose(observed.get('poses', {}).get('board')) is not None
    support = ray['target_support']
    residual_ok = (ray['median_ray_residual_m'] is not None
                   and ray['rms_ray_residual_m'] is not None
                   and ray['median_ray_residual_m'] < float(limits['max_median_residual_m'])
                   and ray['rms_ray_residual_m'] < float(limits['max_rms_residual_m']))
    support_ok = (len(ray['views']) >= int(limits['min_views'])
                  and len(support) >= len(ray['views'])
                  and all(x >= float(limits['min_view_support']) for x in support[:len(ray['views'])]))
    identity_ok = detection.get('visible') is True and not bool(
        detection.get('reason') and any(word in str(detection.get('reason')).lower()
                                        for word in ('occluded', 'inconsistent', 'ambiguous')))
    disjoint = _disjoint_tracks(piece, observed, limits)
    goal_error = _goal_error(piece, observed, parameters)
    return dict(
        schema='puzzler.rgb_ray_cad_evidence.v1',
        piece=piece,
        observation_source=observed.get('observation_source'),
        timestamp=observed.get('timestamp'),
        frame_sha256=deepcopy(observed.get('frame_sha256', {})),
        current_board_present=board_present,
        current_piece_pose_present=pose_present,
        rgb_foreground_ray_cad=ray,
        identity=dict(visible=detection.get('visible'), reason=detection.get('reason'),
                      identity_source=detection.get('identity_source'), passes=identity_ok),
        disjoint_tracks=disjoint,
        q16=q16,
        fk=fk,
        public_cad=dict(goal_error_m=goal_error,
                        goal_error_passes=(goal_error is not None and goal_error < .003)),
        gates=dict(current_rgb_source=observed.get('observation_source') in
                   ('rgb_shape_geometry', 'calibrated_rgb_cad_seeded'),
                   board_present=board_present, piece_pose_present=pose_present,
                   foreground_rays=bool(ray['foreground_rays_recorded']),
                   support=support_ok, residual=residual_ok,
                   view_tracks_disjoint=bool(ray['view_track_disjoint']),
                   identity=identity_ok, q16=q16['valid'], fk=fk['valid'],
                   disjoint=disjoint['passes']),
        evidence_only=True,
        receipt_mutation='forbidden',
        receipt_status='UNCHANGED',
        receipt_eligible=False,
    )


def pair_stability(before: Mapping[str, Any], after: Mapping[str, Any], piece: str,
                   *, fk_before: Mapping[str, Any] | None = None,
                   fk_after: Mapping[str, Any] | None = None,
                   hand_gap_m: float | None = None,
                   thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Audit a fresh release RGB pair; never mint a receipt."""
    limits = dict(DEFAULTS)
    if thresholds:
        limits.update(thresholds)
    t0, t1 = before.get('timestamp'), after.get('timestamp')
    try:
        dt = float(t1) - float(t0)
    except (TypeError, ValueError):
        dt = math.nan
    poses0, poses1 = before.get('poses', {}), after.get('poses', {})
    target0, target1 = _cloud(piece, poses0.get(piece)), _cloud(piece, poses1.get(piece))
    board0, board1 = _cloud('board', poses0.get('board')), _cloud('board', poses1.get('board'))
    target_disp = _max_assignment_distance(target0, target1) if target0 is not None and target1 is not None else None
    board_disp = _max_assignment_distance(board0, board1) if board0 is not None and board1 is not None else None
    q0, q1 = _q16(before), _q16(after)
    qdelta = None
    if q0['values'] is not None and q1['values'] is not None:
        qdelta = float(np.max(np.abs(np.asarray(q1['values']) - np.asarray(q0['values']))))
    fk0, fk1 = _fk(before, fk_before), _fk(after, fk_after)
    fk_delta = None
    if fk0['tcp_position_m'] is not None and fk1['tcp_position_m'] is not None:
        fk_delta = float(np.linalg.norm(np.asarray(fk1['tcp_position_m']) - np.asarray(fk0['tcp_position_m'])))
    frame0, frame1 = before.get('frame_sha256', {}), after.get('frame_sha256', {})
    hand_gap_ok = hand_gap_m is not None and float(hand_gap_m) > float(limits['min_hand_gap_m'])
    checks = dict(
        pair_timestamp_delta_s=dt,
        pair_time_passes=math.isfinite(dt) and dt >= float(limits['min_pair_dt_s']),
        target_displacement_m=target_disp,
        board_displacement_m=board_disp,
        target_stable=target_disp is not None and target_disp < float(limits['max_pair_displacement_m']),
        board_stable=board_disp is not None and board_disp < float(limits['max_pair_displacement_m']),
        q16_before_valid=q0['valid'], q16_after_valid=q1['valid'],
        q16_max_delta=qdelta,
        q16_stable=qdelta is not None and qdelta <= float(limits['max_q16_delta']),
        fk_before_valid=fk0['valid'], fk_after_valid=fk1['valid'],
        fk_tcp_delta_m=fk_delta,
        hand_gap_m=hand_gap_m,
        hand_separated=hand_gap_ok,
        distinct_rgb_frames=bool(frame0 and frame1 and frame0 != frame1),
    )
    required = ('pair_time_passes', 'target_stable', 'board_stable', 'q16_stable',
                'fk_before_valid', 'fk_after_valid', 'hand_separated', 'distinct_rgb_frames')
    checks['stable_pair_passes'] = all(bool(checks.get(key)) for key in required)
    return dict(schema='puzzler.rgb_ray_cad_stability_evidence.v1', piece=piece,
                before_frame_sha256=deepcopy(frame0), after_frame_sha256=deepcopy(frame1),
                checks=checks, evidence_only=True, receipt_mutation='forbidden',
                receipt_status='UNCHANGED', receipt_eligible=False)

