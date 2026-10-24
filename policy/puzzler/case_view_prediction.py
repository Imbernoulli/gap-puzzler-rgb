"""Predict which calibrated RGB views can re-measure the case outline.

Inputs are public only: calibrated camera rig, public robot visual meshes at
an actual or proposed q16 (public FK), the dated actual RGB case pose used as
an optimizer proposal, and the owned held-target estimate. A synthetic
silhouette (case CAD minus projected robot/held-target occluders) is passed
through the UNCHANGED per-view case acceptance gates of board_vision. The
prediction only ranks proposed view motions; it never supplies a measured
pose, and the actual RGB packet taken afterwards remains the only evidence.
"""
import math

import cv2
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from . import model
from .board_vision import _case_components, _fit_case_view
from .vision import calibration, rig_specs

_CAMERAS = None
_MESHES = None


def _cameras():
    global _CAMERAS
    if _CAMERAS is None:
        _CAMERAS = calibration(rig_specs())
    return _CAMERAS


def _meshes(geometry):
    """Public robot visual mesh vertices (local shape frame), cached."""
    global _MESHES
    if _MESHES is None:
        rows = []
        scales = geometry.model.shape_scale.numpy()
        for index, source in enumerate(geometry.model.shape_source):
            vertices = getattr(source, 'vertices', None)
            if vertices is None or not geometry.shape_labels[index].endswith('_visual'):
                continue
            vertices = np.asarray(vertices, dtype=float)*np.asarray(scales[index], dtype=float)
            if len(vertices) > 600:
                vertices = vertices[ConvexHull(vertices).vertices]
            rows.append((index, vertices))
        _MESHES = rows
    return _MESHES


def _robot_world_hulls(geometry, q16):
    world = geometry.fk(np.asarray(q16, dtype=float)).numpy()
    out = []
    for index, vertices in _meshes(geometry):
        transform = world[index]
        R = Rotation.from_quat(transform[3:7]).as_matrix()
        out.append(vertices@R.T+transform[:3])
    return out


def _case_image_polygons(camera, board):
    yaw = 2*math.atan2(board[5], board[6])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    polygons = []
    for vertices in _case_components():
        world = vertices@R.T+np.r_[board[:2], .753]
        pixels = camera.project(world)
        polygons.append(pixels[ConvexHull(pixels).vertices])
    return polygons


def synthetic_view(camera, board, robot_hulls, held_centers=(), shape=(1024, 1280)):
    image = np.full((*shape, 3), 72, np.uint8)
    for polygon in _case_image_polygons(camera, board):
        cv2.fillConvexPoly(image, np.round(polygon).astype(np.int32), (8, 8, 8))
    origin = camera.camera_to_world[:3, 3]
    forward = camera.camera_to_world[:3, 2]
    for hull in robot_hulls:
        depth = (hull-origin)@forward
        if np.any(depth <= .02):
            continue
        pixels = camera.project(hull)
        if not np.isfinite(pixels).all():
            continue
        try:
            polygon = pixels[ConvexHull(pixels).vertices]
        except Exception:
            continue
        cv2.fillConvexPoly(image, np.round(polygon).astype(np.int32), (150, 150, 150))
    for center in np.asarray(held_centers, dtype=float).reshape(-1, 3):
        depth = float((center-origin)@forward)
        if depth <= .02:
            continue
        pixel = camera.project(center[None])[0]
        radius = model.RADIUS*camera.K[0, 0]/depth
        cv2.circle(image, tuple(int(round(v)) for v in pixel), int(math.ceil(radius+1)), (40, 200, 220), -1)
    return image


def _side_coordinates(camera, board, pixels):
    """Plane points of outline pixels in the case frame -> (side, along)."""
    yaw = 2*math.atan2(board[5], board[6])
    c, s = math.cos(yaw), math.sin(yaw)
    world = camera.plane(np.asarray(pixels, dtype=float), .766)[:, :2]-np.asarray(board[:2])
    local = world@np.array([[c, -s], [s, c]])
    distance = np.c_[np.abs(local[:, 0]+.0825), np.abs(local[:, 0]-.1375),
                     np.abs(local[:, 1]+.1575), np.abs(local[:, 1]-.1575)]
    side = distance.argmin(axis=1)
    along = np.where(side < 2, local[:, 1], local[:, 0])
    return side, along, distance.min(axis=1)


def fast_predict_views(geometry, q16, board, held_body_pose=None, views=('rear', 'side'),
                       maximum_occluder_fraction=.5):
    """Cheap outline-visibility version of the same per-view gates.

    The projected CAD case outline is sampled; samples hidden by projected
    public robot meshes or the held target are removed; per-side spans use the
    detector's 35mm/three-side rule, and a view whose visible silhouette
    boundary is dominated by occluder edges (the median-residual gate) is
    predicted rejected.
    """
    board = np.asarray(board, dtype=float)
    hulls = _robot_world_hulls(geometry, q16)
    held = ()
    if held_body_pose is not None:
        body = np.asarray(held_body_pose, dtype=float)
        held = model.local_beads('cyan')@Rotation.from_quat(body[3:]).as_matrix().T+body[:3]
    cameras = _cameras()
    result = {}
    for key in views:
        camera = cameras[key]
        image = synthetic_view(camera, board, hulls, held)
        case = (image.max(axis=2) < 32).astype(np.uint8)
        full = np.zeros_like(case)
        for polygon in _case_image_polygons(camera, board):
            cv2.fillConvexPoly(full, np.round(polygon).astype(np.int32), 1)
        outline, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not outline:
            result[key] = dict(visible=False, reason='case outline outside view', edge_support_m=[0.]*4)
            continue
        points = max(outline, key=cv2.contourArea)[:, 0]
        h, w = case.shape
        inside = (points[:, 0] >= 0)&(points[:, 0] < w)&(points[:, 1] >= 0)&(points[:, 1] < h)
        points = points[inside]
        # The detector fits only the largest dark silhouette component.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(case, connectivity=8)
        if count < 2:
            result[key] = dict(visible=False, reason='case outline absent', edge_support_m=[0.]*4)
            continue
        largest = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        case = (labels == largest).astype(np.uint8)
        visible = case[points[:, 1], points[:, 0]] > 0
        side, along, _ = _side_coordinates(camera, board, points[visible])
        spans = [float(np.ptp(along[side == k])) if np.sum(side == k) > 1 else 0. for k in range(4)]
        supported = sum(v > .035 for v in spans) >= 3 and (min(spans[:2]) > .035 or min(spans[2:]) > .035)
        observed, _ = cv2.findContours(case, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        occluder = 1.
        if observed:
            contour = max(observed, key=cv2.contourArea)[:, 0]
            on_outline = cv2.dilate(np.where(full > 0, 0, 1).astype(np.uint8), np.ones((5, 5), np.uint8))
            occluder = float(1.-np.mean(on_outline[contour[:, 1], contour[:, 0]] > 0))
        ok = bool(supported and occluder <= maximum_occluder_fraction)
        result[key] = dict(visible=ok, reason='' if ok else ('case outline lacks three supported sides' if not supported
                                                             else 'occluder boundary dominates outline'),
                           edge_support_m=spans, occluder_boundary_fraction=occluder)
    return result


def predict_views(geometry, q16, board, held_body_pose=None, views=('front', 'rear', 'side', 'top')):
    """Per-view predicted acceptance using the original case-view gates."""
    board = np.asarray(board, dtype=float)
    hulls = _robot_world_hulls(geometry, q16)
    held = ()
    if held_body_pose is not None:
        body = np.asarray(held_body_pose, dtype=float)
        held = model.local_beads('cyan')@Rotation.from_quat(body[3:]).as_matrix().T+body[:3]
    proposal = np.r_[board[:2], 2*math.atan2(board[5], board[6])]
    cameras = _cameras()
    result = {}
    for key in views:
        image = synthetic_view(cameras[key], board, hulls, held)
        fit = _fit_case_view(image, cameras[key], proposal)
        result[key] = dict(visible=bool(fit.get('visible')), reason=fit.get('reason', ''),
                           edge_support_m=[float(v) for v in fit.get('edge_support_m', [])],
                           median_outline_residual_m=fit.get('median_outline_residual_m'))
    return result


def predicted_score(prediction):
    """Rank by count of accepted views, then by the best accepted view's
    weakest of its three strongest supported CAD edges (acceptance margin)."""
    accepted = [row for row in prediction.values() if row['visible']]
    margin = max((sorted(row['edge_support_m'])[-3] for row in accepted
                  if len(row['edge_support_m']) >= 3), default=0.)
    return (len(accepted), float(margin))
