"""RGB-only detection and calibrated multi-view CAD fitting.

The estimator receives RGB images, fixed camera calibration and optional
image-derived or motion-predicted optimization seeds. It has no simulator-state,
task or evaluator input. CAD geometry, palette and a broad workspace are priors.
The separate rig-spec helper imports RSS's static camera-specification type.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import cv2
import numpy as np
from scipy.ndimage import maximum_filter, map_coordinates
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial.transform import Rotation

from .model import COLORS, CELLS, RADIUS, PITCH, local_beads


# Public visual CAD: assets.py connects adjacent bead centres with these necks.
NECK_RADIUS=.0045


@lru_cache(maxsize=None)
def cad_necks(name):
    local=local_beads(name)
    return tuple((i,j) for i,a in enumerate(local) for j,b in enumerate(local)
                 if i<j and abs(np.linalg.norm(a-b)-PITCH)<1.e-7)


def ray_shape_distance(name,centers,rays,origin):
    """Ray tangency residual to public bead + neck CAD, without scene state.

    A radius-4.5mm capsule between bead centres has the same external neck
    surface as the finite CAD cylinder: its end caps are inside the beads.
    """
    delta=centers-origin
    cross=np.cross(rays[:,None,:],delta[None,:,:])
    distance=(np.linalg.norm(cross,axis=2)-RADIUS).min(axis=1)
    pairs=cad_necks(name)
    if pairs:
        starts=np.asarray([centers[i]-origin for i,_ in pairs])
        axes=np.asarray([centers[j]-centers[i] for i,j in pairs])
        aa=np.cross(rays[:,None,:],starts[None,:,:])
        vv=np.cross(rays[:,None,:],axes[None,:,:])
        fractions=np.clip(-np.sum(aa*vv,axis=2)/np.maximum(np.sum(vv*vv,axis=2),1.e-15),0.,1.)
        neck=(np.linalg.norm(aa+fractions[...,None]*vv,axis=2)-NECK_RADIUS).min(axis=1)
        distance=np.minimum(distance,neck)
    return distance


@dataclass(frozen=True)
class Camera:
    name: str
    K: np.ndarray
    camera_to_world: np.ndarray

    def project(self, points):
        p = (np.asarray(points) - self.camera_to_world[:3, 3]) @ self.camera_to_world[:3, :3]
        return (p @ self.K.T)[:, :2] / p[:, 2, None]

    def rays(self, pixels):
        ray = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(self.K).T
        ray = ray @ self.camera_to_world[:3, :3].T
        return ray / np.linalg.norm(ray, axis=1, keepdims=True)

    def plane(self, pixels, z):
        rays = self.rays(pixels)
        origin = self.camera_to_world[:3, 3]
        return origin + rays * ((z-origin[2])/rays[:, 2])[:, None]


def rig_specs():
    """Fixed, explicitly calibrated RGB rig. No depth or instance-ID channels."""
    from robosimstudio import CameraSpec
    common = dict(width=1280, height=1024, channels=('rgb',))
    return [
        CameraSpec(name='top', eye=(.6, 0, 1.65), look_at=(.6, 0, .75),
                   up=(1, 0, 0), fov_y=53, **common),
        CameraSpec(name='front', eye=(1., -.70, 1.4), look_at=(.55, 0, .78), fov_y=58, **common),
        CameraSpec(name='rear', eye=(1., .70, 1.4), look_at=(.55, 0, .78), fov_y=58, **common),
        CameraSpec(name='side', eye=(.12, 0, 1.5), look_at=(.58, 0, .80), fov_y=65, **common),
    ]


def calibration(specs):
    return {c.name: Camera(c.name, c.intrinsics.copy(), c.world_matrix.copy()) for c in specs}


def detect(rgb, *, only_names=None):
    """Palette nearest-neighbour detection in normalized RGB; returns image masks.

    Brightness normalization handles diffuse shading. Low saturation and very
    dark pixels are rejected. Colour similarity is a stated task assumption.
    """
    a = rgb.astype(np.float32)/255
    norm = np.linalg.norm(a, axis=2)
    chroma = a / np.maximum(norm[..., None], 1e-6)
    names = list(CELLS)
    palette = np.array([COLORS[n] for n in names])
    palette /= np.linalg.norm(palette, axis=1, keepdims=True)
    # Max dot product avoids a full H x W x colours x channels allocation.
    scores = chroma @ palette.T
    labels = scores.argmax(axis=2)
    valid = (scores.max(axis=2) > .995) & ((a.max(axis=2)-a.min(axis=2)) > .10) & (norm > .12)
    masks = {}
    for i, name in enumerate(names):
        # Keep the full palette competition above: omitted masks cannot
        # reassign a pixel to one of the requested parts.
        if only_names is not None and name not in only_names:
            continue
        mask = ((labels == i) & valid).astype(np.uint8)
        # Keep the largest spatial group. Specular highlights on a different
        # colour can otherwise create distant, tiny false positive patches.
        joined = cv2.dilate(mask, np.ones((7, 7), np.uint8))
        count, components, stats, _ = cv2.connectedComponentsWithStats(joined)
        if count > 1:
            sizes = np.bincount(components.ravel(), weights=mask.ravel(), minlength=count)
            sizes[0] = 0
            masks[name] = (mask & (components == sizes.argmax())).astype(np.uint8)
        else:
            masks[name] = mask
    return masks


def boundary(mask, limit=240):
    # Holes between adjacent beads are physical silhouette evidence too.
    # Keeping only the outer perimeter erases most evidence for interior parts
    # once a monochrome scene becomes packed.
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2))
    pts = np.concatenate(contours)[:, 0].astype(float)
    return pts[np.linspace(0, len(pts)-1, min(limit, len(pts))).astype(int)]


def _initial_planar(name, mask, camera):
    """Initialize from visible bead centres; height is only an optimizer seed."""
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    maxima = (dt == maximum_filter(dt, size=13)) & (dt > 5)
    n, labels, _, centers = cv2.connectedComponentsWithStats(maxima.astype(np.uint8))
    peaks = centers[1:]
    if len(peaks) < 3:
        raise ValueError('fewer than three visible bead centres')
    world = camera.plane(peaks, .7625)
    local = local_beads(name)
    best = None
    for yaw in np.linspace(-np.pi, np.pi, 72, endpoint=False):
        R = Rotation.from_euler('z', yaw).as_matrix()
        rotated = local @ R.T
        for a in rotated:
            for b in world:
                center = b-a
                distances = np.linalg.norm((rotated+center)[:, None, :]-world[None, :, :], axis=2)
                ii, jj = linear_sum_assignment(distances)
                cost = np.mean(distances[ii, jj]**2) + abs(len(world)-len(local))*.00001
                if best is None or cost < best[0]:
                    best = (cost, np.r_[center, 0., 0., yaw])
    return best[1]


def fit_piece(name, masks, cameras, previous=None, contour_points=None):
    """Fit rigid bead-and-neck CAD silhouettes to RGB contours from >= 2 views.

    Residuals are ray-to-sphere tangent errors, in metres. The observed outer
    contour must touch the union of known spheres and connecting cylinders.
    Robust loss tolerates limited occlusion. No target pose is supplied here.
    """
    views = []
    for key, mask in masks.items():
        pts = boundary(mask) if contour_points is None else contour_points.get(key,np.empty((0,2)))
        if len(pts) >= 35 and mask.sum() >= 100:
            camera = cameras[key]
            views.append((key, camera, pts, camera.rays(pts)))
    if len(views) < 2:
        return dict(visible=False, reason='fewer than two RGB views', views=[v[0] for v in views])
    if previous is None:
        if 'top' not in masks or masks['top'].sum() < 100:
            return dict(visible=False, reason='no initialization view')
        try:
            seed = _initial_planar(name, masks['top'], cameras['top'])
        except ValueError as exc:
            return dict(visible=False, reason=str(exc))
    else:
        seed = np.r_[previous[:3], Rotation.from_quat(previous[3:]).as_rotvec()]
    local = local_beads(name)

    def residual(x):
        centers = local @ Rotation.from_rotvec(x[3:]).as_matrix().T + x[:3]
        out = []
        for key, camera, pts, rays in views:
            out.append(ray_shape_distance(name,centers,rays,camera.camera_to_world[:3,3]))
        return np.concatenate(out)

    fitted = least_squares(residual, seed, loss='soft_l1', f_scale=.0008,
                           diff_step=1e-4, max_nfev=85, xtol=1e-8, ftol=1e-6)
    # Occluders cut through the interior of a projected sphere. Their image
    # edges are not the sphere's silhouette. Refit using a fixed inlier set
    # selected by the first robust image fit, never by a true pose.
    initial_residual=residual(fitted.x)
    inliers=(initial_residual>-.0014)&(initial_residual<.0025)
    if inliers.sum()>=80:
        fitted=least_squares(lambda x:residual(x)[inliers],fitted.x,loss='soft_l1',
                             f_scale=.0005,diff_step=1e-4,max_nfev=60,xtol=1e-8,ftol=1e-6)
    x = fitted.x
    centers = local @ Rotation.from_rotvec(x[3:]).as_matrix().T+x[:3]
    # A small patch cannot certify an entire part. Check predicted centre
    # support, contour agreement, and physical workspace bounds independently.
    support = []
    boxes = {}
    for key, camera, pts, _ in views:
        uv = camera.project(centers)
        dt = cv2.distanceTransform(1-masks[key], cv2.DIST_L2, 5)
        distances = map_coordinates(dt, [uv[:, 1], uv[:, 0]], order=1, mode='constant', cval=1000.)
        support.append(float(np.mean(distances < 4.)))
        ys, xs = np.nonzero(masks[key])
        boxes[key] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    residual_m = float(np.median(np.abs(residual(x))))
    rms = float(np.sqrt(np.mean(residual(x)**2)))
    valid = (residual_m < .0018 and rms < .004 and sum(v >= .6 for v in support) >= 2
             and .1 < x[0] < 1.1 and abs(x[1]) < .6 and .73 < x[2] < 1.3)
    return dict(visible=bool(valid), pose=np.r_[x[:3], Rotation.from_rotvec(x[3:]).as_quat()].tolist(),
                median_residual_m=residual_m, rms_residual_m=rms, center_support=support,
                boxes=boxes, views=[v[0] for v in views], reason='' if valid else 'inconsistent or occluded silhouettes')


def fit_board(rgb, camera):
    """Robust known-outline fit, with explicit level-table and yaw priors.

    Height and tilt are geometric priors, not RGB measurements. Unsupported
    outlines refuse; the rectangle cannot expand to include a robot hand.
    """
    from .board_vision import fit_board_outline
    return fit_board_outline(rgb,camera)


def estimate(images, cameras, previous=None, *, only_names=None):
    """All image-derived scene estimates; no simulator argument or truth fallback."""
    previous = previous or {}
    masks = {key: detect(rgb, only_names=only_names) for key, rgb in images.items()}
    detections = {name: fit_piece(name, {key: value[name] for key, value in masks.items()},
                                   cameras, previous.get(name)) for name in CELLS
                  if only_names is None or name in only_names}
    detections['board'] = fit_board(images['top'], cameras['top'])
    poses = {name: value['pose'] for name, value in detections.items() if value['visible']}
    return dict(poses=poses, detections=detections, observation_source='calibrated_rgb_only')
