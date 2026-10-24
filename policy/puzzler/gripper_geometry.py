"""Static YAM hand CAD queries. Inputs are TCP poses and image-derived beads.

This module does not import a simulator or read a physical state. MJCF primitive
colliders are transformed through the known hand linkage, including its TCP
offset. It predicts local hand clearance, not whole-arm collision freedom.
"""
from dataclasses import dataclass
from functools import lru_cache
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .model import ROOT


@dataclass(frozen=True)
class Primitive:
    kind: str
    center: np.ndarray
    rotation: np.ndarray
    size: np.ndarray
    name: str


def _vector(node, attribute, default):
    return np.fromstring(node.get(attribute, default), sep=' ')


def _rotation(node):
    wxyz = _vector(node, 'quat', '1 0 0 0')
    return Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()


@lru_cache(maxsize=64)
def hand_primitives(width):
    """Known colliders in TCP coordinates at a commanded jaw gap [metres].

    RSS's uncalibrated YAM width command maps 0.095 m to 0.0475 m
    travel on each mirrored joint. This models that command, not a measured
    contact gap; no object pose is inferred from the fingers.
    """
    path = ROOT/'third_party/robosimstudio/assets/embodiments/yam/yam_bimanual_arms.xml'
    root = ET.parse(path).getroot()
    wrist = root.find(".//body[@name='left_link_6']")
    tcp_R = Rotation.from_euler('z', -90, degrees=True).as_matrix()
    tcp_p = np.array([0., -.044, .130])
    primitives = []

    def visit(body, position, rotation):
        for geom in body.findall('geom'):
            if geom.get('class') == 'visual' or geom.get('contype') == '0':
                continue
            cls = geom.get('class', '')
            kind = geom.get('type', 'sphere' if cls == 'sphere_collision' else 'capsule')
            if kind not in ('sphere', 'capsule', 'box'):
                raise ValueError('Unsupported hand collider: '+kind)
            size = _vector(geom, 'size', '.0006 0 0' if cls == 'sphere_collision' else '')
            if not len(size):
                raise ValueError('Missing hand collider dimensions')
            center = position + rotation @ _vector(geom, 'pos', '0 0 0')
            orientation = rotation @ _rotation(geom)
            primitives.append(Primitive(kind, tcp_R.T @ (center-tcp_p),
                                        tcp_R.T @ orientation, size,
                                        body.get('name', '')+':'+str(len(primitives))))
        for child in body.findall('body'):
            child_p = position + rotation @ _vector(child, 'pos', '0 0 0')
            child_R = rotation @ _rotation(child)
            joint = child.find('joint')
            if joint is not None:
                name = joint.get('name', '')
                if not name.endswith(('left_finger', 'right_finger')):
                    raise ValueError('Unexpected moving hand link: '+name)
                travel = np.clip(float(width)/2, 0, .0475)
                if name.endswith('right_finger'):
                    travel = -travel
                child_p += child_R @ _vector(joint, 'axis', '0 0 1') * travel
            visit(child, child_p, child_R)

    # The wrist transform itself is supplied through the requested TCP pose.
    visit(wrist, np.zeros(3), np.eye(3))
    return tuple(primitives)


def clearance(position, rotation, width, beads, *, radius=.0125, plane_z=None,
              primitives=None):
    """Minimum signed clearance to bead spheres and a horizontal support plane.

    Positive values are separated, negative values predict intersection.
    Capsule/sphere and oriented-box/sphere distances are analytic. The plane
    uses each primitive's exact support function. Inputs must come from RGB
    estimates or public static geometry; this function has no state access.
    """
    p = np.asarray(position, dtype=float)
    R = np.asarray(rotation, dtype=float)
    centers = np.asarray(beads, dtype=float).reshape(-1, 3)
    best = float('inf')
    witness = None
    shapes = tuple(hand_primitives(float(width)) if primitives is None else primitives)
    for shape in shapes:
        c = p + R @ shape.center
        A = R @ shape.rotation
        if shape.kind == 'capsule':
            r, half = shape.size[:2]
            axis = A[:, 2]
            if len(centers):
                delta = centers-c
                closest = delta-np.clip(delta @ axis, -half, half)[:, None]*axis
                distances = np.linalg.norm(closest, axis=1)-r-radius
            bottom = c[2]-abs(axis[2])*half-r
        elif shape.kind == 'sphere':
            r = shape.size[0]
            if len(centers):
                distances = np.linalg.norm(centers-c, axis=1)-r-radius
            bottom = c[2]-r
        else:
            if len(centers):
                local = (centers-c) @ A
                delta = np.abs(local)-shape.size
                distances = np.linalg.norm(np.maximum(delta, 0), axis=1)+np.minimum(delta.max(axis=1), 0)-radius
            bottom = c[2]-np.abs(A[2]) @ shape.size
        if len(centers) and float(distances.min()) < best:
            best = float(distances.min())
            witness = dict(collider=shape.name, obstacle='bead', index=int(distances.argmin()))
        if plane_z is not None and bottom-plane_z < best:
            best = float(bottom-plane_z)
            witness = dict(collider=shape.name, obstacle='support_plane')
    return dict(clearance_m=best, witness=witness)


def scene_clearance(position, rotation, width, beads, *, table_z, plane_z=None, board=None):
    """Hand versus visual bead estimates, support and known fixed robot bases.

    RSS's two fixed YAM base capsules are at the public bench mount. Their
    short centerlines are conservatively covered by overlapping spheres with
    0.1 mm radial inflation. This also catches base/finger collisions that a
    task-object-only check, or Newton's self-collision filter, can omit.
    """
    result=clearance(position,rotation,width,beads,plane_z=plane_z)
    base_points=np.array([[.6-.3475,y,table_z+.022+.026+dz]
                          for y in (-.31,.31) for dz in np.linspace(-.010,.010,5)])
    bases=clearance(position,rotation,width,base_points,radius=.0331)
    if bases['clearance_m']<result['clearance_m']:
        result=bases
        result['witness']['obstacle']='fixed_robot_base'
    if board is not None:
        pose=np.asarray(board)
        Rb=Rotation.from_quat(pose[3:]).as_matrix()
        # Public board CAD: the hinge is taller than the rim/socket plane.
        # A sphere cover of the finite cylinder is conservative at the ends.
        hinge=np.array([[.095,y,.014] for y in np.linspace(-.1575,.1575,64)])
        hinge=hinge@Rb.T+pose[:3]
        found=clearance(position,rotation,width,hinge,radius=.01325)
        if found['clearance_m']<result['clearance_m']:
            result=found
            result['witness']['obstacle']='board_hinge'
    return result
