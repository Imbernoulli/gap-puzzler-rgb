"""Scratch public-CAD query; no RSS scene, state, contacts, or evaluator.

Uses the pinned Newton/Warp CUDA GJK distance and MPR overlap functions.
Board convex proxies come verbatim from the local public asset manifest.
Hand spheres/capsules/boxes come from the existing public MJCF parser.
All relative poses are supplied by the caller (saved RGB + robot encoders).
"""
from pathlib import Path
import os

import numpy as np
import trimesh
import warp as wp
from scipy.spatial.transform import Rotation

from newton._src.geometry.simplex_solver import create_solve_closest_distance
from newton._src.geometry.mpr import create_solve_mpr
from newton._src.geometry.support_function import GenericShapeData, SupportMapDataProvider, support_map
from newton._src.geometry.types import GeoType

from puzzler.assets import manifest
from puzzler.gripper_geometry import hand_primitives

wp.config.quiet = True
wp.init()
GJK = create_solve_closest_distance(support_map)
MPR = create_solve_mpr(support_map)


class GeometryQueryRefusal(ValueError):
    """Invalid geometry or numerical output cannot certify positive clearance."""


def finite_array(value, shape, label):
    try:
        a=np.asarray(value,dtype=float)
    except (TypeError,ValueError,OverflowError) as error:
        raise GeometryQueryRefusal(label+' must be numeric') from error
    if a.shape!=shape or not np.all(np.isfinite(a)):
        raise GeometryQueryRefusal(label+' must have shape '+str(shape)+' and finite entries')
    # Every native query uses float32. Finite float64 input must not silently
    # overflow while crossing that boundary.
    if np.any(np.abs(a)>np.finfo(np.float32).max):
        raise GeometryQueryRefusal(label+' exceeds the native float32 range')
    return a


def finite_scalar(value,label):
    a=finite_array(value,(),label)
    return float(a)


def validate_quaternion(value,label):
    q=finite_array(value,(4,),label)
    if abs(float(np.linalg.norm(q))-1.)>1e-5:
        raise GeometryQueryRefusal(label+' must be a unit xyzw quaternion')
    return q


def validate_rotation(value,label):
    r=finite_array(value,(3,3),label)
    if not np.allclose(r.T@r,np.eye(3),rtol=0.,atol=1e-5) or abs(float(np.linalg.det(r))-1.)>1e-5:
        raise GeometryQueryRefusal(label+' must be a proper orthonormal rotation')
    return r


def validate_query_inputs(board_pose,position,rotation,width):
    b=finite_array(board_pose,(7,),'board pose')
    validate_quaternion(b[3:],'board quaternion')
    p=finite_array(position,(3,),'TCP position')
    r=validate_rotation(rotation,'TCP rotation')
    w=finite_scalar(width,'jaw width')
    if not 0.<=w<=.095:
        raise GeometryQueryRefusal('jaw width is outside public YAM command range [0,.095] m')
    return b,p,r,w


def validate_result(result,label,*,require_native_witness=False):
    if not isinstance(result,dict) or 'clearance_m' not in result:
        raise GeometryQueryRefusal(label+' lacks a clearance result')
    finite_scalar(result['clearance_m'],label+' clearance')
    witness=result.get('witness')
    if require_native_witness and not isinstance(witness,dict):
        raise GeometryQueryRefusal(label+' lacks native witness points')
    for key in ('point_board_local','point_hand_board_local'):
        if isinstance(witness,dict) and key in witness:
            finite_array(witness[key],(3,),label+' '+key)
        elif require_native_witness:
            raise GeometryQueryRefusal(label+' lacks '+key)
    groups=result.get('groups')
    if require_native_witness and (not isinstance(groups,dict) or
            set(groups)!=set(('floor','socket_post','rim','hinge','lid_border'))):
        raise GeometryQueryRefusal(label+' lacks complete board group results')
    if isinstance(groups,dict):
        for key,value in groups.items():
            if not isinstance(value,dict) or 'clearance_m' not in value:
                raise GeometryQueryRefusal(label+' has malformed group '+key)
            finite_scalar(value['clearance_m'],label+' '+key+' clearance')
    return result


def create_pair_query(gjk_core,mpr_core):
    """Factory permits controlled bad-backend injection in the scratch audit."""
    @wp.kernel(enable_backward=False)
    def checked_pair_query(
        board: wp.array[GenericShapeData],
        hand: wp.array[GenericShapeData],
        positions: wp.array[wp.vec3],
        rotations: wp.array[wp.quat],
        distance: wp.array[float],
        points_a: wp.array[wp.vec3],
        points_b: wp.array[wp.vec3],
        flags: wp.array[int],
    ):
        k = wp.tid()
        i = k // hand.shape[0]
        j = k % hand.shape[0]
        provider = SupportMapDataProvider()
        separated, pa, pb, normal, dist = wp.static(gjk_core)(
            board[i], hand[j], rotations[j], positions[j], 0.0, provider, 96, 1.0e-7, 0.0
        )
        flag = int(0)
        if not wp.isfinite(dist) or not wp.isfinite(pa) or not wp.isfinite(pb) or not wp.isfinite(normal):
            # Flag 3 is never an admissible public result. The host rejects
            # the entire query before argmin; no bad pair can be hidden by
            # another finite pair or by an MPR fallback.
            dist = 0.0
            flag = 3
        elif not separated:
            collision, ma, mb, mn, penetration = wp.static(mpr_core)(
                board[i], hand[j], rotations[j], positions[j], 0.0, provider, 96, 1.0e-7
            )
            if not wp.isfinite(penetration) or not wp.isfinite(ma) or not wp.isfinite(mb) or not wp.isfinite(mn):
                dist = 0.0
                flag = 3
            elif collision:
                dist = -wp.abs(penetration)
                pa = ma
                pb = mb
                flag = 1
            else:
                # GJK overlap/touching without an MPR penetration remains
                # zero, conservatively failing the positive margin.
                dist = 0.0
                flag = 2
        distance[k] = dist
        points_a[k] = pa
        points_b[k] = pb
        flags[k] = flag
    return checked_pair_query


pair_query=create_pair_query(GJK.core,MPR.core)


def validate_pair_outputs(distances,points_a,points_b,flags,n):
    d=finite_array(distances,(n,),'all native pair distances')
    pa=finite_array(points_a,(n,3),'all native board witness points')
    pb=finite_array(points_b,(n,3),'all native hand witness points')
    f=finite_array(flags,(n,),'all native pair status flags')
    if not np.all(np.isin(f,[0,1,2])):
        raise GeometryQueryRefusal('native GJK/MPR returned an invalid pair status')
    return d,pa,pb,f.astype(np.int32)


def descriptor(kind, size, mesh_id=None, center=None):
    value = GenericShapeData()
    value.shape_type = int(kind)
    value.scale = wp.vec3(*np.pad(np.asarray(size, dtype=float), (0, 3-len(size))))
    value.auxiliary = wp.vec3(0.0)
    value.center = wp.vec3(*(np.zeros(3) if center is None else center))
    value.shape_index = -1
    if mesh_id is not None:
        # Identical 22/22/20-bit representation to Newton's pack_mesh_ptr.
        ptr = int(mesh_id)
        value.auxiliary = wp.vec3(float(ptr & 0x3FFFFF), float((ptr >> 22) & 0x3FFFFF), float((ptr >> 44) & 0xFFFFF))
    return value


def board_group(index):
    if index == 0:
        return 'floor'
    if index <= 72:
        return 'socket_post'
    if index <= 76:
        return 'rim'
    return 'hinge' if index == 77 else 'lid_border'


class PublicBoardQuery:
    def __init__(self):
        requested_device = os.environ.get('ROBOSIMSTUDIO_DEVICE', 'cuda:0')
        try:
            self.device = wp.get_device(requested_device)
        except Exception as error:
            raise GeometryQueryRefusal('Public board query requires an available CUDA device: '+requested_device) from error
        if not self.device.is_cuda:
            raise GeometryQueryRefusal('Public board query requires CUDA, got '+str(self.device))
        record = manifest()['board']
        self.paths = [record['collision_path'], *record['collision_paths']]
        if len(self.paths)!=79:
            raise GeometryQueryRefusal('Expected all 79 public board collision components')
        self.meshes = []
        self.raw_meshes = []
        shapes = []
        for path in self.paths:
            mesh = trimesh.load(path, force='mesh', process=False)
            finite_array(mesh.vertices,(len(mesh.vertices),3),'public board vertices')
            if not mesh.is_convex:
                raise ValueError('Public convex component is not convex: '+path)
            native = wp.Mesh(points=wp.array(mesh.vertices, dtype=wp.vec3, device=self.device),
                             indices=wp.array(mesh.faces.reshape(-1), dtype=wp.int32, device=self.device))
            self.raw_meshes.append(mesh)
            self.meshes.append(native)
            shapes.append(descriptor(GeoType.CONVEX_MESH, [1, 1, 1], native.id, mesh.bounds.mean(axis=0)))
        self.board = wp.array(shapes, dtype=GenericShapeData, device=self.device)

    def _native_pair_results(self,primitives,positions,rotations,kernel=pair_query):
        types = {'sphere': GeoType.SPHERE, 'capsule': GeoType.CAPSULE, 'box': GeoType.BOX}
        hand = wp.array([descriptor(types[s.kind], s.size) for s in primitives], dtype=GenericShapeData, device=self.device)
        n = len(self.paths)*len(primitives)
        distances = wp.empty(n, dtype=float, device=self.device)
        pa = wp.empty(n, dtype=wp.vec3, device=self.device)
        pb = wp.empty(n, dtype=wp.vec3, device=self.device)
        flags = wp.empty(n, dtype=int, device=self.device)
        wp.launch(kernel, n, inputs=[self.board, hand,
            wp.array(positions, dtype=wp.vec3, device=self.device),
            wp.array(rotations, dtype=wp.quat, device=self.device)],
            outputs=[distances, pa, pb, flags], device=self.device)
        return distances.numpy(),pa.numpy(),pb.numpy(),flags.numpy()

    def query_shapes(self, primitives, positions, rotations):
        n_hand=len(primitives)
        if n_hand==0:
            raise GeometryQueryRefusal('Empty hand geometry cannot certify clearance')
        positions=finite_array(positions,(n_hand,3),'native shape positions')
        rotations=finite_array(rotations,(n_hand,4),'native shape quaternions')
        for q in rotations:
            validate_quaternion(q,'native shape quaternion')
        for shape in primitives:
            if shape.kind not in ('sphere','capsule','box'):
                raise GeometryQueryRefusal('Unsupported public hand primitive')
            size=finite_array(shape.size,np.shape(shape.size),'hand primitive size')
            dimensions={'sphere':1,'capsule':2,'box':3}[shape.kind]
            if size.ndim!=1 or not dimensions<=len(size)<=3 or np.any(size[:dimensions]<=0.):
                raise GeometryQueryRefusal('Public hand primitive has invalid dimensions')
        n=len(self.paths)*n_hand
        d,pa,pb,flags=validate_pair_outputs(*self._native_pair_results(primitives,positions,rotations),n)
        d = d.reshape(len(self.paths), n_hand)
        k = int(d.argmin())
        i, j = np.unravel_index(k, d.shape)
        result = dict(clearance_m=float(d[i, j]),
            witness=dict(board_mesh=Path(self.paths[i]).name, board_group=board_group(i),
                         hand_collider=primitives[j].name, point_board_local=pa[k].tolist(),
                         point_hand_board_local=pb[k].tolist(), overlap_flag=int(flags[k])))
        result['groups'] = {}
        for group in ('floor', 'socket_post', 'rim', 'hinge', 'lid_border'):
            indices = [i for i in range(len(self.paths)) if board_group(i) == group]
            local = d[indices]
            a,b = np.unravel_index(int(local.argmin()),local.shape)
            result['groups'][group] = dict(clearance_m=float(local[a,b]),
                board_mesh=Path(self.paths[indices[a]]).name, hand_collider=primitives[b].name)
        return validate_result(result,'public board',require_native_witness=True)

    def query(self, board_pose, position, rotation, width):
        b,position,rotation,width=validate_query_inputs(board_pose,position,rotation,width)
        Rb = Rotation.from_quat(b[3:]).as_matrix()
        p = Rb.T @ (np.asarray(position)-b[:3])
        R = Rb.T @ np.asarray(rotation)
        primitives = hand_primitives(float(width))
        positions = [p+R@s.center for s in primitives]
        rotations = [Rotation.from_matrix(R@s.rotation).as_quat() for s in primitives]
        return self.query_shapes(primitives, positions, rotations)
