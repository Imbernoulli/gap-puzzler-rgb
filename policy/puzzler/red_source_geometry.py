"""Explicit RED source-contact diagnostic; public geometry only, no state access."""
import numpy as np
from scipy.spatial.transform import Rotation
from puzzler.gripper_geometry import hand_primitives
from puzzler.public_board_clearance import (PublicBoardQuery,GeometryQueryRefusal,
    validate_query_inputs,validate_result,finite_array)
FINGER_BODIES=('left_lf_rot','left_lf_down','left_rf_rot','left_rf_down')

def target_nonfinger_clearance(position, rotation, width, beads, *, radius=.0125, plane_z=None):
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
    for shape in hand_primitives(float(width)):
        if shape.name.split(':')[0] in FINGER_BODIES:
            continue
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


_BOARD_QUERY=None

def calibrated_source_rgb_supported(observed):
    """Accept the actual calibrated multiview fit without changing its label.

    Plain CAD seeds remain optimizer proposals. Only independently supported
    measured RGB poses and that image's complete native q16 permit contact.
    """
    if observed.get('observation_source')!='calibrated_rgb_cad_seeded':return False
    from .modular_controller import _current_rgb_fk_gate
    provenance=observed.get('robot_geometry_provenance',{})
    required_views={'top','front','rear','side'}
    frames=observed.get('frame_sha256',{})
    complete_frames=(isinstance(frames,dict) and required_views.issubset(frames)
        and all(isinstance(frames[v],str) and len(frames[v])==64
            and all(ch in '0123456789abcdef' for ch in frames[v]) for v in required_views))
    return bool(complete_frames
        and provenance.get('coordinate_indices')==list(range(16))
        and provenance.get('timestamp')==observed.get('timestamp')
        and _current_rgb_fk_gate(observed,'red') is not None)

def source_contact_clearance(position,rotation,width,scene,conditional,*,target):
    """Allow only RED versus moving fingers; retain every other public guard."""
    if target!='red' or scene.get('target')!='red':
        raise GeometryQueryRefusal('RED source-contact diagnostic requires its explicit static-prefix scene')
    measurement=scene['measurement']
    source=measurement.get('observation_source')
    if (source not in ('rgb_shape_geometry','calibrated_rgb_cad_seeded')
        or (source=='calibrated_rgb_cad_seeded' and not calibrated_source_rgb_supported(measurement))
        or not measurement.get('frame_sha256') or not {'board','red'}.issubset(measurement.get('poses',{}))):
        raise GeometryQueryRefusal('Contact source guard requires actual current board and RED RGB')
    b,position,rotation,width=validate_query_inputs(measurement['poses']['board'],position,rotation,width)
    item=scene['occupancy']['red']
    if item['source']!='current_rgb_measurement' or item['inflation_m']!=0. or len(item['pose_hypotheses'])!=1:
        raise GeometryQueryRefusal('RED target must remain a single current RGB measurement')
    if not np.array_equal(np.asarray(item['pose_hypotheses'][0]),np.asarray(measurement['poses']['red'])):
        raise GeometryQueryRefusal('Target guard and current RGB pose disagree')
    retained=conditional.hand_clearance(position,rotation,width,scene,exclude=('red',),plane_z=float(b[2]-.003))
    validate_result(retained,'source non-target/table/base/hinge guards')
    pose=finite_array(item['pose_hypotheses'][0],(7,),'current RED pose')
    local,radii=conditional.local_sphere_cover('red')
    centers=local@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]
    target_guard=target_nonfinger_clearance(position,rotation,width,centers,radius=radii)
    validate_result(target_guard,'RED versus wrist/camera guard')
    global _BOARD_QUERY
    if _BOARD_QUERY is None:_BOARD_QUERY=PublicBoardQuery()
    board=_BOARD_QUERY.query(b,position,rotation,width)
    validate_result(board,'source finite board guard',require_native_witness=True)
    guards=dict(non_target_table_base_hinge=retained,target_wrist_camera=target_guard,finite_board=board)
    key=min(guards,key=lambda name:guards[name]['clearance_m'])
    answer=dict(clearance_m=guards[key]['clearance_m'],witness=dict(component=key,detail=guards[key].get('witness')),
        components={name:value['clearance_m'] for name,value in guards.items()},
        allowed_contact='RED current bead/neck cover versus26 moving finger primitives ONLY')
    return validate_result(answer,'explicit RED source diagnostic result')
