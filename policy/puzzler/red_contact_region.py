"""Public bead sphere and homologous jaw-blade corridor; no simulator/state."""
import numpy as np
from scipy.spatial.transform import Rotation
from puzzler import model
from puzzler.gripper_geometry import hand_primitives
from puzzler.public_board_clearance import (GeometryQueryRefusal,finite_array,
    finite_scalar,validate_quaternion,validate_rotation)

PAIRS=(('left_lf_down:9','left_rf_down:22'),('left_lf_down:10','left_rf_down:23'))

def box_sphere_distance(center,box_center,box_rotation,half,radius):
    delta=np.abs(box_rotation.T@(center-box_center))-half
    return float(np.linalg.norm(np.maximum(delta,0.))+min(float(delta.max()),0.)-radius)

def selected_bead_region(position,rotation,width,red_pose,*,selected_bead_index=4,
                         max_target_finger_penetration_m=0.003):
    position=finite_array(position,(3,),'actual TCP')
    rotation=validate_rotation(rotation,'actual TCP rotation')
    width=finite_scalar(width,'actual encoder jaw width')
    body=finite_array(red_pose,(7,),'fresh RED pose')
    validate_quaternion(body[3:],'fresh RED quaternion')
    if not 0.<width<=.095:raise GeometryQueryRefusal('Actual open jaw width is outside public range')
    if int(selected_bead_index) not in range(len(model.local_beads('red'))):
        raise GeometryQueryRefusal('Selected RED bead index is outside public CAD')
    selected_bead_index=int(selected_bead_index)
    bead=body[:3]+Rotation.from_quat(body[3:]).as_matrix()@model.local_beads('red')[selected_bead_index]
    local=rotation.T@(bead-position)
    opened={s.name:s for s in hand_primitives(width)}
    closed={s.name:s for s in hand_primitives(0.)}
    rows=[]
    for aa,bb in PAIRS:
        a,b=opened[aa],opened[bb]
        if a.kind!='box'or b.kind!='box'or not np.allclose(a.size,b.size,atol=1.e-12,rtol=0.):
            raise GeometryQueryRefusal('Public homologous blade dimensions changed')
        middle=(a.center+b.center)/2.;axis=a.center-b.center
        length=float(np.linalg.norm(axis))
        if length<=0.:raise GeometryQueryRefusal('Public opposed blade centers are degenerate')
        axis/=length
        if abs(abs(float(axis@a.rotation[:,1]))-1.)>1.e-9:
            raise GeometryQueryRefusal('Public blade closing axis changed')
        inner_half=length/2.-float(a.size[1])
        corridor_half=a.size.copy();corridor_half[1]=inner_half
        corridor_distance=box_sphere_distance(local,middle,a.rotation,corridor_half,model.RADIUS)
        penetration=float(max_target_finger_penetration_m)
        fits=(abs(float((local-middle)@axis))+model.RADIUS<=inner_half+penetration
              and -penetration<=corridor_distance<=0.)
        distances=[box_sphere_distance(local,closed[name].center,closed[name].rotation,closed[name].size,model.RADIUS)for name in(aa,bb)]
        passed=bool(fits and corridor_distance<=0. and all(d<=0. for d in distances))
        rows.append(dict(blades=[aa,bb],open_inner_half_gap_m=inner_half,
            selected_sphere_between_open_inner_planes=bool(fits),finite_corridor_sphere_distance_m=corridor_distance,
            permitted_target_finger_penetration_m=penetration,
            hypothetical_close0_blade_sphere_distances_m=distances,both_opposed_closed_blades_overlap=all(d<=0. for d in distances),passes=passed))
    return dict(passes=any(row['passes']for row in rows),selected_bead_index=selected_bead_index,bead_radius_m=model.RADIUS,
        selected_bead_tcp_local=local.tolist(),actual_width_m=width,pairs=rows,
        definition='Full bead sphere fits between open inner planes; sphere intersects the finite extrusion of a homologous blade footprint through that gap; both opposed homologous solid boxes intersect that same sphere at hypothetical close0. No expanded TCP pose-error tolerance is used.')
