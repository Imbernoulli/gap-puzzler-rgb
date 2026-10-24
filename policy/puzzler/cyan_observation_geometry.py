"""Pure public-CAD camera visibility ranking; proposes poses, never RGB poses.

Inputs are ambiguous RGB candidates, current robot TCP/jaw width and calibrated
cameras. Candidate transforms are applied equally to every hypothesis. No
render, Env, simulator state, target pose or hypothesis selection is used.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from puzzler import model
from puzzler.gripper_geometry import hand_primitives


def _hand_ray_hits(origin, rays, tcp, rotation, width, angular_padding):
    """Conservative projection footprint of public TCP-attached collider CAD."""
    blocked=np.zeros(len(rays),dtype=bool)
    for shape in hand_primitives(float(width)):
        center=tcp+rotation@shape.center;axes=rotation@shape.rotation
        if shape.kind=='box':
            local_o=(origin-center)@axes;local_d=rays@axes
            pad=np.linalg.norm(center-origin)*angular_padding
            half=np.asarray(shape.size)+pad
            safe=np.where(np.abs(local_d)>1.e-12,local_d,np.where(local_d>=0,1.e-12,-1.e-12))
            a=(-half-local_o)/safe;b=(half-local_o)/safe
            near=np.minimum(a,b).max(axis=1);far=np.maximum(a,b).min(axis=1)
            blocked|=(far>=np.maximum(near,0.))
            continue
        if shape.kind=='sphere':
            points=center[None,:];radius=float(shape.size[0])
        else:
            radius,half=map(float,shape.size[:2]);count=max(2,int(np.ceil(2*half/.003))+1)
            points=center+np.linspace(-half,half,count)[:,None]*axes[:,2]
            # Cover the entire capsule, including the region between samples.
            radius=float(np.hypot(radius,half/(count-1)))
        delta=points-origin;along=rays@delta.T
        perpendicular2=np.maximum(np.sum(delta*delta,axis=1)[None,:]-along*along,0.)
        padded=radius+np.linalg.norm(delta,axis=1)*angular_padding
        blocked|=((along>0)&(perpendicular2<=padded[None,:]**2)).any(axis=1)
    return blocked


def score_observation_pose(tcp_pose,width,candidate_poses,cameras,new_tcp_pose):
    """Rank a robot pose by robust separation of both current RGB hypotheses.

    Four-pixel centre-support tolerance matches shape_vision. A projected CAD
    background prediction is only a viewpoint proposal; fresh actual RGB and
    the unchanged estimator still decide whether any pose is measured.
    """
    tcp=np.asarray(tcp_pose,float);new=np.asarray(new_tcp_pose,float)
    r0=Rotation.from_quat(tcp[3:]).as_matrix();r1=Rotation.from_quat(new[3:]).as_matrix();delta=r1@r0.T
    original=[];clouds=[]
    for pose in candidate_poses:
        p=np.asarray(pose,float);cloud=model.local_beads('cyan')@Rotation.from_quat(p[3:]).as_matrix().T+p[:3]
        original.append(cloud);clouds.append((cloud-tcp[:3])@delta.T+new[:3])
    hypotheses=[]
    for i,true in enumerate(clouds):
        alternatives=[]
        for j,false in enumerate(clouds):
            if i==j:continue
            distinct=np.linalg.norm(original[j][:,None]-original[i][None,:],axis=2).min(axis=1)>.003
            per_view={};view_background_counts=np.zeros(len(false),dtype=int);clear_actual_counts=np.zeros(len(true),dtype=int)
            for key,camera in cameras.items():
                origin=camera.camera_to_world[:3,3];padding=4./camera.K[0,0]
                false_rays=(false-origin);false_rays/=np.linalg.norm(false_rays,axis=1)[:,None]
                td=true-origin;along=false_rays@td.T
                perpendicular=np.sqrt(np.maximum(np.sum(td*td,axis=1)[None,:]-along*along,0.))
                gap=(perpendicular-model.RADIUS)/np.maximum(np.linalg.norm(td,axis=1)[None,:],1.e-12)*camera.K[0,0]
                gap=np.where(along>0,gap,np.inf).min(axis=1)
                hand=_hand_ray_hits(origin,false_rays,new[:3],r1,width,padding)
                background=(gap>4.)&~hand
                view_background_counts+=background
                true_rays=(true-origin);true_rays/=np.linalg.norm(true_rays,axis=1)[:,None]
                true_hand=_hand_ray_hits(origin,true_rays,new[:3],r1,width,0.)
                clear_actual_counts+=~true_hand
                per_view[key]=dict(nonshared_bead_background_prediction=background[distinct].tolist(),
                    nonshared_bead_body_gap_pixels=gap[distinct].tolist(),
                    nonshared_bead_hand_footprint=hand[distinct].tolist(),
                    actual_bead_hand_clear=(~true_hand).tolist())
            distinct_counts=view_background_counts[distinct]
            alternatives.append(dict(false_hypothesis=j,nonshared_beads=np.flatnonzero(distinct).tolist(),
                minimum_background_views=int(distinct_counts.min()) if len(distinct_counts) else 0,
                total_background_views=int(distinct_counts.sum()),views=per_view,
                minimum_actual_center_hand_clear_views=int(clear_actual_counts.min()),
                rear_actual_center_hand_clear=per_view.get('rear',{}).get('actual_bead_hand_clear',[])))
        hypotheses.append(dict(hypothesis=i,alternatives=alternatives))
    pairs=[a for h in hypotheses for a in h['alternatives']]
    robust=min((a['minimum_background_views'] for a in pairs),default=0)
    total=sum(a['total_background_views'] for a in pairs)
    visible=min((a['minimum_actual_center_hand_clear_views'] for a in pairs),default=0)
    # The ranking is purely geometric, and includes both directions of the
    # ambiguity. It cannot select which hypothesis is the actual object pose.
    return dict(robust_background_views=robust,total_background_views=total,
                minimum_actual_center_hand_clear_views=visible,hypotheses=hypotheses,
                score=float(100*robust+10*visible+total))


def rank_bounded_observation_poses(tcp_pose,width,candidate_poses,cameras):
    tcp=np.asarray(tcp_pose,float);r0=Rotation.from_quat(tcp[3:]).as_matrix();proposals=[]
    specs=[('rotation',axis,value) for axis in range(3) for value in (-25.,-15.,15.,25.)]
    specs += [('translation',axis,value) for axis in range(3) for value in (-.020,-.010,.010,.020)]
    for kind,axis,value in specs:
        new=tcp.copy()
        if kind=='rotation':
            rotvec=np.zeros(3);rotvec[axis]=np.deg2rad(value);new[3:]=Rotation.from_matrix(Rotation.from_rotvec(rotvec).as_matrix()@r0).as_quat()
        else:new[axis]+=value
        detail=score_observation_pose(tcp,width,candidate_poses,cameras,new)
        proposals.append(dict(kind=kind,axis='xyz'[axis],delta=value,position=new[:3].tolist(),rotation_xyzw=new[3:].tolist(),**detail))
    return sorted(proposals,key=lambda row:(-row['score'],abs(row['delta'])))
