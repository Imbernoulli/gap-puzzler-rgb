"""Public robot-only finite-IK pose sequences with reserved FK samples.
Only the owning adapter reads live robot encoders; this solver receives no scene.
"""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .vision_planning import finite_fk,target_pose


def solve_sequence(ik,seed,waypoints,margin=.035):
    lower=ik.model.joint_limit_lower.numpy()[ik.arm_dofs];upper=ik.model.joint_limit_upper.numpy()[ik.arm_dofs]
    if len(waypoints)<1 or len(waypoints)>10:raise ValueError('Bounded 1..10 pose sequence required')
    current=np.asarray(seed).copy();path=[current[ik.arm_coords].tolist()];p,q=finite_fk(ik,current);tcps=[dict(position=p.tolist(),rotation=dict(zip('xyzw',q.tolist())))];errors=[]
    margins=[float(np.minimum(current[ik.arm_coords]-lower,upper-current[ik.arm_coords]).min())]
    result=dict(feasible=False,path=path,tcp_path=tcps,errors=errors,min_joint_limit_margin_rad=margins[0],requested_min_joint_limit_margin_rad=float(margin))
    if margins[0]<margin:return result
    for idx,waypoint in enumerate(waypoints):
        p0,q0=finite_fk(ik,current);target=np.asarray(waypoint['position']);qr=np.asarray([waypoint['rotation'][k]for k in 'xyzw']);qr/=np.linalg.norm(qr)
        count=max(2,int(np.ceil(np.linalg.norm(target-p0)/.006)),int(np.ceil((Rotation.from_quat(q0).inv()*Rotation.from_quat(qr)).magnitude()/.035)))
        fractions=np.linspace(0,1,count+1);quats=Slerp([0,1],Rotation.from_quat([q0,qr]))(fractions).as_quat()
        for u,qq in zip(fractions[1:],quats[1:]):
            point=p0*(1-u)+target*u;solution=np.asarray(ik.solve_q_full(current,point,qq));pe,re=ik.pose_errors(solution,point,qq);jump=float(np.max(np.abs(solution[ik.arm_coords]-current[ik.arm_coords])));res=float(np.minimum(solution[ik.arm_coords]-lower,upper-solution[ik.arm_coords]).min());err=dict(segment=idx,fraction=float(u),position_error_m=float(pe),rotation_error_rad=float(re),joint_jump_rad=jump,joint_limit_margin_rad=res);errors.append(err)
            if pe>.002 or re>.03 or jump>.5 or res<margin:
                result['min_joint_limit_margin_rad']=min(margins+[res]);return result
            current=solution;p,q=finite_fk(ik,current);path.append(current[ik.arm_coords].tolist());tcps.append(dict(position=p.tolist(),rotation=dict(zip('xyzw',q.tolist()))));margins.append(res)
    result.update(feasible=True,min_joint_limit_margin_rad=min(margins));return result
