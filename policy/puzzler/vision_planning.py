"""Robot-encoder and kinematic-model planning only; no object-state reads."""
import numpy as np


def target_pose(position, rotation, arm_id):
    """Validate the public command before it reaches the native IK solver."""
    if arm_id not in (0, 1):
        raise ValueError('Unknown robot arm')
    position = np.asarray(position, dtype=float)
    quaternion = np.asarray([rotation[k] for k in ('x', 'y', 'z', 'w')], dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if (position.shape != (3,) or quaternion.shape != (4,) or
            not np.isfinite(position).all() or not np.isfinite(quaternion).all() or
            not np.isfinite(norm) or norm < 1.e-8):
        raise ValueError('IK target must be a finite position and nonzero quaternion')
    return position, quaternion / norm


def finite_fk(ik, joints):
    """Reject invalid native feedback instead of authorizing a NaN path."""
    position, quaternion = ik.fk_ee(joints)
    position = np.asarray(position, dtype=float)
    quaternion = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if (position.shape != (3,) or quaternion.shape != (4,) or
            not np.isfinite(position).all() or not np.isfinite(quaternion).all() or
            not np.isfinite(norm) or norm < 1.e-8):
        raise ValueError('Native FK returned an invalid TCP pose')
    return position, quaternion / norm


def robot_seed(env):
    """Neutral scene coordinates plus robot encoders only.

    A Newton joint vector also contains free-object poses. Never copy that
    vector into the planner: read only the robot's declared coordinate slices.
    Unrelated free joints are assigned identity transforms, not live poses.
    """
    model=env.model
    seed=np.zeros(int(model.joint_coord_count),dtype=float)
    starts=model.joint_q_start.numpy()
    kinds=model.joint_type.numpy()
    for kind,start in zip(kinds,starts[:-1]):
        if int(kind)==4:seed[int(start)+6]=1.  # Newton FREE: xyz + xyzw
        elif int(kind)==2:seed[int(start)+3]=1.  # Newton BALL: xyzw
    coords=sorted({int(i) for arm in env.handles.arms.values()
                   for i in (*arm.arm_coords,*arm.gripper_coords)})
    for i in coords:
        # Slice before the readback; no non-robot coordinate crosses this seam.
        seed[i]=float(env.state.joint_q[i:i+1].numpy()[0])
    return seed

def register(registry,env,bus):
    pose_cache={0:[],1:[]}
    def plan_pose(position:list,rotation:dict,arm_id:int=0,min_joint_limit_margin_rad:float=0.):
        """RSS full-pose IK using live, nearby verified and home seeds; no state writes."""
        if not np.isfinite(min_joint_limit_margin_rad)or min_joint_limit_margin_rad<0:raise ValueError('Joint reserve must be finite and nonnegative')
        position,q=target_pose(position,rotation,arm_id)
        def solve():
            ik=env.arm('left' if arm_id==0 else 'right')
            seed=robot_seed(env)
            if not np.isfinite(seed).all():
                raise ValueError('Robot encoder seed is not finite')
            def distance(entry):
                p0,q0,_=entry
                angle=2*np.arccos(np.clip(abs(np.dot(q0,q))/(np.linalg.norm(q0)*np.linalg.norm(q)),0,1))
                return np.linalg.norm(np.asarray(position)-p0)+.05*angle
            nearby=sorted(pose_cache[arm_id],key=distance)[:3]
            seeds=[('live',seed.copy())]
            for _,_,joints in nearby:
                ss=seed.copy();ss[ik.arm_coords]=joints;seeds.append(('nearby_verified',ss))
            ss=seed.copy();ss[ik.arm_coords]=ik._home_q;seeds.append(('home',ss))
            best=None;attempts=0
            lower=ik.model.joint_limit_lower.numpy()[ik.arm_dofs]
            upper=ik.model.joint_limit_upper.numpy()[ik.arm_dofs]
            if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower>=upper):
                raise ValueError('Native joint limits must be finite and ordered')
            for label,ss in seeds:
                attempts+=1
                if not np.isfinite(ss).all():continue
                sol=np.asarray(ik.solve_q_full(ss,position,q),dtype=float)
                if sol.shape!=seed.shape or not np.isfinite(sol).all():continue
                pe,re=ik.pose_errors(sol,position,q)
                joint_margin=float(np.minimum(sol[ik.arm_coords]-lower,upper-sol[ik.arm_coords]).min())
                if not np.isfinite([pe,re,joint_margin]).all() or pe<0 or re<0:continue
                try:finite_fk(ik,sol)
                except ValueError:continue
                cost=pe+.05*re+(1. if joint_margin<min_joint_limit_margin_rad else 0.)
                if best is None or cost<best[0]:best=(cost,sol,pe,re,label,joint_margin)
                if pe<.0001 and re<.001 and joint_margin>=min_joint_limit_margin_rad:break
            if best is None:
                raise ValueError('Native IK produced no finite pose solution')
            cost,sol,pe,re,label,joint_margin=best
            if pe+.05*re<.006:
                pose_cache[arm_id].append((np.asarray(position).copy(),q.copy(),sol[ik.arm_coords].copy()))
                pose_cache[arm_id]=pose_cache[arm_id][-32:]
            return dict(joints=sol[ik.arm_coords].tolist(),current=seed[ik.arm_coords].tolist(),position_error_m=pe,rotation_error_rad=re,
                        seed_kind=label,seed_attempts=attempts,joint_limit_margin_rad=joint_margin)
        return bus.call(solve)
    def plan_joint_path(position:list,rotation:dict,arm_id:int=0,min_joint_limit_margin_rad:float=.035):
        """Robot-only reserved IK endpoint and sampled joint/FK interpolation."""
        endpoint=plan_pose(position,rotation,arm_id,min_joint_limit_margin_rad)
        def solve():
            ik=env.arm('left' if arm_id==0 else 'right');seed=robot_seed(env)
            lower=ik.model.joint_limit_lower.numpy()[ik.arm_dofs];upper=ik.model.joint_limit_upper.numpy()[ik.arm_dofs]
            aa=seed[ik.arm_coords];bb=np.asarray(endpoint['joints'])
            result=dict(feasible=False,path=[],tcp_path=[],errors=[],endpoint_ik=endpoint,requested_min_joint_limit_margin_rad=float(min_joint_limit_margin_rad))
            if endpoint['position_error_m']>.002 or endpoint['rotation_error_rad']>.03 or endpoint['joint_limit_margin_rad']<min_joint_limit_margin_rad:return result
            for u in np.linspace(0,1,max(2,int(np.ceil(np.max(np.abs(bb-aa))/.020)))+1):
                joints=aa+(bb-aa)*u;reserve=float(np.minimum(joints-lower,upper-joints).min())
                if reserve<min_joint_limit_margin_rad:return result
                full=seed.copy();full[ik.arm_coords]=joints;p,q=finite_fk(ik,full)
                result['path'].append(joints.tolist());result['tcp_path'].append(dict(position=p.tolist(),rotation=dict(zip('xyzw',map(float,q)))))
                result['errors'].append(dict(fraction=float(u),joint_limit_margin_rad=reserve))
            result.update(feasible=True,min_joint_limit_margin_rad=min(x['joint_limit_margin_rad'] for x in result['errors']))
            return result
        return bus.call(solve)
    def plan_path(position:list,rotation:dict,arm_id:int=0,min_joint_limit_margin_rad:float=0.):
        """Continuous full-pose Cartesian IK chain; no scene-state mutation."""
        if not np.isfinite(min_joint_limit_margin_rad) or min_joint_limit_margin_rad<0:
            raise ValueError('Joint-limit reserve must be finite and nonnegative')
        target,target_q=target_pose(position,rotation,arm_id)
        def solve():
            from scipy.spatial.transform import Rotation,Slerp
            ik=env.arm('left' if arm_id==0 else 'right')
            seed=robot_seed(env)
            lower=ik.model.joint_limit_lower.numpy()[ik.arm_dofs]
            upper=ik.model.joint_limit_upper.numpy()[ik.arm_dofs]
            def margin(joints):
                return float(np.minimum(joints[ik.arm_coords]-lower,upper-joints[ik.arm_coords]).min())
            def tcp(joints):
                p,q=finite_fk(ik,joints)
                return dict(position=list(map(float,p)),rotation=dict(zip(('x','y','z','w'),map(float,q))))
            path=[];errors=[];tcp_path=[];margins=[]
            def result(feasible):
                return dict(feasible=feasible,path=path if feasible else [],errors=errors,
                            tcp_path=tcp_path,min_joint_limit_margin_rad=min(margins) if margins else None,
                            requested_min_joint_limit_margin_rad=float(min_joint_limit_margin_rad))
            def refuse(reason,fraction):
                errors.append(dict(fraction=float(fraction),reason=reason))
                return result(False)
            if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower>=upper):
                return refuse('Native joint limits must be finite and ordered',0.)
            if not np.isfinite(seed).all():
                return refuse('Robot encoder seed is not finite',0.)
            try:start_p,start_q=finite_fk(ik,seed)
            except ValueError as error:return refuse(str(error),0.)
            start_margin=margin(seed)
            if not np.isfinite(start_margin):
                return refuse('Initial joint-limit margin is not finite',0.)
            angle=(Rotation.from_quat(start_q).inv()*Rotation.from_quat(target_q)).magnitude()
            count=max(2,int(np.ceil(np.linalg.norm(target-start_p)/.010)),int(np.ceil(angle/.06)))
            fractions=np.linspace(0,1,count+1)
            orientations=Slerp([0,1],Rotation.from_quat([start_q,target_q]))(fractions).as_quat()
            path.append(seed[ik.arm_coords].tolist());tcp_path.append(tcp(seed));margins.append(start_margin)
            if min_joint_limit_margin_rad>0 and margins[0]<min_joint_limit_margin_rad:
                errors.append(dict(fraction=0.,position_error_m=0.,rotation_error_rad=0.,
                                   joint_jump_rad=0.,joint_limit_margin_rad=margins[0]))
                return result(False)
            for u,q in zip(fractions[1:],orientations[1:]):
                point=(1-u)*start_p+u*target
                solution=np.asarray(ik.solve_q_full(seed,point,q),dtype=float)
                if solution.shape!=seed.shape or not np.isfinite(solution).all():
                    return refuse('Native IK solution is not finite or has invalid shape',u)
                pe,re=ik.pose_errors(solution,point,q)
                jump=float(np.max(np.abs(solution[ik.arm_coords]-seed[ik.arm_coords])))
                joint_margin=margin(solution)
                if not np.isfinite([pe,re,jump,joint_margin]).all() or pe<0 or re<0:
                    return refuse('Native IK errors or joint-limit margin are invalid',u)
                try:sample=tcp(solution)
                except ValueError as error:return refuse(str(error),u)
                margins.append(joint_margin)
                errors.append(dict(fraction=float(u),position_error_m=pe,rotation_error_rad=re,
                                   joint_jump_rad=jump,joint_limit_margin_rad=joint_margin))
                if pe>.004 or re>.06 or jump>.5 or (min_joint_limit_margin_rad>0 and joint_margin<min_joint_limit_margin_rad):
                    return result(False)
                seed=solution;path.append(seed[ik.arm_coords].tolist())
                tcp_path.append(sample)
            return result(True)
        return bus.call(solve)
    registry.register_callable('puzzle.plan_pose',plan_pose,summary='RSS native full-pose IK with verified nearby and home restarts, no state mutation.',tags=('planning',))
    registry.register_callable('puzzle.plan_path',plan_path,summary='Continuous full-pose Cartesian IK with optional joint reserve and robot-only FK samples; no collision-free guarantee or state writes.',tags=('planning',))
    def plan_pose_sequence(waypoints:list,arm_id:int=0,min_joint_limit_margin_rad:float=.035):
        from .robot_pose_sequence import solve_sequence
        if arm_id not in(0,1)or not np.isfinite(min_joint_limit_margin_rad)or min_joint_limit_margin_rad<0:raise ValueError('Invalid arm/reserve')
        return bus.call(lambda:solve_sequence(env.arm('left'if arm_id==0 else'right'),robot_seed(env),waypoints,min_joint_limit_margin_rad))
    registry.register_callable('puzzle.plan_pose_sequence',plan_pose_sequence,summary='Bounded current robot-encoder Cartesian waypoint IK chain with native FK and joint reserve; no scene state access or motor writes.',tags=('planning',))
    registry.register_callable('puzzle.plan_joint_path',plan_joint_path,summary='Current encoder to live reserved IK endpoint joint path and public robot FK samples. Near-limit start only exits monotonically; no collision-free claim.',tags=('planning',))
