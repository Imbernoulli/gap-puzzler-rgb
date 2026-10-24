"""RGB-driven motion and CAD grasp search. No simulator/evaluator imports."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from . import model as m
from . import conditional_scene

MOTION_SCALE=1.

class PlanningRefusal(RuntimeError):
    """A screened motion failed before sending any motor command."""

class MotorAccuracyError(RuntimeError):
    """The measured hand is stationary but outside the requested pose region."""
    def __init__(self,message,actual,rotation,pe,re):
        super().__init__(message)
        self.actual=actual;self.rotation=rotation;self.pe=pe;self.re=re

def params(path):
    global MOTION_SCALE
    p=json.loads((Path(path).resolve().parent.parent/'params.json').read_text())
    MOTION_SCALE=p.get('motion_scale',1.)
    return p
def pose(p,q):
    return dict(position=dict(zip('xyz',map(float,p))),rotation=q)
def qarr(q):return np.array([q[k] for k in ['x','y','z','w']])
def qdict(q):return dict(zip(['x','y','z','w'],map(float,q)))
def state(ctx,hints=None):return ctx.tool('vision.observe',hints=hints)

def await_motion(ctx,p,q,arm,contact=False,accuracy=.0025):
    """A streamed trajectory is not evidence that the servo has arrived.

    Require encoder-derived TCP accuracy and repeated-pose stability before
    closing/opening the jaws or taking another object measurement.
    """
    target=np.asarray(p);rotation=Rotation.from_quat(qarr(q))
    previous,previous_q=hand(ctx,arm)
    for _ in range(12):
        waited=ctx.tool('robot.wait_steps',steps=5)
        if waited is not None and waited.get('waited',5)<5:
            raise RuntimeError('Motor settling wait did not complete')
        actual,actual_q=hand(ctx,arm)
        r=Rotation.from_quat(qarr(actual_q))
        pe=float(np.linalg.norm(actual-target))
        re=float((rotation.inv()*r).magnitude())
        drift=float(np.linalg.norm(actual-previous))
        angular_drift=float((Rotation.from_quat(qarr(previous_q)).inv()*r).magnitude())
        # Contact approaches have explicit geometric regions. The bead grasp
        # permits upward fingertip/table compliance while retaining tight xy;
        # it still requires a subsequent RGB lift check to establish a grasp.
        if contact=='brace':
            position_ok=np.linalg.norm(actual[:2]-target[:2])<.010 and -.002<actual[2]-target[2]<.020
        elif contact=='grasp':
            position_ok=np.linalg.norm(actual[:2]-target[:2])<m.RADIUS and -.002<actual[2]-target[2]<m.RADIUS
        else:position_ok=pe<accuracy
        if position_ok and re<(.12 if contact=='brace' else .04) and drift<.0004 and angular_drift<.015:return
        previous,previous_q=actual,actual_q
    message=f'Motor did not settle at requested pose: position={pe:.6f}m rotation={re:.6f}rad drift={drift:.6f}m arm={arm} contact={contact} target={list(map(float,target))} actual={actual.tolist()}'
    if drift<.0004 and angular_drift<.015:
        raise MotorAccuracyError(message,actual,actual_q,pe,re)
    raise RuntimeError(message)

def feed_joint_plan(ctx,plan,arm):
    a=np.array(plan['current']);b=np.array(plan['joints'])
    count=max(20,min(120,int(np.max(np.abs(b-a))/.018)))
    count=max(14,int(count/MOTION_SCALE))
    path=[]
    for u in np.linspace(0,1,count):
        t=u*u*(3-2*u);path.append({'positions':(a+(b-a)*t).tolist()})
    ctx.tool('robot.execute_trajectory',trajectory={'waypoints':path},arm_id=arm,max_steps_per_waypoint=2)


def settle_motion(ctx,p,q,arm,contact,accuracy):
    command=np.asarray(p,dtype=float).copy();command_R=Rotation.from_quat(qarr(q))
    previous_error=float('inf')
    for attempt in range(3):
        try:
            await_motion(ctx,p,q,arm,contact=contact,accuracy=accuracy)
            return
        except MotorAccuracyError as error:
            # Small free-space load offsets can be corrected using encoders.
            # A large or non-improving obstruction must not be forced through.
            if contact or accuracy>=.005 or attempt==2 or error.pe>.006 or error.re>.04 or error.pe>=previous_error*.85:
                raise
            previous_error=error.pe
            command+=np.asarray(p)-error.actual
            command_R=Rotation.from_quat(qarr(q))*Rotation.from_quat(qarr(error.rotation)).inv()*command_R
            if np.linalg.norm(command-np.asarray(p))>.006 or (Rotation.from_quat(qarr(q)).inv()*command_R).magnitude()>.06:raise
            correction=ctx.tool('puzzle.plan_pose',position=command.tolist(),rotation=qdict(command_R.as_quat()),arm_id=arm)
            if correction['position_error_m']>.002 or correction['rotation_error_rad']>.03:
                raise RuntimeError('Encoder correction planning refused after motion '+str(correction))
            if np.max(np.abs(np.asarray(correction['joints'])-np.asarray(correction['current'])))>.12:
                raise RuntimeError('Encoder correction would change the IK branch')
            feed_joint_plan(ctx,correction,arm)


def move(ctx,p,q,arm=1,steps=180,cartesian=False,contact=False,accuracy=.006):
    if cartesian:
        result=ctx.tool('puzzle.plan_path',position=list(map(float,p)),rotation=q,arm_id=arm)
        if not result['feasible']:raise PlanningRefusal('Continuous Cartesian planning refused '+str(result['errors'][-1]))
        path=[];joints=np.asarray(result['path'])
        for a,b in zip(joints[:-1],joints[1:]):
            count=max(1,int(np.ceil(np.max(np.abs(b-a))/(.018*MOTION_SCALE))))
            path.extend({'positions':(a+(b-a)*u).tolist()} for u in np.linspace(0,1,count+1)[1:])
        ctx.tool('robot.execute_trajectory',trajectory={'waypoints':path},arm_id=arm,max_steps_per_waypoint=2)
        settle_motion(ctx,p,q,arm,contact,accuracy)
        return {'outcome':'reached','plan':result}
    plan=ctx.tool('puzzle.plan_pose',position=list(map(float,p)),rotation=q,arm_id=arm)
    if plan['position_error_m']>.002 or plan['rotation_error_rad']>.03:
        raise PlanningRefusal('Full-pose planning refused '+str(plan))
    feed_joint_plan(ctx,plan,arm)
    settle_motion(ctx,p,q,arm,contact,accuracy)
    return {'outcome':'reached','plan':plan}
def grip(ctx,arm,width):ctx.tool('robot.set_grip',arm_id=arm,width_m=width,ramp_steps=18,settle_steps=15)
def frame(ctx,heading,arm,approach=None):
    kwargs=dict(close_heading_deg=float(heading),arm_id=arm)
    if approach is not None:kwargs['approach']=dict(zip('xyz',map(float,approach)))
    return ctx.tool('robot.grasp_frame',**kwargs)['rotation']
def hand(ctx,arm):
    h=ctx.tool('robot.get_ee_pose',arm_id=arm)
    h=h.get('pose',h)
    return np.array([h['position'][k] for k in 'xyz']),h['rotation']


def part_hand_gap(name,body,tcp):
    body=np.asarray(body)
    centers=m.local_beads(name)@Rotation.from_quat(body[3:]).as_matrix().T+body[:3]
    return float(np.linalg.norm(centers-np.asarray(tcp),axis=1).min())

def withdraw(ctx,arm):
    """Clear the released part before changing the empty hand's orientation.

    Keeping an image-corrected grasp orientation for the entire ascent can hit
    a wrist limit. First lift straight up with the measured orientation, then
    test continuous paths to calibrated top-down hand frames at a safe height.
    A refused path has sent no motor commands; execution failures propagate.
    """
    tcp,q=hand(ctx,arm)
    for distance in (.035,.025,.015):
        try:
            move(ctx,tcp+[0,0,distance],q,arm,cartesian=True)
            break
        except PlanningRefusal:
            continue
    else:raise PlanningRefusal('No reachable initial empty-hand lift after release')
    tcp,q=hand(ctx,arm)
    target=np.array([tcp[0],tcp[1],max(.90,tcp[2]+.025)])
    alternatives=[q,*[frame(ctx,heading,arm) for heading in (0.,90.,180.,270.)]]
    alternatives.sort(key=lambda r:-abs(float(np.dot(qarr(q),qarr(r)))))
    for rotation in alternatives:
        try:
            move(ctx,target,rotation,arm,cartesian=True)
            return
        except PlanningRefusal:
            continue
    raise PlanningRefusal('No reachable empty-hand withdrawal after release')

def anchor(ctx,p):
    if not p.get('anchor',True):
        s=required(ctx,['board'])
        return dict(route='ready',initial_board=s['poses']['board'],brace=False)
    arm=int(p.get('support_arm',0))
    if p.get('safe_anchor'):
        tcp,q0=hand(ctx,arm)
        if tcp[2]<.865:move(ctx,[tcp[0],tcp[1],.865],q0,arm)
    s=state(ctx);b=np.array(s['poses']['board']);R=Rotation.from_quat(b[3:]).as_matrix()
    grasp=b[:3]+R@np.array([-.078,.145 if arm==0 else -.145,.020])
    q=None
    for heading in ((-90.,90.) if arm==0 else (90.,270.)):
        candidate=frame(ctx,heading,arm)
        checks=[ctx.tool('puzzle.plan_pose',position=position.tolist(),rotation=candidate,arm_id=arm)
                for position in (grasp+[0,0,.09],grasp)]
        if all(c['position_error_m']<.002 and c['rotation_error_rad']<.03 for c in checks):
            q=candidate;break
    if q is None:raise PlanningRefusal('No feasible wrist orientation for board brace')
    if not p.get('safe_anchor'):grip(ctx,arm,.008)
    move(ctx,grasp+[0,0,.09],q,arm)
    if p.get('safe_anchor'):grip(ctx,arm,.008)
    if p.get('anchor',True):
        move(ctx,grasp,q,arm,steps=100,contact='brace')
    return dict(route='ready',initial_board=b.tolist(),hand=hand(ctx,arm)[0].tolist())

def plan(ctx,name,p,arm=1,*,observation=None,planning_scene=None):
    s=state(ctx) if observation is None else observation
    poses=s['poses'];b=np.array(poses['board']if'board'in poses else planning_scene['board_for_planning']['pose']);body=np.array(poses[name])
    if planning_scene is not None and (not p.get('conditional_planning') or not conditional_scene.matches_measurement(planning_scene,s)):
        raise PlanningRefusal('Conditional planning scene differs from its raw current RGB source')
    if planning_scene is None and set(m.CELLS)-set(poses):
        raise RuntimeError('Cannot plan grasp with unobserved obstacles: '+str(sorted(set(m.CELLS)-set(poses))))
    R=Rotation.from_quat(body[3:]).as_matrix();Rb=Rotation.from_quat(b[3:]).as_matrix()
    from puzzler.vision_observer import goals
    target_center,target_R=goals(p.get('mode','flat'),p.get('pyramid_plan','pyramid_robot6_plan.json'))[name]
    # Footprint is solved, while grasp candidates are geometrically outside the part.
    local=m.local_beads(name)
    yaw=math.degrees(Rotation.from_quat(body[3:]).as_euler('xyz')[2])
    rotations=[target_R]
    if p.get('shape_symmetry'):
        for k in range(4):
            for flip in (False,True):
                symmetry=m.rotation(k*np.pi/2)@(np.diag([1.,-1.,-1.]) if flip else np.eye(3))
                transformed=local@symmetry.T
                if np.max(np.min(np.linalg.norm(transformed[:,None]-local[None,:],axis=2),axis=1))<1e-7:
                    candidate=target_R@symmetry
                    if not any(np.allclose(candidate,r) for r in rotations):rotations.append(candidate)
    # A source-only grasp has not yet committed to the public target frame.
    # Applying the target approach normal here can reject every valid source
    # orientation (CYAN's source body normal is opposite its public goal
    # normal).  Target-frame direction is screened only after a held belief
    # exists, during transport/release planning.
    if 'target_normal' in p and not p.get('source_grasp_only'):
        normal=np.asarray(p['target_normal'],dtype=float)
        normal=normal/np.linalg.norm(normal)
        rotations=[r for r in rotations if np.dot(r[:,2],normal)>.99]
        if not rotations:raise ValueError(f'No rigid shape symmetry matches target normal for {name}')
    shape_local=local.copy()
    pairs=p.get('grasp_pairs',[])
    if pairs:
        for i,j in pairs:
            if abs(np.linalg.norm(shape_local[i]-shape_local[j])-m.PITCH)>1e-6:
                raise ValueError(f'Grasp pair {(i,j)} must be adjacent beads')
        local=np.concatenate((local,[shape_local[[i,j]].mean(axis=0) for i,j in pairs]))
        p={**p,'grasp_indices':list(range(len(shape_local),len(local)))}
    obstacles=(conditional_scene.spheres(planning_scene,exclude=(name,))[0] if planning_scene is not None else
        np.concatenate([m.local_beads(n)@Rotation.from_quat(np.array(bp)[3:]).as_matrix().T+np.array(bp)[:3]
                        for n,bp in poses.items() if n not in ('board',name)]))
    source_beads=shape_local@R.T+body[:3]
    if p.get('hand_clearance'):
        from puzzler.gripper_geometry import scene_clearance as hand_clearance
        table_z=float(b[2]-.003)  # Public board CAD on its observed level support.
        margin=p.get('hand_clearance_margin',.0005)
    candidates=[]
    for candidate_R in rotations:
      heights=(local@candidate_R.T)[:,2]
      for i,a in enumerate(local):
        if 'grasp_indices' in p and i not in p['grasp_indices']:continue
        headings=p.get('grasp_headings',[90.,0.,270.,180.]) if p.get('grasp_auto') else ([float(p['world_heading'])-yaw] if 'world_heading' in p else ([float(p['grasp_heading'])] if 'grasp_heading' in p else (0.,90.)))
        for heading in headings:
            direction=np.array([math.cos(math.radians(heading)),math.sin(math.radians(heading)),0.])
            neighbours=shape_local
            if i>=len(shape_local):
                neighbours=np.delete(shape_local,pairs[i-len(shape_local)],axis=0)
            delta=neighbours-a
            blocked=sum(.005<abs(d@direction)<p.get('max_grasp_obstacle_distance',float('inf'))
                        and np.linalg.norm(d-(d@direction)*direction)<p.get('neighbor_clearance',.018) for d in delta)
            clearance=float(np.linalg.norm(obstacles-(body[:3]+R@a),axis=1).min()) if p.get('obstacle_grasp') else 0.
            torque=abs(float(a[0]*direction[1]-a[1]*direction[0])) if p.get('grasp_torque') else 0.
            score=(-float(heights[i]),blocked,torque,-clearance,-float(np.linalg.norm(a))) if p.get('mode')=='pyramid' else (blocked,torque,-clearance,-float(np.linalg.norm(a)))
            if p.get('hand_clearance') and p.get('grasp_stability'):
                # Prefer a short center-of-mass lever once full hand geometry
                # replaces the old terminal-bead heuristic. An outer bead can
                # be reachable yet let a long rigid piece roll out of the jaws.
                lever=float(np.linalg.norm(a))
                axial=abs(float(a[0]*direction[1]-a[1]*direction[0]))
                score=(-float(heights[i]),lever,axial,-clearance) if p.get('mode')=='pyramid' else (lever,axial,-clearance)
            candidates.append((score,i,heading,candidate_R))
    candidates.sort(key=lambda x:x[0])
    best_cost=float('inf')
    refused=dict(neighbor_heuristic=0,source_clearance=0,release_clearance=0,kinematics=0,joint_limit=0)
    heights=p.get('grasp_heights',[p.get('grasp_z',0.)])
    expanded=((score,idx,heading,candidate_R,gz) for gz in heights
              for score,idx,heading,candidate_R in candidates)
    for score,idx,heading,candidate_R,grasp_z in expanded:
      for lean in p.get('grasp_leans',[0.]):
        if score[1 if p.get('mode')=='pyramid' else 0]>0 and p.get('grasp_auto') and not p.get('hand_clearance'):
            refused['neighbor_heuristic']+=1;continue
        actual=body[:3]+R@local[idx]
        learned=p.get('object_relative_grasp_se3')
        world_heading=heading+yaw if p.get('perceive_yaw',True) else heading
        if learned:
            relative_p=np.asarray(learned['position_m'],float);relative_q=np.asarray(learned['quaternion_xyzw'],float)
            if relative_p.shape!=(3,)or relative_q.shape!=(4,)or not np.isfinite(relative_p).all()or not np.isfinite(relative_q).all()or abs(np.linalg.norm(relative_q)-1.)>1.e-4:raise PlanningRefusal('Invalid public object-relative grasp profile')
            grasp=body[:3]+R@relative_p
            learned_hand_R=R@Rotation.from_quat(relative_q).as_matrix();approach=learned_hand_R[:,2]
            q=qdict(Rotation.from_matrix(learned_hand_R).as_quat())
        else:
            grasp=actual+np.array([p.get('pick_bias',0.),0,grasp_z])
            h=np.radians(world_heading);theta=np.radians(lean)
            approach=np.array([np.sin(h)*np.sin(theta),-np.cos(h)*np.sin(theta),-np.cos(theta)])
            q=frame(ctx,world_heading,arm,approach)
        goal_axis=Rb@candidate_R@R.T@approach
        if not p.get('source_grasp_only')and 'min_target_approach_down' in p and goal_axis[2]>-p['min_target_approach_down']:continue
        release_height=p.get('release_height',.025)
        if p.get('hand_clearance'):
            # Check the actual fingers/housing, including other beads of the
            # source part, throughout the open-hand descent. All scene geometry
            # here is reconstructed from this observation, never simulator poses.
            hand_R=Rotation.from_quat(qarr(q)).as_matrix()
            source_scene=np.concatenate((source_beads,obstacles))
            if name=='red'and p.get('source_contact_grasp'):
                from puzzler.red_source_geometry import source_contact_clearance
                source_blocked=any(source_contact_clearance(grasp+[0,0,dz],hand_R,p.get('source_open_width',p.get('open_width',.055)),planning_scene,conditional_scene,target=name)['clearance_m']<margin
                                   for dz in np.linspace(0,p.get('source_hover',.10),7))
            elif name=='cyan'and p.get('source_contact_grasp'):
                from puzzler.cyan_geometry import source_contact_clearance
                source_blocked=any(not source_contact_clearance(grasp+[0,0,dz],hand_R,p.get('source_open_width',p.get('open_width',.055)),planning_scene,p)['passed']
                                   for dz in np.linspace(0,p.get('source_hover',.10),7))
            else:
                source_blocked=any((conditional_scene.hand_clearance(grasp+[0,0,dz],hand_R,p.get('source_open_width',p.get('open_width',.055)),planning_scene,plane_z=table_z)
                    if planning_scene is not None else
                    hand_clearance(grasp+[0,0,dz],hand_R,p.get('source_open_width',p.get('open_width',.055)),source_scene,
                             table_z=table_z,plane_z=table_z,board=b))['clearance_m']<margin
                   for dz in np.linspace(0,p.get('source_hover',.10),7))
            if source_blocked:
                refused['source_clearance']+=1
                continue
            # The held part is intentionally between the jaws. Exclude it from
            # release-obstacle checks; test the other observed parts and a
            # conservative plane at the board rim/socket height. Opening is
            # also screened, rather than checking only the narrow held hand.
            goal_R=Rb@candidate_R@R.T@hand_R
            release=(b[:3]+Rb@target_center+Rb@candidate_R@local[idx]
                     +Rb@candidate_R@R.T@np.array([0.,0.,grasp_z]))
            if not p.get('source_grasp_only'):
                try:
                    release_height=clear_release_height(name,poses,release,qdict(Rotation.from_matrix(goal_R).as_quat()),p,planning_scene=planning_scene)
                except PlanningRefusal:
                    refused['release_clearance']+=1
                    continue
        if p.get('grasp_auto'):
            goal_q=qdict(Rotation.from_matrix(Rb@candidate_R@R.T@Rotation.from_quat(qarr(q)).as_matrix()).as_quat())
            goal_p=b[:3]+Rb@target_center+Rb@candidate_R@local[idx]+Rb@candidate_R@R.T@np.array([0.,0.,grasp_z])
            cost=0.
            feasible=True
            checks=[(goal_p+[0,0,p.get('transport',.10)],goal_q),
                    (grasp+[0,0,p.get('source_hover',.10)],q),
                    (grasp,q),
                    (grasp+[0,0,p.get('lift',.085)],q),
                    (goal_p+[0,0,release_height],goal_q)]
            if p.get('source_grasp_only'):checks=[(grasp+[0,0,p.get('source_hover',.10)],q),(grasp,q)]
            for position,rotation in checks:
                check=ctx.tool('puzzle.plan_pose',position=position.tolist(),rotation=rotation,arm_id=arm)
                cost=max(cost,check['position_error_m']+.05*check['rotation_error_rad'])
                if check['position_error_m']>.002 or check['rotation_error_rad']>.03:
                    feasible=False;break
                # A small pose residual can hide a saturated IK solution. The
                # measured grasp transform will change after closing; leave a
                # configured reserve before committing to the pickup. This is
                # a kinematic screen, not a guarantee against later slipping.
                reserve=p.get('min_ik_joint_margin_rad',0.)
                if reserve>0 and check.get('joint_limit_margin_rad',float('-inf'))<reserve:
                    refused['joint_limit']+=1
                    feasible=False;break
            best_cost=min(best_cost,cost)
            if not feasible:
                refused['kinematics']+=1;continue
        return dict(s=s,body=body,grasp=grasp,rotation=q,index=idx,target_center=target_center,target_R=candidate_R,heading=heading,lean=lean,grasp_height_m=grasp_z,planned_release_height_m=release_height,refused_candidates=refused)
    raise RuntimeError(f'No feasible source/target grasp for {name}: {best_cost}; refusals={refused}')


def required(ctx, names, hints=None):
    s=state(ctx,hints)
    missing=[n for n in names if n not in s['poses']]
    if missing:raise RuntimeError('RGB perception uncertain: '+str({n:s['detections'][n]['reason'] for n in missing}))
    return s


def inspect_released(ctx,name,prediction,arm,support_arm=None):
    """Park the empty work hand before measuring the released scene.

    Partial rim occlusion can bias a board fit without making it disappear.
    Clearing the hand only after a refusal does not address that error.
    """
    hints={name:prediction.tolist()}
    # Withdrawal has already raised the open hand. Park it before requesting
    # a new image; never substitute the last board pose for current evidence.
    ctx.tool('robot.go_home',arm_id=arm)
    if support_arm is not None:
        tcp,q=hand(ctx,support_arm)
        move(ctx,tcp+[0,0,.060],q,support_arm,cartesian=True)
        ctx.tool('robot.go_home',arm_id=support_arm)
    ctx.tool('robot.wait_steps',steps=25)
    return required(ctx,['board',name],hints)


def clear_release_height(name,poses,target_tcp,q,p,*,planning_scene=None):
    """Bounded release-height choice from fresh visual geometry and hand CAD.

    A compliant grasp changes the actual hand/object transform. Screen that
    measured transform again, including the jaw-opening sweep, rather than
    relying on the transform assumed before grasping.
    """
    missing=({'board'}|set(m.CELLS))-set(poses)
    if planning_scene is not None:
        expected=planning_scene['measurement']['poses']
        if (not p.get('conditional_planning') or set(poses)!=set(expected)
                or any(not np.array_equal(np.asarray(poses[n]),np.asarray(expected[n])) for n in poses)):
            raise PlanningRefusal('Conditional release occupancy differs from its raw measured poses')
    if missing and planning_scene is None:
        raise PlanningRefusal('Cannot screen release with unobserved scene geometry: '+str(sorted(missing)))
    from puzzler.gripper_geometry import scene_clearance
    b=np.array(poses['board']if'board'in poses else planning_scene['board_for_planning']['pose']);R=Rotation.from_quat(qarr(q)).as_matrix()
    obstacles=np.concatenate([m.local_beads(n)@Rotation.from_quat(np.array(bp)[3:]).as_matrix().T+np.array(bp)[:3]
                              for n,bp in poses.items() if n not in ('board',name)])
    start=p.get('release_height',.012);limit=p.get('max_release_height',.024)
    for height in np.linspace(start,max(start,limit),7):
        position=np.asarray(target_tcp)+[0,0,height]
        if all((conditional_scene.hand_clearance(position,R,float(w),planning_scene,exclude=(name,),plane_z=float(b[2]+.013))
                if planning_scene is not None else
                scene_clearance(position,R,float(w),obstacles,table_z=float(b[2]-.003),
                               plane_z=float(b[2]+.013),board=b))['clearance_m']>=p.get('hand_clearance_margin',.0005)
               for w in np.linspace(.024,p.get('release_open_width',p.get('open_width',.055)),5)):
            return float(height)
    raise PlanningRefusal('Measured grasp has no clear bounded release pose: '+name)


def inspect_held(ctx,name,prediction,arm,require_board=True,max_moves=None):
    """Change viewpoint via known robot motion when a held part is occluded.

    A rigid-grasp prediction seeds the fit; only new RGB evidence can return
    a pose. This does not attach the part or consult a grasp/contact oracle.
    """
    inspection_origin=prediction.copy()
    def usable(observation,expected):
        if name not in observation['poses']:return None
        if 'board' in observation['poses']or not require_board:return observation
        observed=np.asarray(observation['poses'][name])
        error=float(np.linalg.norm(observed[:3]-expected[:3]))
        # An image-confirmed grasp loss can be handled before the board is
        # visible: clear the empty hand, then take a new board measurement.
        if error>.025:
            tcp,_=hand(ctx,arm)
            if part_hand_gap(name,observed,tcp)>.045:
                return {**observation,'grasp_prediction_error_m':error}
        return None
    measured=state(ctx,{name:prediction.tolist()})
    result=usable(measured,prediction)
    if result is not None:return result
    # Try a higher inspection pose first, then bounded wrist yaw changes.
    tcp,q=hand(ctx,arm)
    inspection=np.array([.60,.28 if arm==0 else -.28,max(.90,tcp[2]+.035)])
    moves=[([0,0,.10],'z',0.),(None,'z',0.),([0,0,0],'z',60.),([0,0,0],'z',-120.),([0,0,0],'x',30.),([0,0,0],'x',-60.),([0,0,0],'y',30.),([0,0,0],'y',-60.)]
    if max_moves is not None:moves=moves[:max(0,int(max_moves))]
    for offset,axis,angle in moves:
        delta=Rotation.from_euler(axis,angle,degrees=True).as_matrix()
        new_q=qdict(Rotation.from_matrix(delta@Rotation.from_quat(qarr(q)).as_matrix()).as_quat())
        # The first lift already changed tcp; keep the declared inspection point absolute.
        target=inspection.copy() if offset is None else tcp+offset
        next_prediction=np.r_[target+delta@(prediction[:3]-tcp),
                              (Rotation.from_matrix(delta)*Rotation.from_quat(prediction[3:])).as_quat()]
        try:move(ctx,target,new_q,arm)
        except PlanningRefusal:continue
        prediction=next_prediction;tcp,q=hand(ctx,arm)
        measured=state(ctx,{name:prediction.tolist()})
        result=usable(measured,prediction)
        if result is not None:return result
        # A part may have dropped while the wrist turned. Retain the original
        # image-search hypothesis as well as the rigid-motion hypothesis; do
        # not rotate every possible part location with an empty gripper.
        recovered=state(ctx,{name:inspection_origin.tolist()})
        result=usable(recovered,prediction)
        if result is not None:return result
    raise RuntimeError('RGB perception remains uncertain after active inspection: '+name)


def _scene_inspection_cloud(observed,held_name):
    """Current RGB measurements plus explicit current RGB pose alternatives.

    Hypotheses bound planning obstacles only. They never fill a missing entry
    in the returned observation or become an accepted object measurement.
    """
    poses=observed.get('poses',{})
    if 'board' not in poses or held_name not in poses:
        raise PlanningRefusal('Scene inspection needs fresh board and held-part RGB poses')
    clouds=[]
    for name in m.CELLS:
        if name==held_name:continue
        options=[poses[name]] if name in poses else observed.get('detections',{}).get(name,{}).get('supported_pose_candidates',[])
        if not options or len(options)>32:
            raise PlanningRefusal('Scene inspection has no bounded current RGB hypotheses for '+name)
        for option in options:
            body=np.asarray(option,dtype=float)
            if body.shape!=(7,) or not np.isfinite(body).all() or np.linalg.norm(body[3:])<1.e-8:
                raise PlanningRefusal('Scene inspection received an invalid RGB hypothesis for '+name)
            clouds.append(m.local_beads(name)@Rotation.from_quat(body[3:]).as_matrix().T+body[:3])
    return np.concatenate(clouds)


def _scene_inspection_candidates(name,body,tcp,q,obstacles):
    """Bounded hand-relative translations ranked by projected public CAD.

    The score proposes a measurement action. Only the RGB acquired after that
    action can establish visibility or resolve any identity ambiguity.
    """
    import cv2
    from puzzler.vision import calibration,rig_specs
    cameras=calibration(rig_specs())
    body=np.asarray(body);tcp=np.asarray(tcp)
    R=Rotation.from_quat(body[3:]).as_matrix()
    local=m.local_beads(name)@R.T
    def mask(camera,centers):
        height,width=int(round(2*camera.K[1,2])),int(round(2*camera.K[0,2]))
        result=np.zeros((height,width),dtype=np.uint8)
        depths=((centers-camera.camera_to_world[:3,3])@camera.camera_to_world[:3,:3])[:,2]
        for pixel,depth in zip(camera.project(centers),depths):
            if depth<=m.RADIUS:continue
            radius=int(np.ceil(camera.K[0,0]*m.RADIUS/(depth-m.RADIUS)))
            cv2.circle(result,tuple(np.rint(pixel).astype(int)),radius,1,-1)
        return result
    scene={key:mask(camera,obstacles) for key,camera in cameras.items()}
    candidates=[]
    for dx in (-.08,-.04,0.,.04,.08):
      for dy in (-.08,-.04,0.,.04,.08):
       for dz in (0.,.04):
        if dx==dy==dz==0:continue
        offset=np.array([dx,dy,dz]);centers=local+body[:3]+offset
        overlap=sum(int((mask(camera,centers)&scene[key]).sum()) for key,camera in cameras.items())
        candidates.append(dict(position=tcp+offset,rotation=dict(q),
                               score=(overlap,float(np.linalg.norm(offset)))))
    return sorted(candidates,key=lambda candidate:candidate['score'])


def _screen_scene_inspection_path(plan,name,body,tcp,q,obstacles,board,p):
    """Screen the complete native plan before any trajectory is sent.

    All supported obstacle alternatives are retained. Hand geometry is checked
    over a conservative jaw-width envelope because this interface has no jaw
    encoder tool. These are sampled local-CAD checks, not whole-arm collision
    freedom or a prediction of physical grasp stability.
    """
    from puzzler.gripper_geometry import scene_clearance
    reserve=max(.035,float(p.get('min_ik_joint_margin_rad',.035)))
    if not plan.get('feasible'):
        raise PlanningRefusal('Scene inspection continuous IK refused')
    path=np.asarray(plan.get('path',[]),dtype=float)
    samples=plan.get('tcp_path',[])
    minimum=plan.get('min_joint_limit_margin_rad')
    if (minimum is None or not np.isfinite(minimum) or minimum<reserve or
            path.ndim!=2 or path.shape[0]<2 or path.shape[1]!=6 or
            not np.isfinite(path).all() or len(samples)!=len(path)):
        raise PlanningRefusal('Scene inspection lacks a complete reserved native path')
    errors=plan.get('errors',[])
    if len(errors)!=len(path)-1 or any(
            not all(np.isfinite(error.get(key,float('nan'))) for key in
                    ('position_error_m','rotation_error_rad','joint_jump_rad','joint_limit_margin_rad')) or
            error['position_error_m']>.004 or error['rotation_error_rad']>.06 or
            error['joint_jump_rad']>.5 or error['joint_limit_margin_rad']<reserve for error in errors):
        raise PlanningRefusal('Scene inspection path evidence does not meet motion limits')
    start_R=Rotation.from_quat(qarr(q)).as_matrix();body=np.asarray(body);board=np.asarray(board)
    relative_p=start_R.T@(body[:3]-tcp)
    relative_R=start_R.T@Rotation.from_quat(body[3:]).as_matrix()
    positions=[];rotations=[]
    for sample in samples:
        point=np.asarray(sample['position'],dtype=float)
        rotation=np.asarray([sample['rotation'][key] for key in 'xyzw'],dtype=float)
        if point.shape!=(3,) or not np.isfinite(point).all() or not np.isfinite(rotation).all() or np.linalg.norm(rotation)<1.e-8:
            raise PlanningRefusal('Scene inspection received an invalid FK sample')
        positions.append(point);rotations.append(Rotation.from_quat(rotation).as_matrix())
    if np.linalg.norm(positions[0]-tcp)>.004 or (Rotation.from_matrix(start_R.T@rotations[0])).magnitude()>.04:
        raise PlanningRefusal('Scene inspection start no longer matches the robot feedback')
    if any((Rotation.from_matrix(start_R.T@R)).magnitude()>.04 for R in rotations):
        raise PlanningRefusal('Scene inspection translation unexpectedly rotates the held part')
    largest_gap=max(np.linalg.norm(b-a) for a,b in zip(positions[:-1],positions[1:]))
    if largest_gap>.018:
        raise PlanningRefusal('Scene inspection FK samples are too sparse for local CAD screening')
    width=max(.055,float(p.get('source_open_width',p.get('open_width',.055))))
    if not np.isfinite(width) or width>.095:
        raise PlanningRefusal('Scene inspection jaw envelope is invalid')
    # Linear finger travel is width/2. Half a width-sample gap bounds the
    # unsampled collider translation; include it and half a TCP sample gap.
    clearance=max(.004,float(p.get('scene_inspection_clearance_m',.004)))+width/16+largest_gap/2
    plane=float(board[2]+.013)
    for point,R in zip(positions,rotations):
        for jaw in np.linspace(0.,width,5):
            result=scene_clearance(point,R,float(jaw),obstacles,
                    table_z=float(board[2]-.003),plane_z=plane,board=board)
            if not np.isfinite(result['clearance_m']) or result['clearance_m']<clearance:
                raise PlanningRefusal('Scene inspection hand CAD clearance refused '+str(result))
        centers=m.local_beads(name)@(R@relative_R).T+point+R@relative_p
        if centers[:,2].min()-m.RADIUS-plane<clearance:
            raise PlanningRefusal('Scene inspection held part is too close to the support')
        if np.linalg.norm(centers[:,None]-obstacles[None,:],axis=2).min()-2*m.RADIUS<clearance:
            raise PlanningRefusal('Scene inspection held part intersects a current RGB hypothesis')
    return plan


def _execute_scene_inspection_path(ctx,plan,position,rotation,arm):
    """Execute the already screened native path without selecting a new plan."""
    waypoints=[];joints=np.asarray(plan['path'])
    for a,b in zip(joints[:-1],joints[1:]):
        count=max(1,int(np.ceil(np.max(np.abs(b-a))/(.018*MOTION_SCALE))))
        waypoints.extend({'positions':(a+(b-a)*u).tolist()} for u in np.linspace(0,1,count+1)[1:])
    executed=ctx.tool('robot.execute_trajectory',trajectory={'waypoints':waypoints},arm_id=arm,max_steps_per_waypoint=2)
    if not isinstance(executed,dict) or executed.get('status')!='ok' or executed.get('success') is False:
        raise RuntimeError('Scene inspection trajectory execution failed: '+str(executed))
    settle_motion(ctx,position,rotation,arm,False,.006)


def inspect_scene_held(ctx,name,prediction,arm,p=None):
    """Acquire fresh held-part, board and obstacle poses with bounded motion.

    A current separated target retains the loss route. Every additional motion
    is continuously preflighted with joint reserve and sampled CAD clearance.
    Planning refusals may try another proposal; execution failures propagate.
    No missing pose is filled from a prior observation, prediction or alias.
    """
    p=p or {};reserve=max(.035,float(p.get('min_ik_joint_margin_rad',.035)))
    maximum=max(0,min(3,int(p.get('scene_inspection_max_moves',3))))
    observed=state(ctx,{name:np.asarray(prediction,dtype=float).tolist()})
    attempted=[];moves=0;prior_timestamp=None
    while True:
        timestamp=observed.get('timestamp')
        if (not isinstance(timestamp,(int,float)) or not np.isfinite(timestamp) or
                not observed.get('frame_sha256') or
                (prior_timestamp is not None and timestamp<=prior_timestamp)):
            raise PlanningRefusal('Scene inspection requires fresh recorded RGB after motion')
        if name not in observed.get('poses',{}):
            raise PlanningRefusal('Scene inspection has no fresh RGB held-part pose')
        tcp,q=hand(ctx,arm);body=np.asarray(observed['poses'][name])
        quaternion=qarr(q)
        if (body.shape!=(7,) or np.asarray(tcp).shape!=(3,) or not np.isfinite(body).all() or
                not np.isfinite(tcp).all() or not np.isfinite(quaternion).all() or
                np.linalg.norm(body[3:])<1.e-8 or np.linalg.norm(quaternion)<1.e-8):
            raise PlanningRefusal('Scene inspection received invalid RGB or robot feedback')
        if part_hand_gap(name,body,tcp)>.045:
            return observed
        missing=set(['board',*m.CELLS])-set(observed['poses'])
        if not missing:return observed
        if moves>=maximum:
            raise PlanningRefusal('Fresh RGB scene remains incomplete after bounded inspection: '+str(sorted(missing)))
        obstacles=_scene_inspection_cloud(observed,name)
        candidates=_scene_inspection_candidates(name,body,tcp,q,obstacles)
        selected=None
        for candidate in candidates[:24]:
            position=np.asarray(candidate['position']);rotation=candidate['rotation']
            if any(np.linalg.norm(position-prior)<.003 for prior in attempted):continue
            attempted.append(position.copy())
            try:
                plan=ctx.tool('puzzle.plan_path',position=position.tolist(),rotation=rotation,
                              arm_id=arm,min_joint_limit_margin_rad=reserve)
                _screen_scene_inspection_path(plan,name,body,tcp,q,obstacles,observed['poses']['board'],p)
            except PlanningRefusal:
                continue
            selected=(plan,position,rotation);break
        if selected is None:
            raise PlanningRefusal('No clear reserved path for a bounded RGB scene inspection')
        plan,position,rotation=selected
        _execute_scene_inspection_path(ctx,plan,position,rotation,arm)
        # The feedback, rather than the requested endpoint, supplies this seed.
        actual,actual_q=hand(ctx,arm)
        delta=Rotation.from_quat(qarr(actual_q)).as_matrix()@Rotation.from_quat(qarr(q)).as_matrix().T
        predicted=np.r_[actual+delta@(body[:3]-tcp),
                         (Rotation.from_matrix(delta)*Rotation.from_quat(body[3:])).as_quat()]
        prior_timestamp=timestamp
        observed=state(ctx,{name:predicted.tolist()});moves+=1


def place(ctx,name,p):
    """Detect, choose source arm, grasp, remeasure, carry, release and remeasure."""
    p={**p,**p.get('pieces',{}).get(name,{})}
    s=required(ctx,['board',name])
    # Arm selection follows the measured source position, not an initial layout.
    arm=0 if s['poses'][name][1] >= 0 else 1
    tcp,q0=hand(ctx,arm)
    if tcp[2]<.865:move(ctx,[tcp[0],tcp[1],.865],q0,arm)
    anchor(ctx,{**p,'support_arm':1-arm})
    pl=plan(ctx,name,p,arm);g=pl['grasp'];q=pl['rotation']
    grip(ctx,arm,p.get('source_open_width',p.get('open_width',.055)))
    move(ctx,g+[0,0,p.get('source_hover',.11)],q,arm)
    move(ctx,g,q,arm,cartesian=True,contact='grasp')
    grip(ctx,arm,p.get('close_width',0.))
    lift=p.get('lift',.085)
    move(ctx,g+[0,0,lift],q,arm)
    prediction=pl['body'].copy();prediction[2]+=lift
    measured=inspect_held(ctx,name,prediction,arm)
    now=np.array(measured['poses'][name])
    rise=float(now[2]-pl['body'][2])
    if rise<.025 or measured.get('grasp_prediction_error_m',0.)>.025:
        grip(ctx,arm,.055)
        withdraw(ctx,arm)
        ctx.tool('robot.go_home',arm_id=arm)
        return dict(route='retry',failure='visually_measured_grasp_miss',piece=name,rise_m=rise)
    tcp,qnow=hand(ctx,arm)
    b=np.array(measured['poses']['board']);Rb=Rotation.from_quat(b[3:]).as_matrix()
    desired_R=Rb@pl['target_R']
    delta=desired_R@Rotation.from_quat(now[3:]).as_matrix().T
    qtarget=qdict(Rotation.from_matrix(delta@Rotation.from_quat(qarr(qnow)).as_matrix()).as_quat())
    target=b[:3]+Rb@pl['target_center']
    target_tcp=target+delta@(tcp-now[:3])
    transport=p.get('transport',.09)
    move(ctx,target_tcp+[0,0,transport],qtarget,arm)
    prediction=np.r_[target+[0,0,transport],Rotation.from_matrix(desired_R).as_quat()]
    carried=inspect_held(ctx,name,prediction,arm)
    body=np.array(carried['poses'][name])
    tcp,qnow=hand(ctx,arm)
    gap=part_hand_gap(name,body,tcp)
    if carried.get('grasp_prediction_error_m',0.)>.025 or (np.linalg.norm(body[:3]-prediction[:3])>.025 and gap>.045):
        # RGB has localized a part away from the predicted rigid grasp. Do not
        # reinterpret that gap as a new hand/object transform and carry on.
        grip(ctx,arm,.055)
        withdraw(ctx,arm)
        after=inspect_released(ctx,name,body,arm,support_arm=1-arm if p.get('anchor',True) else None)
        error=after['assessment']['bead_error_m'][name]
        return dict(route='placed' if error is not None and error<.003 else 'retry',
                    piece=name,visual_error_m=error,hand_bead_gap_m=gap,failure='visually_detected_grasp_loss')
    b=np.array(carried['poses']['board'])
    Rb=Rotation.from_quat(b[3:]).as_matrix();desired_R=Rb@pl['target_R']
    correction=desired_R@Rotation.from_quat(body[3:]).as_matrix().T
    qtarget=qdict(Rotation.from_matrix(correction@Rotation.from_quat(qarr(qnow)).as_matrix()).as_quat())
    target_tcp=b[:3]+Rb@pl['target_center']+correction@(tcp-body[:3])
    release=p.get('release_height',.028)
    if p.get('hand_clearance'):
        release=clear_release_height(name,carried['poses'],target_tcp,qtarget,p)
    move(ctx,target_tcp+[0,0,release],qtarget,arm,accuracy=.0025)
    grip(ctx,arm,.055)
    withdraw(ctx,arm)
    ctx.tool('robot.wait_steps',steps=50)
    prediction=np.r_[b[:3]+Rb@pl['target_center'],Rotation.from_matrix(desired_R).as_quat()]
    after=inspect_released(ctx,name,prediction,arm,support_arm=1-arm if p.get('anchor',True) else None)
    error=after['assessment']['bead_error_m'][name]
    placed=error is not None and error<.003
    return dict(route='placed' if placed else 'retry',piece=name,visual_error_m=error,
                visual_rise_m=rise,release_height_m=release,failure='' if placed else 'visual_goal_residual',
                observation_source=after['observation_source'])


def finish(ctx,p):
    for arm in (1,0):
        tcp,q=hand(ctx,arm)
        grip(ctx,arm,.055)
        if tcp[2]<.9:move(ctx,tcp+[0,0,.05],q,arm)
        ctx.tool('robot.go_home',arm_id=arm)
    ctx.tool('robot.wait_steps',steps=100)
    names=p.get('diagnostic_subset') or list(m.CELLS)
    s=required(ctx,['board',*names])
    ctx.tool('robot.wait_steps',steps=50)
    after=required(ctx,['board',*names])
    # Observable stability is measured from repeated images, never body velocity.
    from scipy.optimize import linear_sum_assignment
    displacement={}
    for n in names:
        a=np.array(s['poses'][n]);b=np.array(after['poses'][n]);local=m.local_beads(n)
        aa=local@Rotation.from_quat(a[3:]).as_matrix().T+a[:3]
        bb=local@Rotation.from_quat(b[3:]).as_matrix().T+b[:3]
        cost=np.linalg.norm(aa[:,None]-bb[None,:],axis=2);ii,jj=linear_sum_assignment(cost)
        displacement[n]=float(cost[ii,jj].max())
    good=all(after['assessment']['bead_error_m'][n]<.003 and displacement[n]<.0015 for n in names)
    return dict(route='complete' if good else 'failed',visual_assessment=after['assessment'],
                displacement_m=displacement,diagnostic_subset=p.get('diagnostic_subset',[]))
