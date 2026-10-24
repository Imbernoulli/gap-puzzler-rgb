"""Four typed Red handoff phases from live RGB, robot q16 and public CAD.

No object state, simulator contacts, saved world poses or joint tapes enter
this policy. Independent graph checkpoints separate close from sender open.
"""
from copy import deepcopy
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation

from . import model, vision_controller as c, visual_memory
from . import modular_red_placement as base
from .modular_controller import result, save_observation, scene_belief, prior_for
from .modular_red_contact import encoder_epoch
from .vision_stages import StageRefusal
from .public_bimanual_geometry import PublicBimanualGeometry

RESERVE = .035
FLOOR = .0005
OPEN = .055
_GEOMETRY = None


class _ReceiverFaceRecenter(Exception):
    """Internal bounded stage recovery; never a successful graph route."""
    def __init__(self,state,qualification,feedback):
        self.state=state;self.qualification=qualification;self.feedback=feedback
        super().__init__('Actual finite-face close qualification requires open-hand recenter')


def geometry():
    global _GEOMETRY
    if _GEOMETRY is None:
        _GEOMETRY = PublicBimanualGeometry(device='cuda:0', minimum_clearance_m=FLOOR)
    return _GEOMETRY


def _failure_summary(guard):
    index=int(guard.get('failed_sample',max(0,len(guard.get('rows',[]))-1)))
    rows=guard.get('rows',[])
    return dict(safe=guard.get('safe',False),failed_sample=guard.get('failed_sample'),
        samples_checked=guard.get('samples_checked'),failed_groups=guard.get('failed_groups',[]),
        witness_row=rows[min(index,len(rows)-1)] if rows else None,
        board_persistence_guard=guard.get('caller_board_persistence_guard'))


def _guard_passes(guard,scene):
    if not guard.get('safe',guard.get('feasible',False)):return False
    margin=float(scene.get('board_for_planning',{}).get('uncertainty_m',0.))
    if not np.isfinite(margin) or margin<0.:
        raise StageRefusal('Board persistence bound must be finite and nonnegative')
    threshold=FLOOR+margin
    rows=guard.get('rows',[])
    if not rows:raise StageRefusal('Full scene guard did not provide per-sample evidence')
    for index,row in enumerate(rows):
        groups=row.get('groups',{})
        for name in ('robot_board79','held_envelope_board79'):
            if name not in groups:raise StageRefusal('Full scene board guard component missing: '+name)
            if groups[name]['clearance_m']<threshold:
                guard['caller_board_persistence_guard']=dict(passed=False,sample=index,group=name,
                    required_clearance_m=threshold,board_uncertainty_m=margin,actual=groups[name])
                return False
    guard['caller_board_persistence_guard']=dict(passed=True,required_clearance_m=threshold,board_uncertainty_m=margin)
    return True


def _roles(state):
    sender = int(state.get('handoff', {}).get('sender', state['arm']))
    receiver = int(state.get('handoff', {}).get('receiver', 1-sender))
    if state.get('piece') != 'red' or {sender, receiver} != {0, 1}:
        raise StageRefusal('Red handoff requires two distinct declared public arms')
    return sender, receiver


def _held_envelope(belief):
    bound=float(belief.get('uncertainty_m',.006))
    if not np.isfinite(bound) or bound<0.:
        raise StageRefusal('Held attachment uncertainty must be a finite nonnegative declared bound')
    return max(.006,bound)


def _body(actual, belief):
    hand = Rotation.from_quat(actual['tcp'][3:]).as_matrix()
    return np.r_[np.asarray(actual['tcp'][:3])+hand@np.asarray(belief['object_in_tcp_position']),
                 Rotation.from_matrix(hand@np.asarray(belief['object_in_tcp_rotation'])).as_quat()]


def _waypoint(position, rotation):
    return dict(position=np.asarray(position).tolist(), rotation=c.qdict(Rotation.from_matrix(rotation).as_quat()))


def _sender_waypoint(pose, belief):
    rr = Rotation.from_quat(pose[3:]).as_matrix()@np.asarray(belief['object_in_tcp_rotation']).T
    return _waypoint(pose[:3]-rr@np.asarray(belief['object_in_tcp_position']), rr)


def _source(observed, role):
    epoch = encoder_epoch(observed)
    if 'red' not in observed.get('poses', {}):
        raise StageRefusal('An actual accepted Red RGB pose is required for this attachment')
    return dict(mode='pyramid', piece='red', pose=deepcopy(observed['poses']['red']),
        timestamp=observed['timestamp'], frame_sha256=deepcopy(observed['frame_sha256']),
        observation_source=observed['observation_source'], robot_encoder_values=epoch['values'],
        robot_encoder_sha256=epoch['sha256'], source_step=epoch['step'], source_role=role)


def _belief(ctx, state, arm):
    p = state['parameters']
    if p.get('conditional_attachment_id'):
        return ctx.tool('puzzle.propagate_conditional_attachment', attachment_id=p['conditional_attachment_id'], arm_id=arm)
    return ctx.tool('puzzle.propagate_pyramid_held_belief', source=p['owned_held_belief_source'], arm_id=arm)


def _fresh_sender(ctx, state, arm, *, optimizer_belief=None):
    expired=False
    try:old=_belief(ctx,state,arm)
    except Exception as error:
        text=str(error)
        if optimizer_belief is None or not ('bounded30s' in text or 'expired' in text.lower()):raise
        # An expired attachment may seed a new image fit, never authorize
        # control. A new authentic RGB/q16 attachment is mandatory below.
        old=optimizer_belief;expired=True
    actual_hint=base.proprio(ctx,arm)
    hint=_body(actual_hint,old).tolist()
    hints={'red':hint}
    # Historical board RGB supplies only an optimizer proposal for current
    # edge fitting.  It is never copied into the current observation or dated
    # as new evidence. This also makes the multiview board reacquirer available
    # while a still-valid dated board exists, rather than only after expiry.
    now=float(actual_hint.get('timestamp',state.get('observation',{}).get('timestamp',-1.)))
    boards=[row for row in state.get('rgb_history',[])if 'board'in row.get('poses',{})
        and 0.<=now-float(row['timestamp'])<=30.]
    if boards:hints['board']=deepcopy(max(boards,key=lambda row:row['timestamp'])['poses']['board'])
    observed = c.state(ctx,hints)
    state = save_observation(state, observed)
    if 'red' not in observed.get('poses', {}):
        if expired:raise StageRefusal('Expired sender control attachment requires new authentic Red RGB/q16')
        return state, old, dict(status='RGB_UNKNOWN', measured=False)
    source = _source(observed, 'Actual Red RGB and same-image q16 before bimanual transfer')
    actual = base.proprio(ctx, arm)
    innovation = visual_memory.shape_displacement('red', hint, source['pose'])
    gap=c.part_hand_gap('red',np.asarray(source['pose']),np.asarray(actual['tcp'][:3]))
    if innovation>.020 or gap>.045:
        final_receiver=(arm==1 and state.get('handoff',{}).get('hop')==2 and
            state.get('handoff',{}).get('stage')in('transferred','verified'))
        if expired or not final_receiver:
            raise StageRefusal('Current RGB contradicts retained sender attachment')
        state,old,retained=_retained_receiver_scene_check(ctx,state,arm)
        rejected=dict(status='RGB_CONFLICT_OLD_BOUND_RETAINED',current_target_measured=False,
            rejected_current_RGB_source=source,old_prediction_vs_RGB_bead_innovation_m=innovation,
            rejected_RGB_hand_gap_m=float(gap),retained_control=retained,
            rejected_candidate_does_not_replace_owned_source=True)
        return state,old,rejected
    p = {**state['parameters'], 'owned_held_belief_source': source}
    p.pop('conditional_attachment_id', None)
    state = {**state, 'parameters': p}
    bound = _belief(ctx, state, arm)
    return state, bound, dict(status='actual_RGB_q16_rebound', source=source, innovation_m=innovation)


def _retained_receiver_scene_check(ctx,state,receiver):
    """Keep a still-valid dated attachment when a new visual fit conflicts.

    This checks the existing registered epoch/TTL and jaw contradiction rule,
    then current robot and full public scene using the unchanged bound.  It
    neither measures the object nor proves rigid persistence from geometry.
    """
    belief=_belief(ctx,state,receiver)
    state,scene,refresh=scene_for_motion(ctx,state,belief)
    belief=_belief(ctx,state,receiver)  # Recheck age/jaw after observation time.
    actual=base.proprio(ctx,receiver);h=state['handoff']
    selected=h['contract'].get(receiver,h['contract'].get(str(receiver)))
    if not selected:raise StageRefusal('Retained receiver requires its original named public contact')
    guard=geometry().screen_q16_path([actual['robot_encoder_values']],scene=scene,moving_arm=1-receiver,
        held_arm=receiver,held_body_pose=_body(actual,belief).tolist(),
        object_in_tcp_position=belief['object_in_tcp_position'],object_in_tcp_rotation=belief['object_in_tcp_rotation'],
        target='red',finger_target_contract={receiver:selected},board_anchor_exit=None,held_envelope_m=_held_envelope(belief))
    if not _guard_passes(guard,scene):
        raise StageRefusal('Retained receiver proposal failed unchanged current full-scene geometry '+json.dumps(_failure_summary(guard)))
    return state,belief,dict(current_target_measured=False,visual_status='RGB_CONFLICT_OR_UNKNOWN',
        existing_source=deepcopy(belief.get('source',{})),source_age_s=belief.get('source_age_s'),
        held_envelope_m=_held_envelope(belief),actual_robot=actual,full_scene=guard,scene_refresh=refresh,
        attachment_is_still_conditional=True,source_or_bound_widened=False)


def _expand(q16, path, arm):
    q = np.asarray(q16, float)
    a = np.asarray(path, float)
    if q.shape != (16,) or a.ndim != 2 or a.shape[1] != 6 or len(a) < 2 or not np.isfinite(a).all():
        raise StageRefusal('Native handoff planner must return finite six-joint samples')
    off = 8*arm
    # Native path[0] is the solver's later actual planning epoch. The other
    # arm remains the explicitly actual/proposed public q16 supplied by caller.
    # This mathematical splice is not relabelled as a sensor observation.
    q=q.copy();q[off:off+6]=a[0]
    out = [q.copy()]
    for first, last in zip(a[:-1], a[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(last-first))/.012)))
        for u in np.linspace(0., 1., count+1)[1:]:
            value = q.copy(); value[off:off+6] = first+(last-first)*u; out.append(value)
    return np.asarray(out)


def _screen(ctx, state, scene, belief, held_arm, moving_arm, waypoints, contract, *, q16=None):
    actual = base.proprio(ctx, moving_arm)
    seed = np.asarray(actual['robot_encoder_values'] if q16 is None else q16, float)
    started=time.perf_counter()
    path = ctx.tool('puzzle.plan_pose_sequence', waypoints=waypoints, arm_id=moving_arm,
                    min_joint_limit_margin_rad=RESERVE)
    plan_wall=time.perf_counter()-started
    print(json.dumps(dict(event='red_handoff_plan',hop=state.get('handoff',{}).get('hop',1),arm=moving_arm,
        feasible=bool(path.get('feasible')),reserve_rad=path.get('min_joint_limit_margin_rad'),
        wall_s=round(plan_wall,4),samples=len(path.get('path',[])))),flush=True)
    if not path.get('feasible') or path.get('min_joint_limit_margin_rad', -1.) < RESERVE:
        raise c.PlanningRefusal('Handoff native reserved path refused '+json.dumps((path.get('errors') or [{}])[-1]))
    qpath = _expand(seed, path['path'], moving_arm)
    # Derive the first held pose from exactly this proposed q16's public FK,
    # including an offline future sender endpoint used for receiver preflight.
    g=geometry();g.fk(qpath[0]);ik=g.robot.iks['left' if held_arm==0 else 'right']
    from .vision_planning import finite_fk
    pp,qq=finite_fk(ik,qpath[0])
    held_pose=_body(dict(tcp=[*np.asarray(pp).tolist(),*np.asarray(qq).tolist()]),belief)
    guard_started=time.perf_counter()
    guard = geometry().screen_q16_path(qpath, scene=scene, moving_arm=moving_arm,
        held_arm=held_arm, held_body_pose=held_pose.tolist(),
        object_in_tcp_position=belief['object_in_tcp_position'],
        object_in_tcp_rotation=belief['object_in_tcp_rotation'], target='red',
        finger_target_contract=contract, board_anchor_exit=None,held_envelope_m=_held_envelope(belief))
    geometry_wall=time.perf_counter()-guard_started
    passed=_guard_passes(guard,scene)
    print(json.dumps(dict(event='red_handoff_geometry',hop=state.get('handoff',{}).get('hop',1),arm=moving_arm,
        safe=passed,wall_s=round(geometry_wall,4),samples=guard.get('samples_checked'),
        failed_groups=guard.get('failed_groups',[]),board_margin=guard.get('caller_board_persistence_guard'))),flush=True)
    if not passed:
        raise c.PlanningRefusal('Handoff full public robot/scene guard refused '+json.dumps(_failure_summary(guard), separators=(',', ':')))
    return dict(arm=moving_arm, native_path=path, q16_path=qpath.tolist(), guard=guard,
                endpoint=waypoints[-1], source_actual=actual,timing=dict(native_plan_wall_s=plan_wall,geometry_wall_s=geometry_wall))


def _execute(ctx, state, motion, *, retained_arm, stable_actual_handoff=None):
    moving = motion['arm']; qpath = np.asarray(motion['q16_path']); off = moving*8
    current = base.proprio(ctx, moving)
    if np.max(np.abs(np.asarray(current['robot_encoder_values'])-qpath[0])) > .002:
        raise StageRefusal('Actual q16 changed since full handoff path screening')
    start_width = base.proprio(ctx, retained_arm)['width_m']
    runs = []
    for begin in range(1, len(qpath), 8):
        grip = base.proprio(ctx, retained_arm)
        if grip['width_m'] < max(.008, start_width-.0025):
            raise StageRefusal('Actual retained hand jaw contradicts handoff attachment')
        commands = [dict(positions=row[off:off+6].tolist()) for row in qpath[begin:begin+8]]
        moved = ctx.tool('robot.execute_trajectory', trajectory=dict(waypoints=commands), arm_id=moving,
                         max_steps_per_waypoint=3)
        if moved.get('status') != 'ok':
            raise StageRefusal('Native handoff motor execution failed')
        runs.append(moved)
    endpoint = motion['endpoint']
    # Wait/read only: no hidden unscreened corrective motor trajectory.
    adaptation=None
    try:
        c.await_motion(ctx,np.asarray(endpoint['position']),endpoint['rotation'],moving,False,.0025)
    except c.MotorAccuracyError as error:
        if stable_actual_handoff is None or moving!=retained_arm or error.pe>.006 or error.re>.03:
            raise
        # Only an explicitly declared intermediate held-hand endpoint can
        # adapt. The original await already requires low drift; final hover,
        # receiver approach and terminal descent keep their original precision.
        # No motor correction is issued: evaluate the actual stable endpoint.
        scene=stable_actual_handoff['scene'];belief=stable_actual_handoff['belief']
        held_arm=int(stable_actual_handoff['held_arm'])
        actual=base.proprio(ctx,moving)
        if actual['width_m']<max(.008,start_width-.0025):
            raise StageRefusal('Actual sender endpoint contradicts retained jaw')
        held_actual=base.proprio(ctx,held_arm)
        guard=geometry().screen_q16_path([actual['robot_encoder_values']],scene=scene,
            moving_arm=moving,held_arm=held_arm,held_body_pose=_body(held_actual,belief).tolist(),
            object_in_tcp_position=belief['object_in_tcp_position'],
            object_in_tcp_rotation=belief['object_in_tcp_rotation'],target='red',
            finger_target_contract=stable_actual_handoff['contract'],board_anchor_exit=None,
            held_envelope_m=_held_envelope(belief))
        if not _guard_passes(guard,scene):
            raise StageRefusal('Stable sender actual handoff endpoint failed full geometry '+json.dumps(_failure_summary(guard)))
        ctx.tool('robot.wait_steps',steps=6)
        stable=base.proprio(ctx,moving)
        delta=float(np.linalg.norm(np.asarray(stable['tcp'][:3])-np.asarray(actual['tcp'][:3])))
        angular=float((Rotation.from_quat(actual['tcp'][3:]).inv()*Rotation.from_quat(stable['tcp'][3:])).magnitude())
        if delta>.0004 or angular>.015 or stable['width_m']<max(.008,start_width-.0025):
            raise StageRefusal('Accepted actual handoff endpoint did not remain stable and retained')
        second=geometry().screen_q16_path([stable['robot_encoder_values']],scene=scene,
            moving_arm=moving,held_arm=held_arm,held_body_pose=_body(stable,belief).tolist(),
            object_in_tcp_position=belief['object_in_tcp_position'],
            object_in_tcp_rotation=belief['object_in_tcp_rotation'],target='red',
            finger_target_contract=stable_actual_handoff['contract'],board_anchor_exit=None,
            held_envelope_m=_held_envelope(belief))
        if not _guard_passes(second,scene):
            raise StageRefusal('Stable sender endpoint second actual geometry failed '+json.dumps(_failure_summary(second)))
        purpose=stable_actual_handoff.get('purpose','intermediate_handoff_sender')
        if purpose not in('intermediate_handoff_sender','intermediate_goal_carry'):
            raise StageRefusal('Stable actual adaptation is restricted to reviewed intermediate phases')
        if purpose=='intermediate_goal_carry' and not(moving==1 and state.get('handoff',{}).get('stage')=='verified'):
            raise StageRefusal('Stable goal intermediate requires the verified right receiver')
        adaptation=dict(contract='Stable_actual_intermediate_endpoint_v2',purpose=purpose,
            intended_TCP=endpoint,actual_robot=stable,position_error_m=float(error.pe),
            actual_initial_robot=actual,
            rotation_error_rad=float(error.re),six_step_TCP_drift_m=delta,
            six_step_angular_drift_rad=angular,actual_geometry=guard,stable_geometry=second,
            accepted_only_for_intermediate_handoff_sender=purpose=='intermediate_handoff_sender',
            accepted_only_for_intermediate_goal_carry=purpose=='intermediate_goal_carry',
            receiver_path_must_recompute_from_actual_RGB_q16_body=True,
            goal_or_receiver_precision_relaxed=False,motor_correction_issued=False)
        print(json.dumps(dict(event='red_handoff_stable_actual_sender' if purpose=='intermediate_handoff_sender' else 'red_goal_stable_actual_intermediate',position_error_m=error.pe,
            rotation_error_rad=error.re,drift_m=delta,full_geometry_safe=True)),flush=True)
    return dict(screen=motion,motor_runs=runs,actual_after=base.proprio(ctx,moving),stable_actual_endpoint=adaptation)


def _receiver_candidates(state):
    # Public body-relative geometry only. The data leaf deliberately contains
    # no world poses, cached IK solutions or trajectories.
    from .red_handoff_candidates import CANDIDATES
    candidates = state['parameters'].get('red_handoff_candidates', CANDIDATES)
    if not isinstance(candidates, list) or not candidates:
        raise StageRefusal('No declared public CAD handoff candidate family')
    return candidates


def _candidate_body_poses(state,board,current):
    br=Rotation.from_quat(board[3:]).as_matrix()
    selected=state.get('handoff',{}).get('active_candidate',{})
    hop=int(state.get('handoff',{}).get('hop',1))
    pose_recipe=selected.get('handoff_pose_proposal')
    if pose_recipe is not None:
        offset=np.asarray(pose_recipe['body_center_board_relative_m'],float)
        rotation=np.asarray(pose_recipe['body_rotation_board_relative'],float)
        if (offset.shape!=(3,) or rotation.shape!=(3,3) or not np.isfinite(offset).all() or
                not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7)):
            raise StageRefusal('Public relative handoff pose recipe is invalid')
        yield np.r_[board[:3]+br@offset,Rotation.from_matrix(br@rotation).as_quat()]
    yield current.copy()
    rotations=[Rotation.from_quat(current[3:]).as_matrix()]
    if int(state.get('handoff',{}).get('hop',1))==2:
        from .vision_observer import goals
        _,public_R=goals('pyramid',state['parameters']['pyramid_plan'])['red']
        rotations.insert(0,br@public_R)
    offsets=state['parameters'].get('red_handoff_common_offsets',
        [[-.055,-.035,.19],[-.055,0.,.15],[-.055,0.,.19],[-.085,0.,.19],
         [-.025,0.,.19],[-.055,.035,.19],[-.055,0.,.23]])
    for rotation in rotations:
        for offset in offsets:
            pose=np.r_[board[:3]+br@np.asarray(offset,float),Rotation.from_matrix(rotation).as_quat()]
            yield pose


def scene_for_motion(ctx,state,belief):
    """Reacquire an actual board before planning another motion.

    A dated board can seed calibrated RGB fitting only. It is never copied as
    a fresh pose, its timestamp is never changed, and no motor call occurs.
    """
    attempts=[]
    try:
        return state,scene_belief(state,state['observation']),dict(status='bounded_scene_already_available')
    except StageRefusal as error:
        if 'No bounded RGB source for board' not in str(error):raise
    board_rows=[row for row in state.get('rgb_history',[]) if 'board' in row.get('poses',{})]
    old_board=max(board_rows,key=lambda row:row['timestamp']) if board_rows else None
    for attempt in range(3):
        actual=base.proprio(ctx,int(state['arm']))
        hints={'red':_body(actual,belief).tolist(),'__observer_mode__':'plain_rgb_cad'}
        if old_board is not None:hints['board']=old_board['poses']['board']
        observed=c.state(ctx,hints);state=save_observation(state,observed)
        attempts.append(dict(timestamp=observed['timestamp'],visible_names=sorted(observed.get('poses',{})),
            board_hint_role='Dated authentic optimizer proposal only; no current pose fill'))
        if 'board' in observed.get('poses',{}):
            scene=scene_belief(state,observed)
            state['scene_refresh_required_before_next_motion']=False
            return state,scene,dict(status='new_actual_board_RGB',attempts=attempts)
    raise StageRefusal('Fresh board RGB unavailable before new motion after retained-hand observation repair '+json.dumps(attempts))


def _rebind_sender_observation(ctx,state,observed,sender,old_belief,role):
    """An accepted current RGB/q16 packet supersedes a rigid prediction.

    Preserve the innovation explicitly; the fresh measurement must itself
    satisfy actual gripper/CAD geometry, rather than inheriting an old source.
    """
    source=_source(observed,role)
    actual=base.proprio(ctx,sender)
    predicted=_body(actual,old_belief)
    innovation=visual_memory.shape_displacement('red',predicted,source['pose'])
    if innovation>.020 or c.part_hand_gap('red',np.asarray(source['pose']),np.asarray(actual['tcp'][:3]))>.045:
        raise StageRefusal('Fresh dual-hand RGB contradicts retained sender geometry beyond bounded recovery')
    parameters={**state['parameters'],'owned_held_belief_source':source}
    parameters.pop('conditional_attachment_id',None)
    state=save_observation(state,observed);state['parameters']=parameters
    belief=_belief(ctx,state,sender)
    return state,belief,dict(source=source,old_prediction_vs_actual_RGB_bead_innovation_m=innovation,
        measured_source_supersedes_old_rigid_attachment=True)


def _recenter_preclose(ctx,state,sender,receiver,belief,scene,candidate,contract,sender_arrival):
    """Reobserve and recenter the open receiver before requesting close.

    Small source slip/rotation during the receiver approach is measured from
    real RGB. CAD-relative grasp recipes remain proposals; current q16/FK and
    complete geometry decide each correction. Sender never opens here.
    """
    records=[];final_right=(receiver==1 and state.get('handoff',{}).get('hop')==2)
    # Three screened moves and a fourth genuine image to assess the last one.
    # A stable servo residual is feedback for another proposal, never success.
    observations=4 if final_right else 3
    for attempt in range(observations):
        predicted=_body(base.proprio(ctx,sender),belief)
        observed=c.state(ctx,{'red':predicted.tolist()});state=save_observation(state,observed)
        if 'red' not in observed.get('poses',{}):
            if attempt==observations-1:break
            ctx.tool('robot.wait_steps',steps=6);continue
        if observed['timestamp']<sender_arrival:
            raise StageRefusal('Preclose image predates sender movement')
        state,belief,rebound=_rebind_sender_observation(ctx,state,observed,sender,belief,
            'Fresh actual preclose Red RGB plus same-image q16 after receiver approach')
        state,scene,scene_refresh=scene_for_motion(ctx,state,belief)
        actual=base.proprio(ctx,receiver)
        if actual['width_m']<.045:raise StageRefusal('Receiver must remain open while recentering on fresh RGB')
        pose=_body(base.proprio(ctx,sender),belief);rr=Rotation.from_quat(pose[3:]).as_matrix()
        handR=rr@np.asarray(candidate['hand_in_body_rotation']);point=pose[:3]+rr@np.asarray(candidate['hand_in_body_position'])
        pe=float(np.linalg.norm(np.asarray(actual['tcp'][:3])-point))
        re=float((Rotation.from_quat(actual['tcp'][3:]).inv()*Rotation.from_matrix(handR)).magnitude())
        record=dict(actual_RGB_rebinding=rebound,scene_refresh=scene_refresh,
            actual_receiver_before=actual,measured_TCP_recenter_error_m=pe,measured_rotation_error_rad=re)
        records.append(record)
        face=_receiver_face_measurement(observed,receiver,candidate) if final_right else None
        record['same_image_finite_face_qualification']=face
        aligned=(pe<=.001 and re<=.025 and face['passed']) if final_right else (pe<=.0025 and re<=.04)
        if aligned:
            return state,belief,scene,_source(observed,'Actual postmotion preclose RGB after open receiver recenter'),records
        if attempt==observations-1:break
        command_point=point.copy();command_rotation=handR.copy()
        if final_right and attempt>0:
            # The true CAD target remains unchanged. Compensate an observed
            # stable motor residual only through a new fully screened path.
            shift=point-np.asarray(actual['tcp'][:3]);length=float(np.linalg.norm(shift))
            if length>.006:shift*=.006/length
            residual=Rotation.from_matrix(handR@Rotation.from_quat(actual['tcp'][3:]).as_matrix().T).as_rotvec()
            angle=float(np.linalg.norm(residual))
            if angle>.06:residual*=.06/angle
            command_point=point+shift
            command_rotation=Rotation.from_rotvec(residual).as_matrix()@handR
            record['explicit_bounded_servo_compensation']=dict(true_target=_waypoint(point,handR),
                translation_m=shift.tolist(),rotation_vector_rad=residual.tolist(),
                maximum_translation_m=.006,maximum_rotation_rad=.06)
        motion=_screen(ctx,state,scene,belief,sender,receiver,[_waypoint(command_point,command_rotation)],contract)
        donor_before=base.proprio(ctx,sender)
        try:
            record['open_receiver_actual_RGB_recenter']=_execute(ctx,state,motion,retained_arm=sender)
        except c.MotorAccuracyError as error:
            if not final_right or error.pe>.006 or error.re>.03:raise
            record['stable_residual_requires_new_RGB_not_success']=_receiver_residual_reobserve(
                ctx,state,sender,receiver,belief,scene,contract,donor_before,error)
    # At least an authentic post-movement source is mandatory. UNKNOWN is not
    # overwritten with the grasp recipe or a prediction at the close epoch.
    raise StageRefusal('Fresh RGB open-receiver recenter did not converge within bounded attempts '+json.dumps(records))


def _receiver_residual_reobserve(ctx,state,sender,receiver,belief,scene,contract,donor_before,error):
    """Qualify safe observation only after an inaccurate open-hand move."""
    snapshots=[];checks=[];reserves=[]
    for index in range(2):
        actual=base.proprio(ctx,receiver);donor=base.proprio(ctx,sender)
        if actual['width_m']<.045 or donor['width_m']<max(.008,donor_before['width_m']-.0025):
            raise StageRefusal('Open receiver residual contradicted actual jaw retention')
        g=geometry();ik=g.robot.iks['left'if receiver==0 else'right']
        q=np.asarray(actual['robot_encoder_values'],float)
        lower=np.asarray(ik.model.joint_limit_lower.numpy())[ik.arm_dofs]
        upper=np.asarray(ik.model.joint_limit_upper.numpy())[ik.arm_dofs]
        values=q[ik.arm_coords]
        if (not np.isfinite(np.r_[values,lower,upper]).all() or np.any(lower>=upper)):
            raise StageRefusal('Open receiver residual requires finite public joint limits and actual encoders')
        reserve=float(np.minimum(values-lower,upper-values).min())
        if reserve<RESERVE:
            raise StageRefusal('Open receiver residual actual joint reserve is below original 0.035rad')
        reserves.append(reserve)
        guard=g.screen_q16_path([actual['robot_encoder_values']],scene=scene,
            moving_arm=receiver,held_arm=sender,held_body_pose=_body(donor,belief).tolist(),
            object_in_tcp_position=belief['object_in_tcp_position'],object_in_tcp_rotation=belief['object_in_tcp_rotation'],
            target='red',finger_target_contract=contract,board_anchor_exit=None,held_envelope_m=_held_envelope(belief))
        if not _guard_passes(guard,scene):
            raise StageRefusal('Open receiver residual actual geometry refused '+json.dumps(_failure_summary(guard)))
        snapshots.append(actual);checks.append(guard)
        if index==0:ctx.tool('robot.wait_steps',steps=6)
    first,last=snapshots
    drift=float(np.linalg.norm(np.asarray(last['tcp'][:3])-np.asarray(first['tcp'][:3])))
    angular=float((Rotation.from_quat(first['tcp'][3:]).inv()*Rotation.from_quat(last['tcp'][3:])).magnitude())
    if drift>.0004 or angular>.015:
        raise StageRefusal('Open receiver residual was not stable enough for fresh feedback')
    return dict(position_error_m=float(error.pe),rotation_error_rad=float(error.re),
        six_step_drift_m=drift,six_step_rotation_rad=angular,actual_samples=snapshots,full_geometry=checks,
        actual_receiver_reserves_rad=reserves,required_receiver_reserve_rad=RESERVE,
        receiver_qualified=False,dual_hold_earned=False,requires_fresh_RGB_and_tight_true_target=True)


def handoff_prepare(ctx, state):
    state=deepcopy(state);sender,receiver=_roles(state)
    inherited=state.get('handoff',{})
    source_bead=int(state['parameters'].get('red_handoff_sender_bead',inherited.get('source_selected_bead',4)))
    if source_bead not in range(len(model.local_beads('red'))):
        raise StageRefusal('Sender contact must name a public Red bead')
    hop=int(inherited.get('hop',1))
    if hop not in (1,2):raise StageRefusal('Red two-hop strategy requires hop 1 or 2')
    state['handoff']=dict(sender=sender,receiver=receiver,stage='search',source_selected_bead=source_bead,hop=hop)
    state,belief,refresh=_fresh_sender(ctx,state,sender)
    state,scene,scene_refresh=scene_for_motion(ctx,state,belief);board=np.asarray(scene['board_for_planning']['pose'])
    actual=base.proprio(ctx,sender);body=_body(actual,belief)
    # Receiver jaw sweep is checked against the actual source and full public
    # scene before issuing an opening command.
    other=base.proprio(ctx,receiver);sweep=[]
    for width in np.linspace(other['width_m'],OPEN,12):
        qq=np.asarray(other['robot_encoder_values']);qq[8*receiver+6]=width/2.;qq[8*receiver+7]=-width/2.
        guard=geometry().screen_q16_path([qq.tolist()],scene=scene,moving_arm=receiver,held_arm=sender,
            held_body_pose=body.tolist(),object_in_tcp_position=belief['object_in_tcp_position'],
            object_in_tcp_rotation=belief['object_in_tcp_rotation'],target='red',
            finger_target_contract={sender:[source_bead]},board_anchor_exit=None,held_envelope_m=_held_envelope(belief))
        if not _guard_passes(guard,scene):
            raise c.PlanningRefusal('Receiver opening public geometry refused '+json.dumps(_failure_summary(guard)))
        sweep.append(guard)
    opening=ctx.tool('robot.set_grip',arm_id=receiver,width_m=OPEN,ramp_steps=12,settle_steps=8)
    attempts=[];chosen=None
    for candidate in _receiver_candidates(state):
        if int(candidate.get('arm',receiver))!=receiver:continue
        state['handoff']['active_candidate']=candidate
        ph=np.asarray(candidate['hand_in_body_position']);rh=np.asarray(candidate['hand_in_body_rotation'])
        bead=int(candidate['bead']);contract={sender:[source_bead],receiver:[bead]}
        for proposal in _candidate_body_poses(state,board,body):
            try:
                sp=_sender_waypoint(proposal,belief)
                sender_motion=_screen(ctx,state,scene,belief,sender,sender,[sp],{sender:[source_bead]})
                future=np.asarray(sender_motion['q16_path'][-1]);rr=Rotation.from_quat(proposal[3:]).as_matrix()
                rotation=rr@rh;point=proposal[:3]+rr@ph
                approach=_waypoint(point-rotation[:,2]*.030,rotation);touch=_waypoint(point,rotation)
                receiver_motion=_screen(ctx,state,scene,belief,sender,receiver,[approach,touch],contract,q16=future)
                chosen=dict(candidate=deepcopy(candidate),body_pose_proposal=proposal.tolist(),
                    sender_motion=sender_motion,receiver_motion_preflight=receiver_motion,contract=contract)
                break
            except c.PlanningRefusal as error:
                attempts.append(dict(bead=bead,body_pose_proposal=proposal.tolist(),refusal=str(error)))
        if chosen is not None:break
    if chosen is None:raise c.PlanningRefusal('No full-path bimanual candidate '+json.dumps(attempts,separators=(',',':')))
    # Candidate search consumes real native time. Reacquire/authenticate the
    # sender source and recompute both arm paths before the first motor move.
    state,belief,preexecute_refresh=_fresh_sender(ctx,state,sender,optimizer_belief=belief)
    scene=scene_belief(state,state['observation'])
    proposal=np.asarray(chosen['body_pose_proposal'])
    sender_motion=_screen(ctx,state,scene,belief,sender,sender,[_sender_waypoint(proposal,belief)],{sender:[source_bead]})
    rr=Rotation.from_quat(proposal[3:]).as_matrix();candidate=chosen['candidate']
    receiver_R=rr@np.asarray(candidate['hand_in_body_rotation']);receiver_p=proposal[:3]+rr@np.asarray(candidate['hand_in_body_position'])
    receiver_way=[_waypoint(receiver_p-receiver_R[:,2]*.030,receiver_R),_waypoint(receiver_p,receiver_R)]
    chosen['sender_motion']=sender_motion
    chosen['receiver_motion_preflight']=_screen(ctx,state,scene,belief,sender,receiver,receiver_way,chosen['contract'],q16=sender_motion['q16_path'][-1])
    chosen['preexecute_source_refresh']=preexecute_refresh
    sender_execution=_execute(ctx,state,chosen['sender_motion'],retained_arm=sender,
        stable_actual_handoff=dict(scene=scene,belief=belief,held_arm=sender,contract={sender:[source_bead]}))
    sender_arrival=float(sender_execution['actual_after']['timestamp'])
    state,belief,post_sender_refresh=_fresh_sender(ctx,state,sender,optimizer_belief=belief)
    if (sender_execution.get('stable_actual_endpoint') is not None and
            post_sender_refresh.get('status')!='actual_RGB_q16_rebound'):
        raise StageRefusal('Adapted actual sender endpoint requires fresh real Red RGB/q16 attachment before receiver motion')
    scene=scene_belief(state,state['observation']);actual=base.proprio(ctx,sender);body=_body(actual,belief)
    candidate=chosen['candidate'];rr=Rotation.from_quat(body[3:]).as_matrix();rotation=rr@np.asarray(candidate['hand_in_body_rotation'])
    point=body[:3]+rr@np.asarray(candidate['hand_in_body_position'])
    waypoints=[_waypoint(point-rotation[:,2]*.030,rotation),_waypoint(point,rotation)]
    receiver_motion=_screen(ctx,state,scene,belief,sender,receiver,waypoints,chosen['contract'])
    receiver_execution=_execute(ctx,state,receiver_motion,retained_arm=sender)
    # Receiver approach can make the target slip/rotate in the sender. A
    # new actual RGB/q16 body rebinds the sender and recenters the open receiver.
    state,belief,scene,preclose,preclose_recenter=_recenter_preclose(ctx,state,sender,receiver,
        belief,scene,candidate,chosen['contract'],sender_arrival)
    ctx.tool('robot.wait_steps',steps=6)
    state['handoff'].update(stage='prepared',candidate=chosen['candidate'],contract=chosen['contract'],
        sender_belief=belief,sender_arrival_timestamp=sender_arrival,preclose_rgb=preclose,body_before_close=_body(base.proprio(ctx,sender),belief).tolist(),
        record=dict(source_refresh=refresh,scene_refresh=scene_refresh,receiver_opening=opening,opening_guards=sweep,
            candidate_refusals=attempts,chosen=chosen,sender_execution=sender_execution,
            post_sender_refresh=post_sender_refresh,receiver_execution=receiver_execution,preclose_recenter=preclose_recenter))
    return result('prepared',state,state['observation'],'handoff_prepare',handoff=state['handoff'])


def warm_left_prepare(ctx,state):
    """Prepare the second transfer from a fresh conditional left-held fixture.

    The public left grasp supplies an optimizer hint only.  Current accepted
    RGB with synchronous q16 supplies the attachment; no fixture history earns
    an earlier handoff.  The normal dual-hold and receiver-only checkpoints
    remain mandatory after this preparation.
    """
    state=deepcopy(state)
    parameters=state['parameters']
    if state.get('piece')!='red' or not parameters.get('red_bimanual_warm_debug',False):
        raise StageRefusal('Warm left preparation requires an explicit fresh Red conditional debug state')
    if (any(parameters.get(key) for key in
            ('owned_held_belief_source','conditional_attachment_id','conditional_attachment','actual_executed_prefix')) or
            any(state.get(key) for key in
            ('handoff','conditional_attachment','execution_tokens','executed_placements',
             'released_placements','pending_placement_receipt','release_execution'))):
        raise StageRefusal('Warm left preparation refuses old sources, handoffs, tokens or earned execution')

    from .red_handoff_candidates import CANDIDATES
    left_recipe=CANDIDATES[0]
    candidate=deepcopy(left_recipe['final_right_public_grasp']);candidate['arm']=1
    if int(left_recipe['bead'])!=1 or int(candidate['bead'])!=4:
        raise StageRefusal('Warm left fixture requires declared left bead1 and right bead4 contacts')
    sender,receiver=0,1
    contract={0:[1],1:[4]}
    left=base.proprio(ctx,sender);right=base.proprio(ctx,receiver)
    if not (.010<float(left['width_m'])<.045 and abs(float(right['width_m'])-OPEN)<=.001):
        raise StageRefusal('Warm left fixture requires actual loaded left and open55mm right jaws')
    state.update(arm=sender,piece='red')
    state['handoff']=dict(sender=sender,receiver=receiver,source_selected_bead=1,hop=2,
        contract=contract,candidate=candidate,stage='conditional_debug_left_fixture',
        previous_handoffs_earned=False,
        source_scope='Fresh current RGB/q16 only; left motor-history fixture earns no prior handoff')

    # CAD-relative fixture geometry proposes an image fit; never bind this hint.
    handR=Rotation.from_quat(left['tcp'][3:]).as_matrix()
    bodyR=handR@np.asarray(left_recipe['hand_in_body_rotation']).T
    hint=np.r_[np.asarray(left['tcp'][:3])-bodyR@np.asarray(left_recipe['hand_in_body_position']),
               Rotation.from_matrix(bodyR).as_quat()]
    observed=c.state(ctx,{'red':hint.tolist()});state=save_observation(state,observed)
    source=_source(observed,'New actual warm-left Red RGB and same-image left q16; no prior epoch')
    actual=base.proprio(ctx,sender)
    if c.part_hand_gap('red',np.asarray(source['pose']),np.asarray(actual['tcp'][:3]))>.045:
        raise StageRefusal('Warm left actual Red RGB is not near the loaded sender')
    parameters={**state['parameters'],'owned_held_belief_source':source}
    parameters.pop('conditional_attachment_id',None)
    state['parameters']=parameters
    belief=_belief(ctx,state,sender)
    state,scene,scene_refresh=scene_for_motion(ctx,state,belief)
    # Scene observation may take time; revalidate the dated left attachment.
    belief=_belief(ctx,state,sender)
    actual=base.proprio(ctx,sender);other=base.proprio(ctx,receiver)
    if not (.010<float(actual['width_m'])<.045 and abs(float(other['width_m'])-OPEN)<=.001):
        raise StageRefusal('Warm left actual jaws changed before full-scene preparation')
    sender_arrival=float(actual['timestamp'])
    if not np.isfinite(sender_arrival) or sender_arrival<float(source['timestamp']):
        raise StageRefusal('Warm left preparation requires a current actual sender encoder epoch')
    guard=geometry().screen_q16_path([actual['robot_encoder_values']],scene=scene,
        moving_arm=receiver,held_arm=sender,held_body_pose=_body(actual,belief).tolist(),
        object_in_tcp_position=belief['object_in_tcp_position'],
        object_in_tcp_rotation=belief['object_in_tcp_rotation'],target='red',
        finger_target_contract=contract,board_anchor_exit=None,held_envelope_m=_held_envelope(belief))
    if not _guard_passes(guard,scene):
        raise StageRefusal('Warm left actual fixture failed full public geometry '+json.dumps(_failure_summary(guard)))

    state.update(planning_scene=scene,conditional_motor_history_debug=True)
    state,belief,scene,preclose,recenter=_recenter_preclose(ctx,state,sender,receiver,
        belief,scene,candidate,contract,sender_arrival)
    state['planning_scene']=scene
    state['handoff'].update(stage='prepared',candidate=candidate,contract=contract,
        sender_belief=belief,sender_arrival_timestamp=sender_arrival,preclose_rgb=preclose,
        body_before_close=_body(base.proprio(ctx,sender),belief).tolist(),previous_handoffs_earned=False,
        record=dict(warm_left_prepare=True,source=source,scene_refresh=scene_refresh,
            actual_geometry=guard,actual_sender_epoch=actual,
            sender_epoch_role='Current actual fixture encoder epoch; no claimed earlier sender motor arrival',
            previous_handoffs_earned=False,preclose_recenter=recenter))
    return result('prepared',state,state['observation'],'warm_left_prepare',handoff=state['handoff'])


def _receiver_face_measurement(observed,receiver,candidate):
    """Selected bead location in the genuine same-image public TCP frame.

    This is a measured finite-pad control qualification, not a retention or
    friction certificate. Physical dual-held/receiver-only CPs remain required.
    """
    source=_source(observed,'Actual close finite-face qualification from RGB and synchronous q16')
    q=np.asarray(source['robot_encoder_values'],float)
    from .vision_planning import finite_fk
    g=geometry();g.fk(q);ik=g.robot.iks['left'if receiver==0 else'right']
    pp,qq=finite_fk(ik,q);handR=Rotation.from_quat(qq).as_matrix()
    body=np.asarray(source['pose']);bodyR=Rotation.from_quat(body[3:]).as_matrix()
    bead=int(candidate['bead']);center=body[:3]+bodyR@np.asarray(model.local_beads('red')[bead])
    local=handR.T@(center-np.asarray(pp))
    face=candidate.get('finite_face_qualification',{})
    low=float(face.get('minimum_bead_depth_m',.006));high=float(face.get('maximum_bead_depth_m',.0135))
    side=float(face.get('maximum_abs_bead_x_m',.002))
    reserve=float(face.get('minimum_front_face_reserve_m',.003))
    if not(.006<=low<=high<=.0135 and 0.<side<=.002 and .003<=reserve<=.0105):
        raise StageRefusal('Final finite-face qualification cannot exceed the reviewed interior control band')
    front=.0165-float(local[2])
    passed=bool(low<=local[2]<=high and abs(local[0])<=side and front>=reserve)
    return dict(passed=passed,selected_bead=bead,bead_in_same_image_TCP_m=local.tolist(),
        permitted_depth_m=[low,high],maximum_abs_x_m=side,source=source,
        public_pad_front_z_m=.0165,front_face_reserve_m=front,minimum_front_face_reserve_m=reserve,
        measured_control_qualification_only=True,
        actual_dynamic_retention_proven=False)


def _open_receiver_for_recenter(ctx,state,sender,receiver,belief,scene):
    """Sender stays closed while a refused receiver close is repaired."""
    actual=base.proprio(ctx,receiver);guards=[]
    for width in np.linspace(actual['width_m'],OPEN,max(2,int(np.ceil(abs(OPEN-actual['width_m'])/.001))+1)):
        q=np.asarray(actual['robot_encoder_values']);q[8*receiver+6]=width/2.;q[8*receiver+7]=-width/2.
        guard=geometry().screen_q16_path([q],scene=scene,moving_arm=receiver,held_arm=sender,
            held_body_pose=_body(base.proprio(ctx,sender),belief).tolist(),
            object_in_tcp_position=belief['object_in_tcp_position'],object_in_tcp_rotation=belief['object_in_tcp_rotation'],
            target='red',finger_target_contract=state['handoff']['contract'],board_anchor_exit=None,
            held_envelope_m=_held_envelope(belief))
        if not _guard_passes(guard,scene):
            raise c.PlanningRefusal('Retained-sender receiver-reopening sweep refused '+json.dumps(_failure_summary(guard)))
        guards.append(guard)
    opened=ctx.tool('robot.set_grip',arm_id=receiver,width_m=OPEN,ramp_steps=16,settle_steps=8)
    if base.proprio(ctx,receiver)['width_m']<.045:
        raise StageRefusal('Actual receiver must reopen before finite-face recenter')
    return dict(opening=opened,full_material_sweep=guards,sender_opened=False)


def receiver_close(ctx,state):
    """Close, qualify the actual finite-face location, and repair in place."""
    state=deepcopy(state);sender,receiver=_roles(state)
    final_right=(receiver==1 and state['handoff'].get('hop')==2)
    if not final_right:return _receiver_close_once(ctx,state)
    attempts=[]
    for attempt in range(3):
        partial_feedback=None
        try:
            closed=_receiver_close_once(ctx,state);state=closed['state'];h=state['handoff']
            observed=state['observation'];qualification=None
            if 'red'in observed.get('poses',{}):
                qualification=_receiver_face_measurement(observed,receiver,h['candidate'])
        except _ReceiverFaceRecenter as recovery:
            state=recovery.state;h=state['handoff'];observed=state['observation']
            qualification=recovery.qualification;partial_feedback=recovery.feedback
        row=dict(attempt=attempt,finite_face_qualification=qualification,
            visual_unknown=qualification is None,sender_still_retained=True,
            partial_close_feedback=partial_feedback)
        attempts.append(row)
        if qualification is not None and qualification['passed']:
            h['receiver_close_record']['finite_face_attempts']=attempts
            h['receiver_close_record']['actual_finite_face_qualified']=True
            return result('dual_held',state,observed,'receiver_close',handoff=h)
        if attempt==2:
            raise StageRefusal('Actual final receiver finite-face qualification exhausted while sender retained '+json.dumps(attempts))
        belief=_belief(ctx,state,sender);state,scene,refresh=scene_for_motion(ctx,state,belief)
        row['scene_refresh']=refresh
        row['reopening']=_open_receiver_for_recenter(ctx,state,sender,receiver,belief,scene)
        # This revokes the unsuccessful receiver epoch through the ordinary
        # opening hook. No independent dual-held CP has yet been earned and
        # the donor hand has never been opened.
        state['handoff'].pop('receiver_attachment',None)
        state['handoff']['stage']='prepared'
        candidate=state['handoff']['candidate'];contract=state['handoff']['contract']
        state,belief,scene,preclose,recenter=_recenter_preclose(ctx,state,sender,receiver,belief,scene,
            candidate,contract,state['handoff']['sender_arrival_timestamp'])
        state['handoff'].update(preclose_rgb=preclose,sender_belief=belief)
        row['recenter']=recenter
    raise AssertionError('Unreachable finite-face retry exit')


def _receiver_close_once(ctx,state):
    state=deepcopy(state);sender,receiver=_roles(state);h=state['handoff']
    if h.get('stage')!='prepared':raise StageRefusal('Receiver close requires prepared typed handoff')
    if h['preclose_rgb']['timestamp']<h['sender_arrival_timestamp']:
        raise StageRefusal('Receiver close refuses a pre-motion dated world source')
    before=base.proprio(ctx,receiver)
    belief=_belief(ctx,state,sender);scene=scene_belief(state,state['observation'])
    expected_loaded=.0242
    if before['width_m']<=expected_loaded:
        raise StageRefusal('Receiver must start genuinely open before close')
    close_sweep=[]
    for width in np.linspace(before['width_m'],expected_loaded,14):
        qq=np.asarray(before['robot_encoder_values']);qq[8*receiver+6]=width/2.;qq[8*receiver+7]=-width/2.
        check=geometry().screen_q16_path([qq.tolist()],scene=scene,moving_arm=receiver,held_arm=sender,
            held_body_pose=_body(base.proprio(ctx,sender),belief).tolist(),
            object_in_tcp_position=belief['object_in_tcp_position'],object_in_tcp_rotation=belief['object_in_tcp_rotation'],
            target='red',finger_target_contract=h['contract'],board_anchor_exit=None,held_envelope_m=_held_envelope(belief))
        if not _guard_passes(check,scene):
            raise c.PlanningRefusal('Receiver public jaw material sweep refused '+json.dumps(_failure_summary(check)))
        close_sweep.append(check)
    close_commands=[];close_feedback=[]
    for command_width in (.030,.026,expected_loaded,0.):
        # Any new RGB binding invalidates the preceding jaw-path proposal.
        # Recompute this stage's material sweep from the latest real q16/body.
        # Fullclose is force-limited by intended sphere contact: command0 is
        # never a purported actual-width0 geometry sample.
        current=base.proprio(ctx,receiver)
        lower=max(expected_loaded,command_width)
        stage_sweep=[]
        for width in np.linspace(current['width_m'],lower,max(2,int(np.ceil(abs(current['width_m']-lower)/.001))+1)):
            qq=np.asarray(current['robot_encoder_values']);qq[8*receiver+6]=width/2.;qq[8*receiver+7]=-width/2.
            check=geometry().screen_q16_path([qq.tolist()],scene=scene,moving_arm=receiver,held_arm=sender,
                held_body_pose=_body(base.proprio(ctx,sender),belief).tolist(),
                object_in_tcp_position=belief['object_in_tcp_position'],object_in_tcp_rotation=belief['object_in_tcp_rotation'],
                target='red',finger_target_contract=h['contract'],board_anchor_exit=None,
                held_envelope_m=_held_envelope(belief))
            if not _guard_passes(check,scene):
                raise c.PlanningRefusal('Fresh RGB progressive jaw-path material sweep refused '+json.dumps(_failure_summary(check)))
            stage_sweep.append(check)
        # Public sphere geometry sets the approach ladder; actual encoders and
        # fresh RGB determine every result. No commanded width is relabelled
        # as actual jaw geometry, and sender remains retained throughout.
        close=ctx.tool('robot.set_grip',arm_id=receiver,width_m=command_width,
                       ramp_steps=8 if command_width else 14,settle_steps=6)
        close_commands.append(close)
        actual=base.proprio(ctx,receiver)
        if not .008<actual['width_m']<.045:
            raise StageRefusal('Receiver actual obstructed jaw is not a retained-grasp candidate')
        belief=_belief(ctx,state,sender)
        observed=c.state(ctx,{'red':_body(base.proprio(ctx,sender),belief).tolist()});state=save_observation(state,observed)
        postclose_rebinding=None
        if 'red' in observed.get('poses',{}):
            state,belief,postclose_rebinding=_rebind_sender_observation(ctx,state,observed,sender,belief,
                'Actual partial/complete-close dual-hand Red RGB and synchronous q16')
            if command_width>0.:
                # This is a genuinely later dated image before the final full
                # close, useful if only that final image becomes occluded.
                h['preclose_rgb']=_source(observed,'Actual partial-close RGB before final fullclose; sender retained')
        state,scene,postclose_scene_refresh=scene_for_motion(ctx,state,belief)
        actual=base.proprio(ctx,receiver)
        guard=geometry().screen_q16_path([actual['robot_encoder_values']],scene=scene,moving_arm=receiver,
            held_arm=sender,held_body_pose=_body(base.proprio(ctx,sender),belief).tolist(),
            object_in_tcp_position=belief['object_in_tcp_position'],object_in_tcp_rotation=belief['object_in_tcp_rotation'],
            target='red',finger_target_contract=h['contract'],board_anchor_exit=None,held_envelope_m=_held_envelope(belief))
        if not _guard_passes(guard,scene):
            raise StageRefusal('Actual dual-jaw material geometry refused '+json.dumps(_failure_summary(guard)))
        close_feedback.append(dict(command_width_m=command_width,fresh_source_material_stage_sweep=stage_sweep,actual_robot=actual,
            actual_material_guard=guard,actual_RGB_rebinding=postclose_rebinding,scene_refresh=postclose_scene_refresh))
        if receiver==1 and h.get('hop')==2:
            face=(_receiver_face_measurement(observed,receiver,h['candidate'])
                if 'red'in observed.get('poses',{})else None)
            close_feedback[-1]['finite_face_qualification']=face
            if face is None or not face['passed']:
                # Stop at this partial close. Never continue squeezing toward
                # an edge grasp or release the donor to make the route pass.
                raise _ReceiverFaceRecenter(state,face,deepcopy(close_feedback))
    if receiver==1 and h.get('hop')==2:
        # Check the actual settled grasp, not only the image taken directly
        # after the last jaw command. A bad settled face stays recoverable in
        # receiver_close before any independent dual-held credit is earned.
        ctx.tool('robot.wait_steps',steps=6)
        belief=_belief(ctx,state,sender)
        observed=c.state(ctx,{'red':_body(base.proprio(ctx,sender),belief).tolist()})
        state=save_observation(state,observed)
        if 'red'not in observed.get('poses',{}):
            raise _ReceiverFaceRecenter(state,None,deepcopy(close_feedback))
        state,belief,postclose_rebinding=_rebind_sender_observation(ctx,state,observed,sender,belief,
            'Actual settled-close dual-hand Red RGB and synchronous q16')
        state,scene,postclose_scene_refresh=scene_for_motion(ctx,state,belief)
        actual=base.proprio(ctx,receiver)
        guard=geometry().screen_q16_path([actual['robot_encoder_values']],scene=scene,moving_arm=receiver,
            held_arm=sender,held_body_pose=_body(base.proprio(ctx,sender),belief).tolist(),
            object_in_tcp_position=belief['object_in_tcp_position'],object_in_tcp_rotation=belief['object_in_tcp_rotation'],
            target='red',finger_target_contract=h['contract'],board_anchor_exit=None,held_envelope_m=_held_envelope(belief))
        if not _guard_passes(guard,scene):
            raise StageRefusal('Actual settled dual-jaw material geometry refused '+json.dumps(_failure_summary(guard)))
        face=_receiver_face_measurement(observed,receiver,h['candidate'])
        close_feedback.append(dict(phase='six_step_settled_close',finite_face_qualification=face,
            actual_RGB_rebinding=postclose_rebinding,actual_material_guard=guard,scene_refresh=postclose_scene_refresh))
        if not face['passed']:
            raise _ReceiverFaceRecenter(state,face,deepcopy(close_feedback))
    if 'red' in observed.get('poses',{}):
        receiver_source=_source(observed,'Actual dual-held Red RGB and same-image receiver q16')
        receiver_belief=ctx.tool('puzzle.propagate_pyramid_held_belief',source=receiver_source,arm_id=receiver)
        attachment=dict(kind='actual_RGB_q16',source=receiver_source,belief=receiver_belief)
    else:
        conditional=ctx.tool('puzzle.capture_conditional_attachment',source_rgb=h['preclose_rgb'],arm_id=receiver,
            assumption=dict(source_stationary_or_bounded_until_close=True,bead_displacement_bound_m=.008,
                statement='Dated actual preclose Red RGB; closed sender remains stationary until actual receiver close. Independent dual-hold checkpoint required.'))
        receiver_belief=ctx.tool('puzzle.propagate_conditional_attachment',attachment_id=conditional['attachment_id'],arm_id=receiver)
        attachment=dict(kind='conditional_dated_RGB_actual_close',record=conditional,belief=receiver_belief)
    if not(receiver==1 and h.get('hop')==2):ctx.tool('robot.wait_steps',steps=6)
    h=state['handoff'];h.update(stage='dual_held',receiver_attachment=attachment,
        receiver_close_record=dict(actual_before=before,expected_loaded_width_m=expected_loaded,public_material_close_sweep=close_sweep,close_commands=close_commands,progressive_actual_feedback=close_feedback,actual_after=actual,actual_material_guard=guard,postclose_RGB_rebinding=postclose_rebinding,postclose_scene_refresh=postclose_scene_refresh))
    return result('dual_held',state,state['observation'],'receiver_close',handoff=h)


def _postopen_receiver_refresh(ctx,state,receiver,belief):
    """Opening can relax a dual-hand load; measure before the empty hand moves.

    The old receiver attachment only supplies an RGB optimizer proposal.  A
    missing image keeps its original declared bound and dated provenance; it
    never creates a new measured source or an earned handoff checkpoint.
    """
    records=[]
    for attempt in range(2):
        predicted=_body(base.proprio(ctx,receiver),belief)
        observed=c.state(ctx,{'red':predicted.tolist()});state=save_observation(state,observed)
        row=dict(timestamp=observed['timestamp'],status='RGB_UNKNOWN',current_target_measured=False)
        if 'red' in observed.get('poses',{}):
            state,belief,rebound=_rebind_sender_observation(ctx,state,observed,receiver,belief,
                'Fresh actual Red RGB and same-image receiver q16 after sender opening')
            row.update(status='actual_RGB_q16_rebound',current_target_measured=True,rebound=rebound)
            records.append(row);break
        records.append(row)
        ctx.tool('robot.wait_steps',steps=6)
    state,scene,scene_refresh=scene_for_motion(ctx,state,belief)
    return state,belief,scene,dict(attempts=records,scene_refresh=scene_refresh,
        held_envelope_m=_held_envelope(belief),sender_opening_does_not_certify_receiver_hold=True)


def _withdrawal_families(actual,sender,body):
    """Public local geometric proposals, freshly solved and fully screened.

    Short extraction precedes a change of direction.  These are Cartesian
    proposals relative to current robot/body observations, never saved motor
    samples or a successful trajectory replay.
    """
    rr=Rotation.from_quat(actual['tcp'][3:]).as_matrix();origin=np.asarray(actual['tcp'][:3])
    outward=np.array([0.,1. if sender==0 else -1.,0.])
    up=np.array([0.,0.,1.]);back=-rr[:,2]
    # Public full-scene math on authentic second-open RGB/q16 found the short
    # 20mm extraction then 65mm up retains .228rad reserve.  It is still only
    # an online geometric proposal, recomputed and screened from this epoch.
    short=('local_back_20_then_up65',[origin+back*.020,origin+back*.020+up*.065])
    original=[('local_back_85',[origin+back*.085]),('world_up_85',[origin+up*.085])]
    candidates=([short]+original) if sender==0 else (original+[short])
    for length in (.020,.030,.015):
        first=origin+back*length
        candidates.extend([
            ('local_back_%d_then_up'%round(length*1000),[first,first+up*.070]),
            ('local_back_%d_then_out_up'%round(length*1000),[first,first+outward*.065+up*.045]),
        ])
    away=origin-np.asarray(body[:3]);away/=max(float(np.linalg.norm(away)),1.e-12)
    for length in (.020,.030):
        first=origin+away*length
        candidates.extend([
            ('radial_%d_then_up'%round(length*1000),[first,first+up*.065]),
            ('radial_%d_then_out_up'%round(length*1000),[first,first+outward*.065+up*.045]),
        ])
    candidates.append(('world_out_up',[origin+outward*.085+up*.030]))
    return [(name,[_waypoint(point,rr) for point in points]) for name,points in candidates]


def _withdraw_open_sender(ctx,state,belief,scene):
    sender,receiver=_roles(state);h=state['handoff']
    actual=base.proprio(ctx,sender)
    if actual['width_m']<=.045:
        raise StageRefusal('Sender must actually be open before empty-hand withdrawal')
    body=_body(base.proprio(ctx,receiver),belief)
    # The opened hand is empty.  Its old selected-bead contact permission is
    # gone as well as its old attachment: ordinary 0.5mm clearance applies.
    receiver_beads=h['contract'].get(receiver,h['contract'].get(str(receiver)))
    if not receiver_beads:raise StageRefusal('Opened-sender withdrawal needs the named receiver contact only')
    contract={receiver:receiver_beads}
    refusals=[];motion=None;chosen=None
    for name,waypoints in _withdrawal_families(actual,sender,body):
        try:
            motion=_screen(ctx,state,scene,belief,receiver,sender,waypoints,contract);chosen=name;break
        except c.PlanningRefusal as error:
            refusals.append(dict(family=name,refusal=str(error)))
    if motion is None:
        raise c.PlanningRefusal('Sender withdrawal refused after retained receiver close '+json.dumps(refusals))
    execution=_execute(ctx,state,motion,retained_arm=receiver)
    actual=base.proprio(ctx,sender);target_pose=_body(base.proprio(ctx,receiver),_belief(ctx,state,receiver))
    if actual['width_m']<=.045 or c.part_hand_gap('red',target_pose,np.asarray(actual['tcp'][:3]))<=.055:
        raise StageRefusal('Actual sender did not open and withdraw from receiver-held target')
    ctx.tool('robot.wait_steps',steps=6)
    return dict(family=chosen,execution=execution,refusals=refusals)


def sender_release(ctx,state):
    state=deepcopy(state);sender,receiver=_roles(state);h=state['handoff']
    if h.get('stage')!='dual_held':raise StageRefusal('Sender release requires preceding independent dual-hold checkpoint')
    attachment=h['receiver_attachment'];p={**state['parameters']}
    if attachment['kind']=='actual_RGB_q16':
        p['owned_held_belief_source']=attachment['source'];p.pop('conditional_attachment_id',None)
    else:
        p['conditional_attachment_id']=attachment['record']['attachment_id'];p.pop('owned_held_belief_source',None)
    state.update(arm=receiver,parameters=p);belief=_belief(ctx,state,receiver)
    preopen_face=None
    if receiver==1 and h.get('hop')==2:
        # A previous good close and dual-held CP cannot authorize opening
        # after the bead has subsequently moved to the finite pad edge.
        predicted=_body(base.proprio(ctx,receiver),belief)
        observed=c.state(ctx,{'red':predicted.tolist()});state=save_observation(state,observed)
        if 'red'not in observed.get('poses',{}):
            raise StageRefusal('Final donor remains held: pre-opening finite-face RGB is UNKNOWN')
        preopen_face=_receiver_face_measurement(observed,receiver,h['candidate'])
        innovation=visual_memory.shape_displacement('red',predicted,observed['poses']['red'])
        if not preopen_face['passed'] or innovation>_held_envelope(belief):
            # A new regrasp must earn a new independent dual-held CP. Never
            # repair inside sender_release and then spend the old CP credit.
            raise StageRefusal('Final donor remains held: pre-opening finite-face/owned bound refused '+json.dumps(preopen_face))
        preopen_face.update(original_attachment_not_replaced=True,original_bound_m=_held_envelope(belief),
            measured_bead_innovation_m=float(innovation),new_regrasp_would_require_new_dual_hold_CP=True)
        state,scene,preopen_scene_refresh=scene_for_motion(ctx,state,belief)
        belief=_belief(ctx,state,receiver)
    else:
        scene=scene_belief(state,state['observation']);preopen_scene_refresh=None
    actual=base.proprio(ctx,sender);body=_body(base.proprio(ctx,receiver),belief)
    guards=[]
    for width in np.linspace(actual['width_m'],OPEN,16):
        qq=np.asarray(actual['robot_encoder_values']);qq[8*sender+6]=width/2.;qq[8*sender+7]=-width/2.
        guard=geometry().screen_q16_path([qq.tolist()],scene=scene,moving_arm=sender,held_arm=receiver,
            held_body_pose=body.tolist(),object_in_tcp_position=belief['object_in_tcp_position'],
            object_in_tcp_rotation=belief['object_in_tcp_rotation'],target='red',
            finger_target_contract=h['contract'],board_anchor_exit=None,held_envelope_m=_held_envelope(belief))
        if not _guard_passes(guard,scene):
            raise c.PlanningRefusal('Sender opening material sweep refused '+json.dumps(_failure_summary(guard)))
        guards.append(guard)
    opened=ctx.tool('robot.set_grip',arm_id=sender,width_m=OPEN,ramp_steps=16,settle_steps=10)
    # The registered opening hook permanently revokes all preopening sender
    # source epochs; receiver attachment is arm-specific and remains retained.
    state,belief,scene,refresh=_postopen_receiver_refresh(ctx,state,receiver,belief)
    withdrawal=_withdraw_open_sender(ctx,state,belief,scene)
    state['handoff'].update(stage='transferred',sender_old_source_revoked=True,
        sender_release_record=dict(width_sweep=guards,opening=opened,postopen_receiver_refresh=refresh,
            actual_preopen_finite_face=preopen_face,preopen_scene_refresh=preopen_scene_refresh,
            withdrawal=withdrawal['execution'],withdrawal_family=withdrawal['family'],refusals=withdrawal['refusals']))
    return result('transferred',state,state['observation'],'sender_release',handoff=state['handoff'])


def warm_prepare(ctx,state):
    """Bind a new debug-tail Env from actual images, never from an old state.

    The harness may establish a conditional fixture with public motor history.
    That history earns no earlier checkpoint and supplies no attachment here.
    """
    state=deepcopy(state)
    if not state['parameters'].get('red_bimanual_warm_debug',False):
        raise StageRefusal('Warm preparation is an explicitly scoped conditional debug-tail API')
    if (state['parameters'].get('owned_held_belief_source') or
            state['parameters'].get('conditional_attachment_id') or
            state.get('execution_tokens') or state.get('executed_placements')):
        raise StageRefusal('Warm debug requires a fresh state without old sources or earned execution')
    from .red_handoff_candidates import CANDIDATES
    candidate=deepcopy(CANDIDATES[0]['final_right_public_grasp']);candidate['arm']=1
    state.update(arm=1,piece='red')
    state['handoff']=dict(sender=0,receiver=1,source_selected_bead=1,hop=2,
        contract={0:[1],1:[int(candidate['bead'])]},candidate=candidate,
        stage='conditional_debug_sender_open',previous_handoffs_earned=False,
        source_scope='Fresh current RGB/q16 only; motor-history fixture is conditional debug')
    sender=base.proprio(ctx,0);receiver=base.proprio(ctx,1)
    if sender['width_m']<=.045 or receiver['width_m']<.010:
        raise StageRefusal('Warm actual jaws contradict open sender and loaded receiver proposal')
    handR=Rotation.from_quat(receiver['tcp'][3:]).as_matrix()
    bodyR=handR@np.asarray(candidate['hand_in_body_rotation']).T
    hint=np.r_[np.asarray(receiver['tcp'][:3])-bodyR@np.asarray(candidate['hand_in_body_position']),
               Rotation.from_matrix(bodyR).as_quat()]
    observed=c.state(ctx,{'red':hint.tolist()});state=save_observation(state,observed)
    source=_source(observed,'New actual warm-tail Red RGB and same-image right q16; no prior epoch')
    if c.part_hand_gap('red',np.asarray(source['pose']),np.asarray(receiver['tcp'][:3]))>.045:
        raise StageRefusal('Warm actual Red RGB is not near the proposed receiver grasp')
    parameters={**state['parameters'],'owned_held_belief_source':source}
    parameters.pop('conditional_attachment_id',None)
    state['parameters']=parameters
    belief=_belief(ctx,state,1)
    state,scene,scene_refresh=scene_for_motion(ctx,state,belief)
    actual=base.proprio(ctx,1)
    guard=geometry().screen_q16_path([actual['robot_encoder_values']],scene=scene,moving_arm=0,held_arm=1,
        held_body_pose=_body(actual,belief).tolist(),object_in_tcp_position=belief['object_in_tcp_position'],
        object_in_tcp_rotation=belief['object_in_tcp_rotation'],target='red',
        finger_target_contract={1:[int(candidate['bead'])]},board_anchor_exit=None,held_envelope_m=_held_envelope(belief))
    if not _guard_passes(guard,scene):
        raise StageRefusal('Warm actual open-sender retained-receiver scene failed full public geometry '+json.dumps(_failure_summary(guard)))
    state.update(planning_scene=scene,conditional_motor_history_debug=True)
    state['handoff'].update(stage='warm_bound',warm_prepare_record=dict(source=source,
        scene_refresh=scene_refresh,actual_geometry=guard,previous_handoffs_earned=False))
    return result('warm_bound',state,observed,'warm_prepare',handoff=state['handoff'])


def warm_withdraw(ctx,state):
    state=deepcopy(state);sender,receiver=_roles(state)
    if (not state['parameters'].get('red_bimanual_warm_debug',False) or
            state['handoff'].get('stage')!='warm_bound' or
            not state.get('conditional_motor_history_debug')):
        raise StageRefusal('Warm withdrawal requires a fresh measured conditional debug-tail binding')
    belief=_belief(ctx,state,receiver)
    state,belief,scene,refresh=_postopen_receiver_refresh(ctx,state,receiver,belief)
    withdrawal=_withdraw_open_sender(ctx,state,belief,scene)
    state['handoff'].update(stage='transferred',sender_old_source_revoked=True,
        previous_handoffs_earned=False,sender_release_record=dict(opening_already_actual=True,
        postopen_receiver_refresh=refresh,withdrawal=withdrawal['execution'],
        withdrawal_family=withdrawal['family'],refusals=withdrawal['refusals']))
    return result('transferred',state,state['observation'],'warm_withdraw',handoff=state['handoff'])


def receiver_bind(ctx,state):
    state=deepcopy(state);sender,receiver=_roles(state);h=state['handoff']
    if h.get('stage')!='transferred' or state['arm']!=receiver:
        raise StageRefusal('Receiver binding requires actual sender-open typed transfer')
    attempts=[];belief=_belief(ctx,state,receiver)
    for attempt in range(2):
        observed=c.state(ctx,{'red':belief['predicted_body_pose']});state=save_observation(state,observed)
        row=dict(timestamp=observed['timestamp'],visible_names=sorted(observed.get('poses',{})))
        if 'red' in observed.get('poses',{}):
            source=_source(observed,'Fresh posttransfer actual Red RGB and same-image receiver q16')
            actual=base.proprio(ctx,receiver);innovation=visual_memory.shape_displacement('red',belief['predicted_body_pose'],source['pose'])
            gap=c.part_hand_gap('red',np.asarray(source['pose']),np.asarray(actual['tcp'][:3]))
            if innovation>.020 or gap>.045:
                # An accepted image fit can still be a tracking ambiguity.
                # Preserve the authentic raw image, reject it for control and
                # try another real observation without replacing the old epoch.
                row.update(status='RGB_CONFLICT_REJECTED_FOR_CONTROL',current_target_measured=False,
                    rejected_source=source,bead_innovation_m=float(innovation),hand_gap_m=float(gap),
                    owned_source_replaced=False)
                attempts.append(row);ctx.tool('robot.wait_steps',steps=6)
                belief=_belief(ctx,state,receiver)
                continue
            p={**state['parameters'],'owned_held_belief_source':source};p.pop('conditional_attachment_id',None)
            state.update(parameters=p,prediction=source['pose'])
            # Binding/physical held verification issues no scene motion. An
            # expired board observation cannot invalidate a fresh actual Red
            # attachment. Every later motion rebuilds its own dated scene.
            state['scene_refresh_required_before_next_motion']=True
            fresh=_belief(ctx,state,receiver);row.update(status='actual_RGB_q16_bound',source=source,belief=fresh)
            attempts.append(row);state['handoff'].update(stage='verified',receiver_bind_record=dict(attempts=attempts,current_target_measured=True))
            ctx.tool('robot.wait_steps',steps=6)
            return result('verified',state,observed,'receiver_bind',handoff=state['handoff'])
        row.update(status='RGB_UNKNOWN',current_target_measured=False);attempts.append(row)
        ctx.tool('robot.wait_steps',steps=6)
    # Physical receiver-only success and current visual localization are
    # separate outputs. Preserve the authenticated conditional token and its
    # dated actual source/close epochs; never mint a measured raw Red pose.
    # The independent receiver-held checkpoint is still mandatory here.
    if any(row.get('status')=='RGB_CONFLICT_REJECTED_FOR_CONTROL'for row in attempts):
        state,belief,retained=_retained_receiver_scene_check(ctx,state,receiver)
        state['handoff']['receiver_control_binding_status']='dated_original_bound_retained_after_visual_conflict'
    else:retained=None
    state['scene_refresh_required_before_next_motion']=True
    state['handoff'].update(stage='verified',receiver_bind_record=dict(attempts=attempts,
        current_target_measured=False,visual_status='RGB_UNKNOWN',
        retained_after_conflict=retained,
        attachment_semantics='Authenticated receiver attachment is conditional or dated; fresh posttransfer Red RGB unavailable',
        placement_or_goal_localization_claim=False))
    state['prediction']=belief['predicted_body_pose']
    return result('verified',state,state['observation'],'receiver_bind',handoff=state['handoff'],
                  visual_status='RGB_UNKNOWN',current_target_measured=False)


def return_handoff_prepare(ctx,state):
    """Typed first receiver state -> second handoff, without pose synthesis.

    The final-right CAD grasp is only a proposal. The current left attachment
    remains authenticated actual RGB/q16 or dated RGB + distinct actual close.
    Both right-hand native paths and all scene gates are recomputed afresh.
    """
    state=deepcopy(state);previous=deepcopy(state.get('handoff',{}))
    if (previous.get('stage')!='verified' or int(previous.get('hop',1))!=1
            or int(state.get('arm',-1))!=int(previous.get('receiver',-2))):
        raise StageRefusal('Return handoff requires preceding typed first receiver-bound state')
    recipe=deepcopy(previous.get('candidate',{}).get('final_right_public_grasp'))
    if not isinstance(recipe,dict) or int(recipe.get('bead',-1))!=4:
        raise StageRefusal('First candidate lacks a public final-right bead4 graph edge')
    sender=int(state['arm']);receiver=1-sender
    if receiver!=1:raise StageRefusal('This two-hop Red strategy ends in the right hand')
    source_bead=int(previous['candidate']['bead'])
    recipe['arm']=receiver
    p={**state['parameters'],'red_handoff_sender_bead':source_bead,'red_handoff_candidates':[recipe]}
    state['handoff_history']=[*state.get('handoff_history',[]),previous]
    state.update(parameters=p,handoff=dict(sender=sender,receiver=receiver,hop=2,
        source_selected_bead=source_bead,stage='start',
        prior_binding_semantics=previous.get('receiver_bind_record',{})))
    return handoff_prepare(ctx,state)
