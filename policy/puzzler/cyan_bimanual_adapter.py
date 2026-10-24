"""Cyan hand-to-hand regrasp from owned RGB, public CAD and actual q16.

Body-relative proposals are rebound at runtime. Independent GaP checkpoints
separate receiver approach, dual hold, donor release and receiver retention.
No measured attachment is inferred from a desired grasp transform.
"""
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from . import cyan_geometry as g, cyan_original_transport as tr, model
from . import vision_controller as c
from .cyan_transfer_candidates import proposal_family
from .modular_controller import save_observation
from .release_admission import check_bounded_release

_GEOMETRY = None


def _geometry():
    global _GEOMETRY
    if _GEOMETRY is None:
        from .public_bimanual_geometry import PublicBimanualGeometry
        _GEOMETRY = PublicBimanualGeometry(device='cuda:0', minimum_clearance_m=.0005)
    if not hasattr(_GEOMETRY, 'screen_q16_path'):
        raise c.PlanningRefusal('Complete public robot/scene path screening is not installed')
    return _GEOMETRY


def _belief(ctx, state):
    return tr._belief(ctx, state)


def _scene(state):
    scene = deepcopy(state['planning_scene'])
    for name, entry in scene['occupancy'].items():
        if name != 'cyan':
            entry['inflation_m'] = max(.004, float(entry['inflation_m']))
    return scene


def _contract(state, receiver_contact=False):
    transfer = state['transfer']
    p = state['parameters']
    # The executed inverse-tip source has a narrower explicit contract than
    # legacy unused source-profile pairs. Never union those legacy permissions.
    if 'source_contact_bead_indices' in p:
        sender_beads = sorted(set(int(i) for i in p['source_contact_bead_indices']))
    else:
        pairs = p.get('source_contact_grasp', {}).get('permitted_pad_bead_pairs', ())
        sender_beads = sorted({int(row['target_bead_index']) for row in pairs})
    if not sender_beads:
        raise c.PlanningRefusal('Sender contact contract has no named public bead')
    result = {int(transfer['sender']): sender_beads}
    if receiver_contact:
        result[int(transfer['receiver'])] = [int(transfer['candidate']['bead_index'])]
    return result


def _guard(state, q16_path, moving_arm, belief, *, receiver_contact=False, anchor_exit=None):
    checked = _geometry().screen_q16_path(
        q16_path=np.asarray(q16_path).tolist(), scene=_scene(state), moving_arm=int(moving_arm),
        held_arm=int(state['arm']), held_body_pose=belief['predicted_body_pose'],
        object_in_tcp_position=belief['object_in_tcp_position'],
        object_in_tcp_rotation=belief['object_in_tcp_rotation'], target='cyan',
        finger_target_contract=_contract(state, receiver_contact), board_anchor_exit=anchor_exit)
    if not checked.get('safe', checked.get('passed', False)):
        raise c.PlanningRefusal('Cyan handoff complete public geometry refused: ' + str(checked))
    return checked


def _qpath(plan, q16, arm):
    arm_path = np.asarray(plan['path'], dtype=float)
    if arm_path.ndim != 2 or arm_path.shape[1] != 6 or not np.isfinite(arm_path).all():
        raise c.PlanningRefusal('Handoff native joint path is malformed')
    rows = [arm_path[0]]
    for a, b in zip(arm_path[:-1], arm_path[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(b-a))/.012)))
        rows.extend(a+(b-a)*u for u in np.linspace(0., 1., count+1)[1:])
    q = np.tile(np.asarray(q16, dtype=float), (len(rows), 1))
    q[:, (0 if arm == 0 else 8):(6 if arm == 0 else 14)] = rows
    return q


def _plan(ctx, arm, poses):
    plan = ctx.tool('puzzle.plan_pose_sequence', waypoints=[
        dict(position=np.asarray(p).tolist(), rotation=q) for p, q in poses],
        arm_id=int(arm), min_joint_limit_margin_rad=.035)
    if not plan.get('feasible') or float(plan.get('min_joint_limit_margin_rad', -1.)) < .035:
        raise c.PlanningRefusal('Cyan handoff continuous reserved native IK refused: '
                                + str(plan.get('errors', ())[-1:]))
    return plan


def _execute(ctx, state, arm, poses, phase, *, receiver_contact=False, anchor_exit=None):
    belief = _belief(ctx, state)
    plan = _plan(ctx, arm, poses)
    full = _qpath(plan, belief['current_robot_encoder_values'], arm)
    guard = _guard(state, full, arm, belief, receiver_contact=receiver_contact, anchor_exit=anchor_exit)
    start = 0 if arm == 0 else 8
    executed = []
    # Check the real held jaw at every small motor batch. A contradicted source
    # cannot be revived by continuing along an earlier geometric proposal.
    for begin in range(1, len(full), 16):
        _belief(ctx, state)
        rows = full[begin:begin+16, start:start+6]
        result = ctx.tool('robot.execute_trajectory', trajectory=dict(
            waypoints=[dict(positions=row.tolist()) for row in rows]),
            arm_id=int(arm), max_steps_per_waypoint=2)
        if result.get('status') != 'ok':
            raise RuntimeError('Cyan handoff motor trajectory failed: ' + str(result))
        executed.append(result)
    # No unscreened pose-correction trajectory is hidden in this settling call.
    c.await_motion(ctx, poses[-1][0], poses[-1][1], arm, accuracy=.0025)
    actual = _belief(ctx, state)
    _guard(state, [actual['current_robot_encoder_values']], arm, actual,
           receiver_contact=receiver_contact)
    return dict(phase=phase, native_ik=plan, geometry=guard, motor=executed,
                actual_encoder_endpoint=actual['current_robot_encoder_values'])


def _width_rows(q16, arm, width):
    q = np.asarray(q16, dtype=float)
    offset = 0 if arm == 0 else 8
    old = float(q[offset+6]-q[offset+7])
    center = float((q[offset+6]+q[offset+7])/2.)
    rows = []
    for w in np.linspace(old, width, max(2, int(np.ceil(abs(old-width)/.001)))+1):
        row = q.copy()
        row[offset+6], row[offset+7] = center+w/2., center-w/2.
        rows.append(row)
    return rows


def _grip(ctx, state, arm, command_width, *, loaded_screen_width=None, receiver_contact=False):
    belief = _belief(ctx, state)
    width = command_width if loaded_screen_width is None else loaded_screen_width
    guard = _guard(state, _width_rows(belief['current_robot_encoder_values'], arm, width),
                   arm, belief, receiver_contact=receiver_contact)
    result = ctx.tool('robot.set_grip', arm_id=int(arm), width_m=float(command_width),
                      ramp_steps=18, settle_steps=15)
    # Width zero is the real close command. Hypothetical empty fully closed
    # fingers are never treated as the expected loaded contact configuration.
    actual = _belief(ctx, state)
    endpoint = _guard(state, [actual['current_robot_encoder_values']], arm, actual,
                      receiver_contact=receiver_contact)
    return dict(command_width_m=command_width, expected_loaded_geometry_width_m=width,
                proposal_screen=guard, actual_endpoint_screen=endpoint, motor=result)


def _world_grasp(body, candidate):
    body = np.asarray(body)
    R = Rotation.from_quat(body[3:]).as_matrix()
    H = R @ np.asarray(candidate['hand_in_body_rotation'])
    return body[:3]+R@np.asarray(candidate['hand_in_body_position']), c.qdict(Rotation.from_matrix(H).as_quat())


def _out(route, state, observed, **evidence):
    state = save_observation(state, observed)
    state.update(last_stage=route, last_outcome=route)
    return g._plain(dict(route=route, state=state, evidence=dict(
        kind=route, timestamp=observed['timestamp'], frame_sha256=observed['frame_sha256'],
        policy_inputs='Owned actual RGB, actual q16/public FK, public CAD/goals only', **evidence)))


def approach_receiver(ctx, state):
    state = deepcopy(state)
    if state['piece'] != 'cyan' or int(state['arm']) != 0:
        raise c.PlanningRefusal('This Cyan source handoff expects actual sender arm 0')
    state['transfer'] = dict(sender=0, receiver=1, candidates=[], execution=[],
                             old_sender_source=deepcopy(state['parameters'].get('owned_held_belief_source')))
    transfer = state['transfer']
    receiver = transfer['receiver']
    belief = _belief(ctx, state)
    body = np.asarray(belief['predicted_body_pose'])
    board = np.asarray(state['parameters']['historical_complete_rgb']['poses']['board'])
    Rb = Rotation.from_quat(board[3:]).as_matrix()
    goal_p, goal_R = g.goals('pyramid', state['parameters']['pyramid_plan'])['cyan']
    goal_body = np.r_[board[:3]+Rb@goal_p, Rotation.from_matrix(Rb@goal_R).as_quat()]
    current, current_q = c.hand(ctx, receiver)
    exit_pose = current+[0., 0., .060]
    anchor = state.get('anchor_evidence') or state.get('support_anchor_evidence')
    if not anchor or int(anchor.get('arm', -1)) != receiver:
        raise c.PlanningRefusal('Receiver brace has no actual named original anchor provenance')
    anchor_exit = dict(arm=receiver,direction_world=[0.,0.,1.],
                       maximum_distance_m=.060,source_anchor=anchor)
    exit_plan = _plan(ctx,receiver,[(exit_pose,current_q)])
    transfer['anchor_exit_preview'] = _guard(state,_qpath(
        exit_plan,belief['current_robot_encoder_values'],receiver),receiver,belief,anchor_exit=anchor_exit)
    family = proposal_family(model.local_beads('cyan'))
    # Finite public grasp priority, never an imported world pose or joint path.
    desired_H = np.array([[.25,.433012701892,.866025403784],
                          [-.866025403784,.5,0.],[-.433012701892,-.75,.5]])
    family.sort(key=lambda row: np.linalg.norm(np.asarray(row['hand_in_body_rotation'])-desired_H)
                + abs(float(row['bead_in_tcp_m'][2])-.014))
    chosen = None
    for candidate in family:
        transfer['candidate'] = candidate
        grasp_p, grasp_q = _world_grasp(body, candidate)
        approach = Rotation.from_quat(c.qarr(grasp_q)).as_matrix()[:, 2]
        pre_p = grasp_p-.080*approach
        target_p, target_q = _world_grasp(goal_body, candidate)
        try:
            checks = [ctx.tool('puzzle.plan_pose', position=(target_p+Rb[:,2]*h).tolist(),
                              rotation=target_q, arm_id=receiver, min_joint_limit_margin_rad=.035)
                      for h in (.100, .003)]
            if not all(row['position_error_m']<.002 and row['rotation_error_rad']<.03
                       and row['joint_limit_margin_rad']>=.035 for row in checks):
                raise c.PlanningRefusal('Receiving grasp lacks original-reserve goal/hover IK')
            plan = _plan(ctx, receiver, [(exit_pose,current_q),(pre_p,grasp_q),(grasp_p,grasp_q)])
            # Screen the candidate's actual IK chain before committing the
            # brace exit. After segment 0 the receiver will be open at 55 mm.
            exit_count = sum(int(row['segment']) == 0 for row in plan['errors'])
            approach_plan = {'path': plan['path'][exit_count:]}
            full = _qpath(approach_plan,belief['current_robot_encoder_values'],receiver)
            center = (full[:,14]+full[:,15])/2.
            full[:,14],full[:,15] = center+.055/2.,center-.055/2.
            approach_guard = _guard(state,full,receiver,belief,receiver_contact=True)
            close_preview = _guard(state,_width_rows(full[-1],receiver,
                candidate['nominal_contact_width_m']),receiver,belief,receiver_contact=True)
            transfer['candidates'].append(dict(proposal=candidate, goal_ik=checks,
                                               approach_native_ik=plan,
                                               approach_full_geometry=approach_guard,
                                               close_loaded_geometry_preview=close_preview,selected=True))
            chosen = candidate
            break
        except (ValueError, RuntimeError) as error:
            transfer['candidates'].append(dict(proposal=candidate, refusal=str(error)))
    if chosen is None:
        raise c.PlanningRefusal('No public receiving proposal has continuous approach and goal IK')
    transfer['execution'].append(_execute(ctx,state,receiver,[(exit_pose,current_q)],'withdraw_original_brace',
        anchor_exit=anchor_exit))
    transfer['execution'].append(_grip(ctx,state,receiver,.055))
    belief = _belief(ctx,state)
    grasp_p,grasp_q = _world_grasp(belief['predicted_body_pose'],chosen)
    direction = Rotation.from_quat(c.qarr(grasp_q)).as_matrix()[:,2]
    transfer['execution'].append(_execute(ctx,state,receiver,[(grasp_p-.080*direction,grasp_q)],'receiver_pregrasp'))
    # A real late source epoch keeps the distinct receiver-close assumption
    # bounded without ever synthesizing an image measurement from donor FK.
    belief = _belief(ctx,state)
    observed = c.state(ctx,{'cyan':belief['predicted_body_pose']})
    state = save_observation(state,observed)
    transfer = state['transfer']
    if 'cyan' in observed.get('poses',{}):
        transfer['preclose_source_admission'] = check_bounded_release(
            'cyan',belief,observed,state['parameters'])
        source = g._source(observed,'cyan')
        state['parameters']['owned_held_belief_source'] = source
        transfer['preclose_source'] = deepcopy(source)
    else:
        transfer['preclose_source'] = deepcopy(state['parameters']['owned_held_belief_source'])
    belief = _belief(ctx,state)
    grasp_p,grasp_q = _world_grasp(belief['predicted_body_pose'],chosen)
    transfer['execution'].append(_execute(ctx,state,receiver,[(grasp_p,grasp_q)],'receiver_approach',receiver_contact=True))
    transfer['receiver_actual_ready_proprio'] = _belief(ctx,state)['current_robot_encoder_values']
    observed = c.state(ctx,{'cyan':_belief(ctx,state)['predicted_body_pose']})
    return _out('receiver_ready',state,observed,receiver=receiver,candidate=chosen,
                scope='Donor remains closed. Independent approach checkpoint precedes receiver close.')


def close_receiver(ctx, state):
    state = deepcopy(state)
    transfer = state['transfer']
    receiver = int(transfer['receiver'])
    transfer['execution'].append(_grip(ctx,state,receiver,0.,
        loaded_screen_width=transfer['candidate']['nominal_contact_width_m'],receiver_contact=True))
    source = transfer['preclose_source']
    assumption = dict(source_stationary_or_bounded_until_close=True,
        bead_displacement_bound_m=.006,
        reason='Donor remains commanded closed and stationary throughout actual receiver close; '
               'physical dual hold must pass independently before donor opens.')
    attachment = ctx.tool('puzzle.capture_conditional_attachment',source_rgb=source,
                          assumption=assumption,arm_id=receiver)
    transfer['receiver_conditional_attachment'] = attachment
    transfer['actual_receiver_close_proprio'] = attachment['close_proprio_epoch']
    ctx.tool('robot.wait_steps',steps=8)
    observed = c.state(ctx,{'cyan':_belief(ctx,state)['predicted_body_pose']})
    return _out('receiver_closed',state,observed,
        actual_receiver_close=attachment['close_proprio_epoch'],
        current_target_measured=False,
        scope='Actual full-close command and dated actual RGB binding; dualhold physical CP still required.')


def release_donor(ctx, state):
    state = deepcopy(state)
    transfer = state['transfer']
    donor,receiver = int(transfer['sender']),int(transfer['receiver'])
    # This function is reachable only after a separate raise-mode dualhold CP.
    # Switch active ownership before the real donor opening revokes its lineage.
    p = state['parameters']
    transfer['retired_sender_source'] = p.pop('owned_held_belief_source',None)
    p.pop('conditional_attachment_id',None)
    attachment = deepcopy(transfer['receiver_conditional_attachment'])
    p['conditional_attachment'] = attachment
    p['conditional_attachment_id'] = attachment['attachment_id']
    state.update(arm=receiver,conditional_attachment=attachment,
                 held_status='UNKNOWN_conditional_receiver_attachment')
    transfer['active_owner'] = receiver
    transfer['old_sender_source_active'] = False
    # The smaller release opening avoids squeezing the receiving hand while
    # still clearing the public 22 mm bead, then the donor retreats axially.
    transfer['execution'].append(_grip(ctx,state,donor,.030,receiver_contact=True))
    tcp,q = c.hand(ctx,donor)
    backward = -Rotation.from_quat(c.qarr(q)).as_matrix()[:,2]
    transfer['execution'].append(_execute(ctx,state,donor,[(tcp+.030*backward,q)],
                                          'donor_initial_exit',receiver_contact=True))
    transfer['execution'].append(_grip(ctx,state,donor,.055,receiver_contact=True))
    tcp,q = c.hand(ctx,donor)
    transfer['execution'].append(_execute(ctx,state,donor,[(tcp+.070*backward,q)],
                                          'donor_full_exit',receiver_contact=True))
    state['anchor_evidence'] = {**state.get('anchor_evidence',{}),'arm':donor,
        'previous_board_support_arm':receiver,'support_cleanup_already_completed':True,
        'donor_open_and_withdrawn_after_independent_dualhold':True}
    ctx.tool('robot.wait_steps',steps=8)
    observed = c.state(ctx,{'cyan':_belief(ctx,state)['predicted_body_pose']})
    return _out('donor_released',state,observed,
                current_target_measured=False,active_receiver=receiver,
                scope='Old sender source retired and opening hook permanently revokes it; receiver-only CP required.')


def verify_receiver(ctx, state):
    state = deepcopy(state)
    belief = _belief(ctx,state)
    observed = c.state(ctx,{'cyan':belief['predicted_body_pose']})
    state = save_observation(state,observed)
    if 'cyan' in observed.get('poses',{}):
        state['transfer']['receiver_source_admission'] = check_bounded_release(
            'cyan',belief,observed,state['parameters'])
        source = g._source(observed,'cyan')
        state['parameters']['owned_held_belief_source'] = source
        state['parameters'].pop('conditional_attachment_id',None)
        state['held_status'] = 'CURRENT_RGB_VERIFIED'
        state['transfer']['receiver_fresh_rgb_source'] = source
        _belief(ctx,state)
        measured = True
    else:
        # This route promises physical retention, never a fabricated current
        # target measurement. The independent CP uses its owner-only contacts.
        state['held_status'] = 'UNKNOWN_conditional_receiver_attachment'
        measured = False
    ctx.tool('robot.wait_steps',steps=8)
    return _out('receiver_verified',state,observed,current_target_measured=measured,
                receiver_binding=('actual_new_RGB_q16' if measured else 'dated_actual_RGB_actual_close_proprio_conditional'),
                scope='Receiver hold qualification only; Cyan goal carry/release remain separate unchanged checkpoints.')
