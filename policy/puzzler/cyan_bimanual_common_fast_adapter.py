"""Single-pass Cyan transfer planning, preserving donor attitude.

Only public board-relative positions and body-relative grasp proposals are
stored. Every actual path is rebuilt from the current RGB/encoders and screened
against both arms, the table, all79 board components and scene occupancy.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation

from . import cyan_geometry as g,vision_controller as c,model
from .modular_controller import save_observation
from .release_admission import check_bounded_release
from .cyan_transfer_candidates import proposal_family
from .cyan_bimanual_adapter import (
    _belief,_guard,_qpath,_width_rows,_plan,_execute,_grip,_world_grasp,_out,
    close_receiver,release_donor,verify_receiver)


def _family():
    rows=proposal_family(model.local_beads('cyan'))
    desired=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
    rows.sort(key=lambda row:np.linalg.norm(np.asarray(row['hand_in_body_rotation'])-desired)
              +abs(float(row['bead_in_tcp_m'][2])-.014))
    # Small established geometric family; runtime certificates below remain
    # mandatory. These proposals contain neither saved world poses nor q.
    return rows[:3]


def _future_receiver(ctx,state,body,board,family):
    receiver=int(state['transfer']['receiver']);Rb=Rotation.from_quat(board[3:]).as_matrix()
    goal_p,goal_R=g.goals('pyramid',state['parameters']['pyramid_plan'])['cyan']
    goal=np.r_[board[:3]+Rb@goal_p,Rotation.from_matrix(Rb@goal_R).as_quat()]
    failures=[]
    for candidate in family:
        grasp_p,grasp_q=_world_grasp(body,candidate)
        direction=Rotation.from_quat(c.qarr(grasp_q)).as_matrix()[:,2]
        target_p,target_q=_world_grasp(goal,candidate)
        try:
            goal_checks=[ctx.tool('puzzle.plan_pose',position=(target_p+Rb[:,2]*h).tolist(),rotation=target_q,
                arm_id=receiver,min_joint_limit_margin_rad=.035) for h in (.100,.003)]
            if not all(row['position_error_m']<.002 and row['rotation_error_rad']<.03
                       and row['joint_limit_margin_rad']>=.035 for row in goal_checks):
                raise c.PlanningRefusal('Receiving proposal lacks reserved public goal/hover IK')
            # The free-space orientation transition has no straight-line
            # requirement. Use reserved joint interpolation to pregrasp, then
            # solve the final straight approach again from its actual endpoint.
            plan=ctx.tool('puzzle.plan_joint_path',position=(grasp_p-.080*direction).tolist(),
                rotation=grasp_q,arm_id=receiver,min_joint_limit_margin_rad=.035)
            endpoint=ctx.tool('puzzle.plan_pose',position=grasp_p.tolist(),rotation=grasp_q,
                arm_id=receiver,min_joint_limit_margin_rad=.035)
            if not plan.get('feasible') or endpoint['position_error_m']>.002 or endpoint['rotation_error_rad']>.03 or endpoint['joint_limit_margin_rad']<.035:
                raise c.PlanningRefusal('Joint-space pregrasp or contact endpoint lacks reserved native IK')
            return candidate,dict(goal_ik=goal_checks,receiver_native_ik=plan,
                                  contact_endpoint_ik=endpoint,pregrasp_position=(grasp_p-.080*direction).tolist(),
                                  pregrasp_rotation=grasp_q,
                                  projected_body_pose=np.asarray(body).tolist(),future_proposal_only=True)
        except (ValueError,RuntimeError) as error:
            failures.append(dict(candidate=candidate,refusal=str(error)))
    raise c.PlanningRefusal('No jointly reachable receiver proposal at this common position: '+str(failures))


def _execute_joint_pre(ctx,state,position,rotation):
    arm=int(state['transfer']['receiver']);belief=_belief(ctx,state)
    plan=ctx.tool('puzzle.plan_joint_path',position=np.asarray(position).tolist(),rotation=rotation,
        arm_id=arm,min_joint_limit_margin_rad=.035)
    if not plan.get('feasible'):raise c.PlanningRefusal('Actual common receiver joint pregrasp IK refused')
    full=_qpath(plan,belief['current_robot_encoder_values'],arm)
    proof=_guard(state,full,arm,belief,receiver_contact=False)
    off=0 if arm==0 else 8;executed=[]
    for start in range(1,len(full),8):
        _belief(ctx,state)
        output=ctx.tool('robot.execute_trajectory',trajectory=dict(waypoints=[dict(positions=row[off:off+6].tolist())
            for row in full[start:start+8]]),arm_id=arm,max_steps_per_waypoint=2)
        if output.get('status')!='ok':raise RuntimeError('Common receiver pregrasp motor refused')
        executed.append(output)
    c.await_motion(ctx,position,rotation,arm,accuracy=.0025)
    actual=_belief(ctx,state);_guard(state,[actual['current_robot_encoder_values']],arm,actual)
    return dict(phase='receiver_common_joint_pregrasp',native_ik=plan,complete_geometry=proof,motor=executed)


def approach_receiver(ctx,state):
    state=deepcopy(state)
    if state['piece']!='cyan' or int(state['arm'])!=0:
        raise c.PlanningRefusal('Cyan common transfer expects the actual arm0 source owner')
    state['transfer']=dict(sender=0,receiver=1,candidates=[],execution=[],common_workspace_attempts=[],
        old_sender_source=deepcopy(state['parameters'].get('owned_held_belief_source')))
    transfer=state['transfer'];receiver=1;donor=0
    anchor=state.get('anchor_evidence') or state.get('support_anchor_evidence')
    if not anchor or int(anchor.get('arm',-1))!=receiver:
        raise c.PlanningRefusal('Original opposite-hand board brace provenance missing')
    tcp,q=c.hand(ctx,receiver)
    exit_spec=dict(arm=receiver,direction_world=[0.,0.,1.],maximum_distance_m=.060,source_anchor=anchor)
    transfer['execution'].append(_execute(ctx,state,receiver,[(tcp+[0.,0.,.060],q)],
        'withdraw_brace_before_donor_transfer',anchor_exit=exit_spec))
    transfer['execution'].append(_grip(ctx,state,receiver,.055))
    # Preserve the measured donor orientation. In particular no inspection
    # pitch/roll is introduced while retaining the original shallow source.
    tcp,donor_q=c.hand(ctx,donor)
    transfer['execution'].append(_execute(ctx,state,donor,[(tcp+[0.,0.,.030],donor_q)],
        'donor_vertical_raise_fixed_attitude'))
    family=_family();chosen=None;accepted=None;selection_belief=None
    offsets=state['parameters'].get('cyan_handoff_common_board_offsets',
        [[.120,.020,.220],[.120,.060,.220],[.160,.020,.220]])
    if not 2<=len(offsets)<=6:
        raise c.PlanningRefusal('A bounded 2..6 public common-position proposal set is required')
    for offset in offsets:
        belief=_belief(ctx,state)
        board=np.asarray(state['parameters']['historical_complete_rgb']['poses']['board'])
        Rb=Rotation.from_quat(board[3:]).as_matrix()
        body=np.asarray(belief['predicted_body_pose']);proposal=body.copy()
        proposal[:3]=board[:3]+Rb@np.asarray(offset,dtype=float)
        proposal[2]=max(proposal[2],body[2])
        record=dict(board_relative_offset_m=list(offset),fixed_donor_attitude=True)
        transfer['common_workspace_attempts'].append(record)
        try:
            candidate,preview=_future_receiver(ctx,state,proposal,board,family)
            record['receiver_future_ik']=preview
            tcp,_=c.hand(ctx,donor)
            target=tcp+(proposal[:3]-body[:3])
            # The move below performs a new full current-q path screen before
            # any motor command; the receiver preview is never treated as a
            # measured object source or an executed state.
            motion=_execute(ctx,state,donor,[(target,donor_q)],'donor_common_translation_fixed_attitude')
            transfer['execution'].append(motion);record['donor_motion']=motion
        except (ValueError,c.PlanningRefusal) as error:
            record['proposal_refusal']=str(error);continue
        current=_belief(ctx,state)
        observed=c.state(ctx,{'cyan':current['predicted_body_pose']})
        state=save_observation(state,observed);transfer=state['transfer']
        record['actual_common_observation']=g.rgb_record(observed)
        if 'cyan' not in observed.get('poses',{}):
            # The previous RGB was taken before the donor translated. The
            # stationary-world conditional-close API cannot use that old pose
            # at a new place. Continue to another screened common position.
            record['source_status']='UNKNOWN_continue_fixed_attitude_translation_for_real_RGB'
            continue
        admission=check_bounded_release('cyan',current,observed,state['parameters'])
        source=g._source(observed,'cyan')
        state['parameters']['owned_held_belief_source']=source
        transfer['preclose_source']=deepcopy(source)
        transfer['common_source_admission']=admission
        current=_belief(ctx,state)
        # Recompute both receiver IK and every geometry sample from the NEW
        # actual RGB/q16. A future-position preview cannot license this move.
        for candidate in family:
            transfer['candidate']=candidate
            try:
                candidate,preview=_future_receiver(ctx,state,np.asarray(current['predicted_body_pose']),board,[candidate])
                # Plan/screen/execute the free-space path exactly once. The
                # prior implementation performed the same full path screen
                # twice and a future close sweep before moving either hand.
                # Its shallow donor lost retention during that idle interval.
                # The actual close sweep remains mandatory at close_receiver.
                motion=_execute_joint_pre(ctx,state,preview['pregrasp_position'],preview['pregrasp_rotation'])
                transfer['candidates'].append(dict(proposal=candidate,new_actual_common_IK=preview,
                    single_pass_actual_pregrasp_motion=motion,selected=True))
                transfer['execution'].append(motion)
                chosen=candidate;accepted=observed;selection_belief=deepcopy(current);break
            except (ValueError,c.PlanningRefusal) as error:
                transfer['candidates'].append(dict(proposal=candidate,refusal=str(error)))
        if chosen is not None:break
    if chosen is None:
        raise c.PlanningRefusal('Screened common-position regrasp proposals exhausted; donor remains closed')
    belief=_belief(ctx,state)
    grasp_p,grasp_q=_world_grasp(belief['predicted_body_pose'],chosen)
    direction=Rotation.from_quat(c.qarr(grasp_q)).as_matrix()[:,2]
    # The receiver is already at its independently screened actual pregrasp.
    # Take a second real bounded source near the actual receiver-close epoch.
    belief=_belief(ctx,state)
    late=c.state(ctx,{'cyan':belief['predicted_body_pose']})
    state=save_observation(state,late);transfer=state['transfer']
    if 'cyan' in late.get('poses',{}):
        transfer['preclose_source_admission']=check_bounded_release('cyan',belief,late,state['parameters'])
        source=g._source(late,'cyan');state['parameters']['owned_held_belief_source']=source
        transfer['preclose_source']=deepcopy(source)
    belief=_belief(ctx,state)
    if 'cyan' in late.get('poses',{}):
        # A rigid CAD fit can choose an equivalent frame. Keep the selected
        # physical contact transform through that measured-frame change.
        old_R=Rotation.from_quat(np.asarray(selection_belief['predicted_body_pose'])[3:]).as_matrix()
        old_p=np.asarray(selection_belief['predicted_body_pose'])[:3]
        p_world=old_p+old_R@np.asarray(chosen['hand_in_body_position'])
        R_world=old_R@np.asarray(chosen['hand_in_body_rotation'])
        new_body=np.asarray(belief['predicted_body_pose']);new_R=Rotation.from_quat(new_body[3:]).as_matrix()
        mapped=deepcopy(chosen)
        mapped['hand_in_body_position']=(new_R.T@(p_world-new_body[:3])).tolist()
        mapped['hand_in_body_rotation']=(new_R.T@R_world).tolist()
        old_bead_world=old_p+old_R@model.local_beads('cyan')[chosen['bead_index']]
        new_centers=model.local_beads('cyan')@new_R.T+new_body[:3]
        distances=np.linalg.norm(new_centers-old_bead_world,axis=1)
        index=int(np.argmin(distances))
        if distances[index]>.006:
            raise c.PlanningRefusal('Late current RGB changed the named receiver bead beyond original6mm bound')
        mapped['bead_index']=index
        mapped['permitted_distal_bead_indices']=[index]
        mapped['actual_late_RGB_frame_mapping']=True
        chosen=mapped;transfer['candidate']=mapped
    grasp_p,grasp_q=_world_grasp(belief['predicted_body_pose'],chosen)
    transfer['execution'].append(_execute(ctx,state,receiver,[(grasp_p,grasp_q)],
                                          'receiver_common_approach',receiver_contact=True))
    transfer['receiver_actual_ready_proprio']=_belief(ctx,state)['current_robot_encoder_values']
    observed=c.state(ctx,{'cyan':_belief(ctx,state)['predicted_body_pose']})
    return _out('receiver_ready',state,observed,receiver=receiver,candidate=chosen,
        scope='Donor retained original attitude through lift/translation; actual common RGB rebound; independent approach CP precedes receiver close')
