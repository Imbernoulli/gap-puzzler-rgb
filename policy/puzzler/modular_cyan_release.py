"""Current held RGB/FK/CAD selection and feedback progressive CYAN release."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from . import vision_controller as c,visual_memory
from .vision_stages import _plain,_observation,StageRefusal
from .modular_controller import save_observation,scene_belief
from .public_board_clearance import PublicBoardQuery
from .progressive_release_feedback import next_release_step
from .modular_cyan_release_geometry import target_candidates,execute_screened_path,robot


def _belief(ctx,state):
    p=state['parameters'];arm=state['arm']
    return (ctx.tool('puzzle.propagate_conditional_attachment',attachment_id=p['conditional_attachment_id'],arm_id=arm)if p.get('conditional_attachment_id')else ctx.tool('puzzle.propagate_pyramid_held_belief',source=p['owned_held_belief_source'],arm_id=arm))


def segment_conditional_release(ctx,state):
    if state['piece']!='cyan':raise StageRefusal('CYAN release module target mismatch')
    arm=state['arm'];belief=_belief(ctx,state)
    observed=c.state(ctx,{'cyan':belief['predicted_body_pose']});state=save_observation(state,observed)
    # A real postcarry RGB measurement is preferred over the dated contact
    # assumption; the observer's raw support/alias gates remain unchanged.
    if 'cyan'in observed['poses']:
        packet=observed['robot_geometry_provenance'];source=dict(mode='pyramid',piece='cyan',pose=observed['poses']['cyan'],timestamp=observed['timestamp'],frame_sha256=observed['frame_sha256'],observation_source='rgb_shape_geometry',robot_encoder_values=packet['robot_encoder_values'],robot_encoder_sha256=packet['robot_encoder_sha256'],source_step=packet['step'],source_role='Actual accepted postcarry RGB and synchronous robot q16')
        p={**state['parameters'],'owned_held_belief_source':source};p.pop('conditional_attachment_id',None);state={**state,'parameters':p};belief=_belief(ctx,state)
    scene=scene_belief(state,observed);board=np.asarray(scene['board_for_planning']['pose']);state={**state,'belief_ledger':scene['belief_ledger']}
    local={**state,'release_held_belief':belief,'release_held_uncertainty_m':belief['uncertainty_m']};query=PublicBoardQuery();actual=robot(ctx,arm)
    candidates=target_candidates(local,scene,board,actual,query);choice=None;refusals=[]
    for candidate in candidates:
        try:
            approach=execute_screened_path(ctx,local,scene,board,query,candidate['hover'],candidate['rotation_q'])
        except c.PlanningRefusal as error:refusals.append(str(error));continue
        choice=candidate;break
    if choice is None:raise c.PlanningRefusal('No current-held CYAN geometric release approach: '+str(refusals))
    descent=execute_screened_path(ctx,local,scene,board,query,choice['endpoint_tcp'],choice['rotation_q'])
    actual=robot(ctx,arm);origin=np.asarray(actual['tcp'][:3]);mark=ctx.tool('robot.execution_journal_cursor');steps=[]
    for _ in range(180):
        actual=robot(ctx,arm);width=actual['actual_jaw_width_m'];rotation=Rotation.from_quat(actual['tcp'][3:]).as_matrix()
        step=next_release_step(actual_position=actual['tcp'][:3],actual_rotation=rotation,actual_width=width,release_origin=origin,retreat_vector=[0,0,.060],final_width=.080,min_release_width=choice['minimum_release_width_m'],scene=scene,piece='cyan',board=board,proprio_provenance=actual,board_query=query,move_step=.003,width_step=.002)
        row=dict(before=actual,suggestion=step);steps.append(row)
        if step['status']=='completed':break
        if step['status']!='command':raise c.PlanningRefusal('Actual CYAN open/withdraw feedback refused: '+str(step))
        command=step['command']
        if step['operation']=='open':row['motor_result']=ctx.tool('robot.set_grip',arm_id=arm,width_m=command['width_m'],ramp_steps=6,settle_steps=5)
        else:row['motor_result']=execute_screened_path(ctx,local,scene,board,query,command['position'],c.qdict(Rotation.from_matrix(np.asarray(command['rotation'])).as_quat()),held=False)
        after=robot(ctx,arm);row['after']=after
        if step['operation']=='open'and after['actual_jaw_width_m']<.0795 and after['actual_jaw_width_m']-width<.0001:raise c.PlanningRefusal('Actual CYAN jaw did not progress before final-width tolerance')
    else:raise c.PlanningRefusal('Bounded CYAN feedback iteration limit')
    c.grip(ctx,arm,.080);home=ctx.tool('robot.go_home',arm_id=arm)
    support_release=None
    if state.get('support_anchor_evidence'):
        from .modular_cyan_anchor import release_support_anchor
        support_release=release_support_anchor(ctx,state)
    ctx.tool('robot.wait_steps',steps=60);first=c.state(ctx);ctx.tool('robot.wait_steps',steps=60);second=c.state(ctx)
    receipt=None;receipt_refusal=None;memory=state['memory'];tcp,_=c.hand(ctx,arm)
    try:
        receipt=visual_memory.release_receipt('cyan',first,second,tcp,state['parameters']);memory=visual_memory.mint_placement(visual_memory.observe(memory,second,state['parameters']),receipt,second,state['parameters'])
    except visual_memory.MemoryRefusal as error:receipt_refusal=str(error)
    packets=[{k:deepcopy(row[k])for k in ['poses','timestamp','frame_sha256','observation_source','assessment','robot_geometry_provenance']if k in row}for row in[first,second]]
    record=dict(contract='current_RGB_FK_CAD_progressive_cyan_release_v1',execution_mark=mark,held_belief_before_open=belief,planning_scene=scene,target_candidates=candidates,selected_target=choice,approach=approach,descent=descent,actual_release_origin=origin.tolist(),feedback_steps=steps,home_result=home,support_release=support_release,release_rgb_pair=packets,new_visual_release_receipt=receipt,receipt_refusal=receipt_refusal,target_nonfinger_guard_retained=False,rigid_attachment_after_first_open='EXPIRED; no post-open rigid body measurement inferred',scope='Actual proprioception feedback and sampled retained non-target/table/board CAD; no target-housing or whole-arm separation certificate. Independent placement/prefix CP standards unchanged.')
    return _plain(dict(route='executed',state={**state,'memory':memory,'release_execution':record,'observation':_observation(second)},evidence=dict(kind='release',timestamp=second['timestamp'],frame_sha256=second['frame_sha256'])))
