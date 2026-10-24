"""Verified mid20 Cyan leaf split into actual transport and release phases."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from . import cyan_inverse_tip as tip, cyan_geometry as g
from . import vision_controller as c,visual_memory,cyan_original_transport as tr
from .modular_controller import result,save_observation
from .release_admission import check_bounded_release
_execute_path=tip._execute_path
_release_scene=tip._release_scene
_select_low_target=tip._select_low_target
observe_and_plan=tip.observe_and_plan
grasp_and_bind=tip.grasp_and_bind

def _inspect_transport_rgb(ctx,state,old,fresh):
 """Use current ambiguous RGB hypotheses to propose a bounded viewpoint.

 Neither hypothesis is selected or placed into measured poses. Every proposed
 motion uses the original owned 6mm estimate and the complete native/CAD guard;
 only a later unique fit may supply the new RGB/q16 source.
 """
 from .cyan_observation_geometry import rank_bounded_observation_poses,score_observation_pose
 from .vision import calibration,rig_specs
 candidates=fresh.get('detections',{}).get('cyan',{}).get('supported_pose_candidates',[])
 if not 2<=len(candidates)<=4:raise c.PlanningRefusal('Cyan bounded inspection needs a finite current RGB ambiguity set')
 origin=np.asarray(old['current_tcp']);width=float(old['current_robot_encoder_values'][6]-old['current_robot_encoder_values'][7])
 cameras=calibration(rig_specs());baseline=score_observation_pose(origin,width,candidates,cameras,origin)
 proposals=rank_bounded_observation_poses(origin,width,candidates,cameras)
 proposals=[row for row in proposals if row['score']>baseline['score']][:3]
 if not proposals:raise c.PlanningRefusal('Cyan current RGB ambiguity has no improved bounded CAD viewpoint')
 records=[]
 for proposal in proposals:
  point=np.asarray(proposal['position']);quat=np.asarray(proposal['rotation_xyzw'])
  angle=float((Rotation.from_quat(origin[3:]).inv()*Rotation.from_quat(quat)).magnitude())
  if np.linalg.norm(point-origin[:3])>.0250001 or angle>np.deg2rad(25.)+1.e-7:raise c.PlanningRefusal('Cyan inspection exceeds bounded25mm/25degree viewpoint')
  old=tr._belief(ctx,state);scene,_,_=_release_scene(state,fresh)
  width=float(old['current_robot_encoder_values'][6]-old['current_robot_encoder_values'][7])
  row=dict(proposal=deepcopy(proposal),baseline=deepcopy(baseline),
           previous_source=deepcopy(old['source']),held_body_envelope_m=.006)
  records.append(row)
  try:row['motion']=_execute_path(ctx,state,point,c.qdict(quat),scene,width,belief=old,joint=True)
  except c.PlanningRefusal as error:
   row['motion_refusal']=str(error);continue
  old=tr._belief(ctx,state);fresh=c.state(ctx,{'cyan':old['predicted_body_pose']})
  row['observation']=g.rgb_record(fresh);row['target_fit']=deepcopy(fresh.get('detections',{}).get('cyan',{}))
  if fresh.get('observation_source')!='rgb_shape_geometry' or fresh.get('is_current_rgb') is False or not fresh.get('frame_sha256') or not isinstance(fresh.get('timestamp'),(int,float)) or not np.isfinite(fresh['timestamp']) or fresh['timestamp']<old['current_timestamp']:raise c.PlanningRefusal('Cyan inspection requires an authentic current shape-RGB epoch')
  if 'cyan' in fresh.get('poses',{}):return old,fresh,records
 raise c.PlanningRefusal('Cyan bounded screened RGB inspection exhausted: '+str([row.get('motion_refusal',row.get('target_fit',{}).get('reason')) for row in records]))


def _current_transport_source(ctx,state,*,inspect_missing=False):
 """Rebind only an accepted current target RGB packet and its same-image q16."""
 old=tr._belief(ctx,state);fresh=c.state(ctx,{'cyan':old['predicted_body_pose']})
 if fresh.get('observation_source')!='rgb_shape_geometry' or fresh.get('is_current_rgb') is False or not fresh.get('frame_sha256') or not isinstance(fresh.get('timestamp'),(int,float)) or not np.isfinite(fresh['timestamp']) or fresh['timestamp']<old['current_timestamp']:raise c.PlanningRefusal('Cyan carry feedback requires an authentic current shape-RGB epoch')
 inspection=[]
 if 'cyan' not in fresh.get('poses',{}) and inspect_missing:old,fresh,inspection=_inspect_transport_rgb(ctx,state,old,fresh)
 if 'cyan' not in fresh.get('poses',{}):raise c.PlanningRefusal('Cyan carry feedback requires current accepted target RGB')
 # A new measurement must still agree with the preceding retained 6mm
 # estimate. Rebinding never erases a contradicted ownership bound.
 admission=check_bounded_release('cyan',old,fresh,state['parameters'])
 scene,board,margin=_release_scene(state,fresh)
 source=g._source(fresh,'cyan');source['source_role']='Actual accepted current transport RGB and synchronous robot q16'
 p={**state['parameters'],'owned_held_belief_source':source}
 updated=save_observation({**state,'parameters':p,'planning_scene':scene},fresh)
 belief=tr._belief(ctx,updated)
 record=dict(previous_source=deepcopy(old['source']),source=deepcopy(source),bounded_RGB_inspection=inspection,
             current_RGB_vs_previous_FK_displacement_m=admission['current_RGB_vs_FK_displacement_m'],
             retained_body_bound_m=admission['conditional_body_bound_m'],
             observation=g.rgb_record(fresh))
 return updated,fresh,scene,belief,record


def _current_hover_target(state,scene,belief,relative_target_R):
 """Recompute the same public hover from current RGB/q16 and scene board."""
 board=np.asarray(scene['board_for_planning']['pose']);br=Rotation.from_quat(board[3:]).as_matrix()
 center,_=g.goals('pyramid',state['parameters']['pyramid_plan'])['cyan']
 desired=br@np.asarray(relative_target_R);body=np.asarray(belief['predicted_body_pose'])
 tcp=np.asarray(belief['current_tcp']);delta=desired@Rotation.from_quat(body[3:]).as_matrix().T
 rotation=c.qdict(Rotation.from_matrix(delta@Rotation.from_quat(tcp[3:]).as_matrix()).as_quat())
 zero=board[:3]+br@center+delta@(tcp[:3]-body[:3])
 hover=zero+br@np.array([0.,0.,state['parameters']['transport']])
 desired_body=np.r_[board[:3]+br@(center+[0.,0.,state['parameters']['transport']]),Rotation.from_matrix(desired).as_quat()]
 error=visual_memory.shape_displacement('cyan',body,desired_body)
 return dict(relative_target_R=deepcopy(relative_target_R),target_tcp=zero.tolist(),hover=hover.tolist(),
             rotation=rotation,board_reference=deepcopy(scene['board_for_planning']),
             current_RGB_public_hover_error_m=float(error),release_revalidation_required=True)


def execute_transport(ctx,state):
 p=state['parameters'];arm=state['arm']
 if state.get('held_status')!='CURRENT_RGB_VERIFIED' or 'owned_held_belief_source' not in p:raise c.PlanningRefusal('Inverse tip release requires actual fresh postlift RGB/q16 transform')
 state,fresh,scene,belief,first_source=_current_transport_source(ctx,state)
 target=_select_low_target(ctx,state,belief,scene);relative=target['relative_target_R']
 state={**state,'planning_scene':scene,'plan':{**state['plan'],'target_R':relative}}
 width=float(belief['current_robot_encoder_values'][6]-belief['current_robot_encoder_values'][7])
 # Stop halfway through the newly solved joint path, while retaining every
 # original native FK/CAD sample guard. The remaining path is solved and
 # screened again after an actual RGB/q16 refresh at this intermediate pose.
 first_leg=_execute_path(ctx,state,target['hover'],target['rotation'],scene,width,belief=belief,joint=True,path_fraction=.5)
 state,middle,scene,belief,middle_source=_current_transport_source(ctx,state,inspect_missing=True)
 target=_current_hover_target(state,scene,belief,relative)
 width=float(belief['current_robot_encoder_values'][6]-belief['current_robot_encoder_values'][7])
 second_leg=_execute_path(ctx,state,target['hover'],target['rotation'],scene,width,belief=belief,joint=True)
 feedback=[];corrections=[]
 # A 3mm observed endpoint target leaves room beneath the unchanged 6mm
 # independent transport postcondition. Movement limits bound this feedback;
 # they do not enlarge the held-body or obstacle uncertainty envelopes.
 for attempt in range(3):
  ctx.tool('robot.wait_steps',steps=6)
  state,observed,scene,belief,source_record=_current_transport_source(ctx,state)
  target=_current_hover_target(state,scene,belief,relative);feedback.append(dict(source=source_record,target=target))
  if target['current_RGB_public_hover_error_m']<=.003:break
  if attempt==2:raise c.PlanningRefusal('Cyan bounded current-RGB hover correction exhausted')
  tcp=np.asarray(belief['current_tcp']);step=float(np.linalg.norm(np.asarray(target['hover'])-tcp[:3]));angle=float((Rotation.from_quat(tcp[3:]).inv()*Rotation.from_quat(c.qarr(target['rotation']))).magnitude())
  if step>.012 or angle>.25:raise c.PlanningRefusal('Cyan current-RGB hover correction exceeds bounded12mm/0.25rad move')
  width=float(belief['current_robot_encoder_values'][6]-belief['current_robot_encoder_values'][7])
  correction=_execute_path(ctx,state,target['hover'],target['rotation'],scene,width,belief=belief,joint=True)
  corrections.append(dict(translation_m=step,rotation_rad=angle,motion=correction))
 carry=dict(contract='current_RGB_q16_segmented_cyan_carry_feedback_v1',
            segments=[first_leg,second_leg],source_refreshes=[first_source,middle_source],
            endpoint_feedback=feedback,corrections=corrections,
            observed_endpoint_target_m=.003,maximum_corrections=2,
            maximum_correction_translation_m=.012,maximum_correction_rotation_rad=.25,
            held_body_envelope_m=.006,scope='Current accepted RGB/q16 at the split and endpoint; each new native IK path retains the original finite79 board, hand, held6mm and prefix margins. No target pose is filled from FK and no receipt is minted.')
 state={**state,'cyan_selected_goal':target,'transport_record':carry,'observation':g.rgb_record(observed)}
 return result('carried',state,state['observation'],'transport',transport_record=carry)

def segment_conditional_release(ctx,state):
 p=state['parameters'];arm=state['arm'];target=state['cyan_selected_goal'];carry=state['transport_record']
 mark=ctx.tool('robot.execution_journal_cursor');belief=tr._belief(ctx,state);preopen=c.state(ctx,{'cyan':belief['predicted_body_pose']});scene,board,margin=_release_scene(state,preopen)
 if 'cyan' in preopen['poses']:
  source=g._source(preopen,'cyan');p={**p,'owned_held_belief_source':source};state={**state,'parameters':p};belief=tr._belief(ctx,state)
 # Even if target/board is hidden, the explicitly enlarged historical scene
 # must re-screen the final FULL opening sweep, not reuse a fresh-board plan.
 target=_select_low_target(ctx,state,belief,scene)
 state={**state,'plan':{**state['plan'],'target_R':target['relative_target_R']}}
 actual_width=float(belief['current_robot_encoder_values'][6]-belief['current_robot_encoder_values'][7])
 requested=np.asarray(target['release_tcp']);q=target['rotation'];recordmove=_execute_path(ctx,state,requested,q,scene,actual_width,belief=belief,terminal=True)
 ctx.tool('robot.wait_steps',steps=30);actual,actual_q=c.hand(ctx,arm)
 # Open in one coordinated ramp at the low seat so a bead is not suspended
 # for seconds while one jaw advances through tiny opening increments.
 opening=ctx.tool('robot.set_grip',arm_id=arm,width_m=.055,ramp_steps=18,settle_steps=15)
 if opening.get('clamped') is not False:raise RuntimeError('Inverse tip opening was clamped')
 retreat=_execute_path(ctx,state,actual+[0,0,.055],actual_q,scene,.055)
 home=ctx.tool('robot.go_home',arm_id=arm);support=state['anchor_evidence']['arm'];stcp,sq=c.hand(ctx,support);c.move(ctx,stcp+[0,0,.060],sq,support,cartesian=True,accuracy=.0025);opened=ctx.tool('robot.set_grip',arm_id=support,width_m=.075);shome=ctx.tool('robot.go_home',arm_id=support)
 ctx.tool('robot.wait_steps',steps=60);rb=Rotation.from_quat(board[3:]).as_matrix();center,rr=g.goals('pyramid',p['pyramid_plan'])['cyan'];hint={'cyan':np.r_[board[:3]+rb@center,Rotation.from_matrix(rb@rr).as_quat()].tolist()}
 before=c.state(ctx,hint);ctx.tool('robot.wait_steps',steps=60);after=c.state(ctx,hint);receipt=None;reason=None;memory=state['memory'];tcp,_=c.hand(ctx,arm)
 try:
  receipt=visual_memory.release_receipt('cyan',before,after,tcp,p);memory=visual_memory.mint_placement(visual_memory.observe(memory,after,p),receipt,after,p)
 except visual_memory.MemoryRefusal as error:reason=str(error)
 execution=dict(release_rgb_pair=[before,after],contract='inverse_tip_actual_rgb_fk_low_release_v1',execution_mark=mark,held_belief=belief,planning_scene={**scene,'measurement':g.rgb_record(preopen)},preopen_rgb=g.rgb_record(preopen),requested_release_tcp=requested.tolist(),requested_rotation=q,actual_preopen_tcp=actual.tolist(),actual_preopen_rotation=actual_q,release_height_m=target['height'],actual_preopen_width_m=actual_width,open_width_m=.055,open_result=opening,home_result=home,new_visual_release_receipt=receipt,receipt_refusal=reason,carry=carry,descent=recordmove,retreat=retreat,full_open_sweep=target['opening_sweep'],support_cleanup=dict(arm=support,open_result=opened,home_result=shome),source=p['owned_held_belief_source'])
 return g._plain(dict(route='executed',state={**state,'memory':memory,'release_execution':execution,'observation':g.rgb_record(after),'last_stage':'conditional_segment_release_execution','last_outcome':'executed'},evidence=dict(kind='conditional_segment_release_execution',timestamp=after['timestamp'],frame_sha256=after['frame_sha256'])))
