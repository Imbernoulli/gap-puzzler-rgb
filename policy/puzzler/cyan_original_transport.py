"""Original-order CYAN bounded active RGB inspection and staged carry.
All targets are computed from current RGB, current robot FK and public profiles.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from . import cyan_geometry as g,vision_controller as c,conditional_scene,model
from .finite_release_scene import finite_clearance
from .cyan_finite_height import clear_release_height

def _belief(ctx,state):
 p=state['parameters'];arm=state['arm']
 if 'owned_held_belief_source' in p:return ctx.tool('puzzle.propagate_pyramid_held_belief',source=p['owned_held_belief_source'],arm_id=arm)
 return ctx.tool('puzzle.propagate_conditional_attachment',attachment_id=p['conditional_attachment']['attachment_id'],arm_id=arm)

def _move(ctx,state,position,rotation,belief,*,conditional=False):
 arm=state['arm'];p=state['parameters'];plan=ctx.tool('puzzle.plan_path',position=list(map(float,position)),rotation=rotation,arm_id=arm,min_joint_limit_margin_rad=.035)
 if not plan.get('feasible') or len(plan.get('tcp_path',()))!=len(plan.get('path',())):raise c.PlanningRefusal('Original-order staged continuous native IK refused')
 scene=deepcopy(state['planning_scene'])
 for n,entry in scene['occupancy'].items():
  if n!='cyan':entry['inflation_m']=max(.004,entry['inflation_m']);entry['source']+='_historical4mm_during_bounded_grasp'
 centers,sizes=conditional_scene.spheres(scene,exclude=('cyan',));board=np.asarray(p['historical_complete_rgb']['poses']['board']);body=np.asarray(belief['predicted_body_pose']);tcp,q=c.hand(ctx,arm);R0=Rotation.from_quat(c.qarr(q)).as_matrix();rp=R0.T@(body[:3]-tcp);rR=R0.T@Rotation.from_quat(body[3:]).as_matrix();local,lr=conditional_scene.local_sphere_cover('cyan')
 q16=belief['current_robot_encoder_values'];offset=0 if arm==0 else 8;width=float(q16[offset+6]-q16[offset+7]);uncertainty=.008 if conditional else .006;checks=[]
 for sample in plan['tcp_path']:
  pos=np.asarray(sample['position']);R=Rotation.from_quat(c.qarr(sample['rotation'])).as_matrix();hand=finite_clearance(pos,R,width,scene,exclude=('cyan',),board_margin=.004);pts=local@(R@rR).T+pos+R@rp
  ds=np.linalg.norm(pts[:,None]-centers[None],axis=2)-lr[:,None]-sizes[None]-uncertainty
  held=min(float(ds.min()),float((pts[:,2]-lr-uncertainty-(board[2]-.003+.004)).min()))
  checks.append(dict(hand_clearance_m=hand['clearance_m'],held_cad_clearance_m=held,held_uncertainty_m=uncertainty))
 if min(min(z['hand_clearance_m'],z['held_cad_clearance_m']) for z in checks)<.002:raise c.PlanningRefusal('Original-order fresh-held staged carry CAD envelope refused: '+str(min(checks,key=lambda z:min(z['hand_clearance_m'],z['held_cad_clearance_m']))))
 waypoints=[]
 for a,b in zip(np.asarray(plan['path'])[:-1],np.asarray(plan['path'])[1:]):
  n=max(1,int(np.ceil(np.max(np.abs(b-a))/.018)));waypoints.extend(dict(positions=(a+(b-a)*u).tolist()) for u in np.linspace(0,1,n+1)[1:])
 result=ctx.tool('robot.execute_trajectory',trajectory=dict(waypoints=waypoints),arm_id=arm,max_steps_per_waypoint=2)
 if result.get('status')!='ok':raise RuntimeError('Original-order staged carry motor refused')
 c.settle_motion(ctx,position,rotation,arm,False,.0025)
 return dict(native_ik=plan,cad_sample_checks=checks,native_motor_result=result,held_belief=belief,policy_inputs='Current robot FK, current owned RGB held pose or explicitly UNKNOWN conditional attachment; public board-relative path profile.')


def refresh_owned_scene(out):
 if out.get('route')!='verified':return out
 state=out['state'];obs=state['observation'];prefix=set(state['parameters']['conditional_prefix'])|set(state.get('executed_placements',()))
 required={'board',*(set(model.CELLS)-prefix)}
 if required-set(obs.get('poses',{})):return out
 p={**state['parameters'],'historical_complete_rgb':g.rgb_record(obs)}
 refreshed={**state,'parameters':p}
 refreshed['planning_scene']=g.planning_scene(obs,refreshed)
 refreshed['fresh_held_scene_handoff']={'timestamp':obs['timestamp'],'frame_sha256':obs['frame_sha256'],'measured_names':sorted(obs['poses']),'required_names':sorted(required),'prior_refresh_source':'Actual accepted postlift RGB only; no synthesized missing pose','unchanged_historical_max_age_s':p.get('historical_max_age_s',30.)}
 return {**out,'state':refreshed}

def grasp_and_bind(ctx,state):
 out=g.grasp_and_bind(ctx,state)
 if out['route']!='conditional':return refresh_owned_scene(out)
 state=out['state'];records=[]
 # Conditional predictions may propose a safe viewpoint, but cannot earn a
 # held source. Only a new accepted RGB packet below can create that source.
 for rise,yaw in ((.025,0.),(0.,15.),(0.,-30.),(.025,15.)):
  belief=_belief(ctx,state);body=np.asarray(belief['predicted_body_pose']);tcp,q=c.hand(ctx,state['arm']);R=Rotation.from_quat(c.qarr(q)).as_matrix();delta=Rotation.from_euler('z',yaw,degrees=True).as_matrix();new_body=body[:3]+[0,0,rise];target=new_body+delta@(tcp-body[:3]);rotation=c.qdict(Rotation.from_matrix(delta@R).as_quat())
  try:
   moved=_move(ctx,state,target,rotation,belief,conditional=True)
  except c.PlanningRefusal as error:
   records.append(dict(rise_m=rise,yaw_deg=yaw,refused=str(error)));continue
  current=_belief(ctx,state);observed=c.state(ctx,{'cyan':current['predicted_body_pose']});records.append(dict(rise_m=rise,yaw_deg=yaw,motion=moved,observation=g.rgb_record(observed)))
  if 'cyan' not in observed['poses']:continue
  now,_=c.hand(ctx,state['arm']);pose=observed['poses']['cyan'];lift=float(pose[2]-state['plan']['body'][2]);gap=c.part_hand_gap('cyan',pose,now)
  if lift<.025 or gap>.045:continue
  source=g._source(observed,'cyan');memory=g.visual_memory.observe(state['memory'],observed,state['parameters']);evidence=dict(kind='modular_grasp',timestamp=observed['timestamp'],frame_sha256=observed['frame_sha256'],rise_m=lift,hand_bead_gap_m=gap,source=source,bounded_current_rgb_inspections=records)
  return refresh_owned_scene(g._plain(dict(route='verified',state={**state,'held_status':'CURRENT_RGB_VERIFIED','memory':memory,'observation':g.rgb_record(observed),'parameters':{**state['parameters'],'owned_held_belief_source':source},'bounded_current_rgb_inspections':records},evidence=evidence)))
 return g._plain(dict(route='conditional',state={**state,'bounded_current_rgb_inspections':records},evidence=out['evidence']))

def carry_target(ctx,state):
 if state['held_status']!='CURRENT_RGB_VERIFIED' or 'owned_held_belief_source' not in state['parameters']:raise c.PlanningRefusal('Original-order native requires actual fresh postlift RGB before transport')
 _,choices,belief=g.carry_target(ctx,state);p=state['parameters'];arm=state['arm'];feasible=[]
 for row in choices:
  try:
   h=clear_release_height('cyan',state['planning_scene']['measurement']['poses'],row['target_tcp'],row['rotation'],p,planning_scene=state['planning_scene']);check=ctx.tool('puzzle.plan_pose',position=(np.asarray(row['target_tcp'])+[0,0,h]).tolist(),rotation=row['rotation'],arm_id=arm)
   row.update(actual_held_release_height_m=h,actual_held_release_ik=check)
   hover=row['native_ik'];ok=all(r['position_error_m']<.002 and r['rotation_error_rad']<.03 and r['joint_limit_margin_rad']>=.035 for r in (hover,check))
   if ok:feasible.append(row)
  except Exception as error:row['actual_held_release_refusal']=str(error)
 if not feasible:raise c.PlanningRefusal('Actual owned held transform lacks original-order finite-CAD release and reserve')
 feasible.sort(key=lambda r:(r['actual_held_release_height_m'],-min(r['native_ik']['joint_limit_margin_rad'],r['actual_held_release_ik']['joint_limit_margin_rad'])))
 return feasible[0],choices,belief

def move_held_screened(ctx,state,target,belief):
 p=state['parameters'];arm=state['arm'];board=np.asarray(p['historical_complete_rgb']['poses']['board']);Rb=Rotation.from_quat(board[3:]).as_matrix();body=np.asarray(belief['predicted_body_pose']);tcp,q=c.hand(ctx,arm);R=Rotation.from_quat(c.qarr(q)).as_matrix();BR=Rotation.from_quat(body[3:]).as_matrix();hand_in_body=BR.T@(tcp-body[:3]);hand_R_in_body=BR.T@R
 public_center,public_R=g.goals('pyramid',p['pyramid_plan'])['cyan'];target_body_R=Rb@np.asarray(target['relative_target_R']);high_z=board[2]+p.get('carry_body_height_board_m',.187);stage=board[:3]+Rb@np.asarray(p.get('carry_stage_board_m',[.133,.200,.187]));highsource=body[:3].copy();highsource[2]=max(high_z,body[2]);stages=[('source_raise',highsource,BR),('outside_translate_rotate',stage,target_body_R),('goal_hover',board[:3]+Rb@public_center+[0,0,p['transport']],target_body_R)];records=[]
 for phase,bp,rr in stages:
  belief=_belief(ctx,state);position=bp+rr@hand_in_body;rotation=c.qdict(Rotation.from_matrix(rr@hand_R_in_body).as_quat());record=_move(ctx,state,position,rotation,belief);record['phase']=phase;records.append(record)
 return dict(contract='original_order_actual_rgb_fk_staged_carry_v1',stages=records,selected_goal=target,source=p['owned_held_belief_source'],sample_clearance_scope='Native FK samples retain 4mm historical obstacle union and 6mm actual-owned held envelope; all finite board CAD. Intended final support contact is handled separately by strict release contract.')
