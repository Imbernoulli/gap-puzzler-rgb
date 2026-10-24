"""Receiver-arm motion with unchanged Cyan terminal support checks.

The held external checks below are copied from cyan_inverse_tip._execute_path;
robot material screening covers the identical dense public FK samples. Only
attachment dispatch, arm width and settling are adapted for honest handoff.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from . import cyan_geometry as g,cyan_original_transport as tr,vision_controller as c
from . import conditional_scene,model,visual_memory
from .cyan_inverse_tip import held_board_clearance
from .finite_release_scene import finite_clearance
from .cyan_bimanual_adapter import _qpath,_geometry

_MATERIAL=None

def _material_geometry():
 global _MATERIAL
 if _MATERIAL is None:
  from .public_bimanual_material_geometry import PublicBimanualMaterialGeometry
  _MATERIAL=PublicBimanualMaterialGeometry(device='cuda:0',minimum_clearance_m=.0005)
 return _MATERIAL


def require_board_margin(proof,scene,*,held=False):
 margin=float(scene.get('board_for_planning',{}).get('uncertainty_m',0.))
 if not np.isfinite(margin) or margin<0.:
  raise c.PlanningRefusal('Invalid conditional board uncertainty')
 groups=['robot_board79','moving_robot_table']+(['held_envelope_board79'] if held else [])
 for row in proof.get('rows',[]):
  for name in groups:
   if float(row['groups'][name]['clearance_m'])<.0005+margin:
    raise c.PlanningRefusal('Complete robot path does not retain conditional board margin: '+str((name,margin,row)))
 proof['caller_preserved_board_uncertainty_m']=margin
 return proof

def move_held(ctx,state,position,rotation,scene,width,*,belief=None,terminal=False,joint=False,path_fraction=1.,allow_actual_intermediate=False):
 if allow_actual_intermediate and (terminal or not joint or not 0.<path_fraction<1.):
  raise c.PlanningRefusal('Actual-intermediate feedback is limited to an explicit nonterminal partial joint carry')
 arm=state['arm'];plan=ctx.tool('puzzle.plan_joint_path' if joint else 'puzzle.plan_path',position=np.asarray(position).tolist(),rotation=rotation,arm_id=arm,min_joint_limit_margin_rad=.035)
 if not plan.get('feasible'):raise c.PlanningRefusal('Inverse tip continuous native IK path refused: '+str(plan.get('errors',[])[-1:]))
 if path_fraction!=1.:
  if not joint or terminal or belief is None or not 0.<path_fraction<1.:raise c.PlanningRefusal('Partial carry requires a bounded owned nonterminal joint path')
  if len(plan.get('path',()))!=len(plan.get('tcp_path',())) or len(plan['path'])<3:raise c.PlanningRefusal('Partial carry requires matching native FK path samples')
  count=len(plan['path']);last=max(1,min(count-2,int(np.ceil((count-1)*path_fraction))))
  plan={**plan,'path':plan['path'][:last+1],'tcp_path':plan['tcp_path'][:last+1],'executed_path_fraction':last/(count-1),'original_path_sample_count':count}
  position=np.asarray(plan['tcp_path'][-1]['position']);rotation=plan['tcp_path'][-1]['rotation']
 if belief is None or float(belief.get('uncertainty_m',float('inf')))> .006000001:
  raise c.PlanningRefusal('Receiver terminal helper preserves original6mm held bound; no unknown/unbounded source')
 geometry=_material_geometry()
 full=geometry.densify_q16_path(_qpath(plan,belief['current_robot_encoder_values'],arm))
 tcps=[];target_poses=[]
 rp0=np.asarray(belief['object_in_tcp_position']);rR0=np.asarray(belief['object_in_tcp_rotation'])
 for row in full:
  geometry.fk(row);pp,qq=geometry.robot.iks[geometry._arm(arm)].ee_pose(geometry.scratch);rr=Rotation.from_quat(qq).as_matrix()
  tcps.append(dict(position=np.asarray(pp).tolist(),rotation=c.qdict(qq)))
  target_poses.append(np.r_[pp+rr@rp0,Rotation.from_matrix(rr@rR0).as_quat()])
 off=0 if arm==0 else 8
 plan={**plan,'path':full[:,off:off+6].tolist(),'tcp_path':tcps}
 material=geometry.screen_robot_material_path(full,scene,target='cyan',target_body_poses=target_poses,
                                            finger_target_contract={arm:[state['receiver_delivery']['selected_receiver_bead']]})
 if not material.get('safe'):raise c.PlanningRefusal('Receiver robot material path refused: '+str(material))
 require_board_margin(material,scene)
 checks=[]
 if belief is not None:
  rp=np.asarray(belief['object_in_tcp_position']);rR=np.asarray(belief['object_in_tcp_rotation']);local,lr=conditional_scene.local_sphere_cover('cyan');centers,sizes=conditional_scene.spheres(scene,exclude=('cyan',));board=np.asarray(scene['board_for_planning']['pose']);rb=Rotation.from_quat(board[3:]).as_matrix();public=g.goals('pyramid',state['parameters']['pyramid_plan']);gc,_=public['cyan'];gR=np.asarray(state['plan']['target_R']);goal=board[:3]+rb@gc
  goal_cover=local@(rb@gR).T+goal;support_chunks=[];inflations=[];support_names=[]
  for name,item in scene['occupancy'].items():
   if name=='cyan':continue
   nl,nr=conditional_scene.local_sphere_cover(name);cc,rr=public[name];nominal=nl@(rb@rr).T+board[:3]+rb@cc
   cyan_beads=model.local_beads('cyan')@(rb@gR).T+goal;other_beads=model.local_beads(name)@(rb@rr).T+board[:3]+rb@cc
   adjacent=bool(np.any(np.abs(np.linalg.norm(cyan_beads[:,None]-other_beads[None],axis=2)-2*model.RADIUS)<1.e-6))
   for option in item['pose_hypotheses']:
    pose=np.asarray(option);actual=nl@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]
    # A contradicted hypothesis cannot inherit a public-goal support licence.
    goal_pose=np.r_[board[:3]+rb@cc,Rotation.from_matrix(rb@rr).as_quat()];observed_goal_displacement=float(visual_memory.shape_displacement(name,goal_pose,pose));compatibility=observed_goal_displacement<=.0040001
    # The public rigid object can have exact CAD symmetries. A connector's
    # conservative sphere-cover sampling need not share those symmetries, so
    # nearest equal-radius *sample* matching can invent >4mm displacement for
    # a measured object whose actual bead geometry moved only1mm. Align the
    # rigid CAD frame using an exact public symmetry, then map each SAME
    # actual CAD cover point continuously into the public goal frame.
    beads=model.local_beads(name);actual_R=Rotation.from_quat(pose[3:]).as_matrix();goal_R=rb@rr;best=None
    for turn in range(4):
     for flip in (False,True):
      symmetry=model.rotation(turn*np.pi/2)@(np.diag([1.,-1.,-1.]) if flip else np.eye(3))
      set_error=np.linalg.norm((beads@symmetry.T)[:,None]-beads[None],axis=2).min(axis=1).max()
      if set_error>1.e-7:continue
      alignment=float((Rotation.from_matrix(actual_R@symmetry).inv()*Rotation.from_matrix(goal_R)).magnitude())
      if best is None or alignment<best[0]:best=(alignment,symmetry)
    if best is None:raise c.PlanningRefusal('Public CAD has no identity symmetry')
    mapped=nl@(goal_R@best[1].T).T+board[:3]+rb@cc
    mapping_distance=np.linalg.norm(actual-mapped,axis=1)
    delta=goal_cover[:,None]-mapped[None];gap=np.linalg.norm(delta,axis=2)-lr[:,None]-nr[None]
    swept=delta.copy();swept[:,:,2]+=np.clip(-delta[:,:,2],.003,.036);swept_gap=np.linalg.norm(swept,axis=2)-lr[:,None]-nr[None]
    # Build the public swept-neighbor tube before evaluating this trajectory.
    # A compatible raw hypothesis may sit off its public goal by its measured
    # <=4mm displacement. Its neighbor tube includes precisely that observed
    # displacement; it does not remove the hypothesis or shrink any envelope.
    neighbor_tube=.006+float(item['inflation_m'])+.0005001+observed_goal_displacement
    allowed=(gap>=-1.e-6)&(swept_gap<=neighbor_tube)&(mapping_distance[None,:]<=.0040001)
    valid=bool(name in state['parameters']['conditional_prefix'] and adjacent and compatibility)
    support_chunks.append(allowed if valid else np.zeros_like(allowed,dtype=bool));inflations.extend([item['inflation_m']]*len(nl))
    if valid and allowed.any() and name not in support_names:support_names.append(name)
  support_mask=np.concatenate(support_chunks,axis=1);inflations=np.asarray(inflations)
 for sample in plan['tcp_path']:
  point=np.asarray(sample['position']);R=Rotation.from_quat(c.qarr(sample['rotation'])).as_matrix();hand=finite_clearance(point,R,width,scene,exclude=('cyan',));check=dict(hand=hand)
  if hand['clearance_m']<.0005:raise c.PlanningRefusal('Inverse tip path finite hand CAD refused: '+str(hand))
  if belief is not None:
   bp=point+R@rp;BR=R@rR;cloud=local@BR.T+bp;nom=np.linalg.norm(cloud[:,None]-centers[None],axis=2)-lr[:,None]-sizes[None]
   held=float((nom-.006).min());table=float((cloud[:,2]-lr-.006-(board[2]-.003)).min())
   in_support_zone=bool(terminal and np.linalg.norm((bp-goal)[:2])<=.001 and -.00025<bp[2]-goal[2]<.0365 and (Rotation.from_matrix(rb@gR).inv()*Rotation.from_matrix(BR)).magnitude()<.04)
   board_check=held_board_clearance(bp,BR,scene,terminal=in_support_zone);check['held_finite79_board']=board_check
   if not board_check['passed']:raise c.PlanningRefusal('Inverse tip held finite79 board guard refused: '+str(board_check))
   if in_support_zone:
    # Along this aligned final vertical descent support contact is intended.
    # Retain a physical public-CAD no-penetration test without the planning
    # uncertainty inflation; all hand/non-target constraints stay unchanged.
    physical=nom+inflations[None,:]
    contact_min=float(physical[support_mask].min()) if support_mask.any() else float('inf')
    free_min=float((nom-.006)[~support_mask].min())
    if contact_min<-.001 or free_min<.0005 or table<.0005:raise c.PlanningRefusal('Inverse tip support contact contract refused: '+str(dict(contact=contact_min,free=free_min,table=table)))
    check.update(held_envelope_m=held,intended_vertical_support_contact=True,named_supports=support_names,nominal_support_gap_m=contact_min,non_support_envelope_gap_m=free_min)
   elif min(held,table)<.0005:raise c.PlanningRefusal('Inverse tip held CAD envelope refused before intended support zone')
   else:check['held_envelope_m']=min(held,table)
  checks.append(check)
 waypoints=[]
 for a,b in zip(np.asarray(plan['path'])[:-1],np.asarray(plan['path'])[1:]):
  n=max(1,int(np.ceil(np.max(np.abs(b-a))/.012)));waypoints.extend(dict(positions=(a+(b-a)*u).tolist()) for u in np.linspace(0,1,n+1)[1:])
 executed=[]
 for index in range(0,len(waypoints),8):
  if belief is not None:
   # The registered propagator permanently revokes a loaded RGB lineage
   # when actual encoder jaws close empty. FK cannot keep carrying a fallen
   # object as an unexamined rigid prediction.
   current=tr._belief(ctx,state)
   enc=current['current_robot_encoder_values'];off=0 if arm==0 else 8
   if float(enc[off+6]-enc[off+7])<.004:raise c.PlanningRefusal('Actual empty jaws invalidate inverse-tip held state')
  moved=ctx.tool('robot.execute_trajectory',trajectory=dict(waypoints=waypoints[index:index+8]),arm_id=arm,max_steps_per_waypoint=2)
  if moved.get('status')!='ok':raise RuntimeError('Inverse tip motor trajectory refused')
  executed.append(moved)
 feedback=None
 try:c.await_motion(ctx,position,rotation,arm,accuracy=.0025)
 except c.MotorAccuracyError as error:
  if not allow_actual_intermediate or error.pe>.006 or error.re>.03:raise
  # An intermediate waypoint has no task postcondition of its own. Certify
  # this REAL stable state and require a fresh image/replan before the next
  # leg; never claim the unachieved waypoint or the terminal goal was reached.
  first=tr._belief(ctx,state);p0,q0=c.hand(ctx,arm)
  def screen(actual):
   if (not np.isfinite(actual.get('uncertainty_m',float('inf')))
       or float(actual['uncertainty_m'])>.006000001):
    raise c.PlanningRefusal('Actual intermediate must preserve the original finite6mm held uncertainty')
   body=np.asarray(actual['predicted_body_pose']);row=actual['current_robot_encoder_values']
   complete=_geometry().screen_q16_path([row],scene,moving_arm=arm,held_arm=arm,
    held_body_pose=body,object_in_tcp_position=actual['object_in_tcp_position'],
    object_in_tcp_rotation=actual['object_in_tcp_rotation'],target='cyan',held_envelope_m=.006,
    finger_target_contract={arm:[state['receiver_delivery']['selected_receiver_bead']]})
   if not complete.get('safe'):raise c.PlanningRefusal('Actual intermediate complete geometry refused: '+str(complete))
   require_board_margin(complete,scene,held=True)
   geometry=_material_geometry()
   mp=geometry.screen_robot_material_path([row],scene,target='cyan',target_body_poses=body,
    finger_target_contract={arm:[state['receiver_delivery']['selected_receiver_bead']]})
   if not mp.get('safe'):raise c.PlanningRefusal('Actual intermediate material geometry refused: '+str(mp))
   require_board_margin(mp,scene)
   tcp=np.asarray(actual['current_tcp']);H=Rotation.from_quat(tcp[3:]).as_matrix()
   finite=finite_clearance(tcp[:3],H,float(row[off+6]-row[off+7]),scene,exclude=('cyan',))
   if finite['clearance_m']<.0005:raise c.PlanningRefusal('Actual intermediate finite hand geometry refused')
   return dict(full_public_geometry=complete,robot_material=mp,finite_hand=finite)
  proof0=screen(first);ctx.tool('robot.wait_steps',steps=6)
  second=tr._belief(ctx,state);p1,q1=c.hand(ctx,arm)
  drift=float(np.linalg.norm(p1-p0));angle=float((Rotation.from_quat(c.qarr(q0)).inv()*Rotation.from_quat(c.qarr(q1))).magnitude())
  if drift>.0004 or angle>.015:raise c.PlanningRefusal('Actual intermediate is not stable within original measured-pose bounds')
  proof1=screen(second)
  feedback=dict(original_partial_waypoint_reached=False,terminal_goal_reached=False,
   original_position_residual_m=error.pe,original_rotation_residual_rad=error.re,
   maximum_local_position_residual_m=.006,maximum_local_rotation_residual_rad=.03,
   actual_drift_m=drift,actual_angular_drift_rad=angle,
   before_wait=proof0,after_wait=proof1,actual_q16=second['current_robot_encoder_values'],
   fresh_actual_RGB_required_before_replanning=True)
 return dict(native_ik=plan,checks=checks,motor=executed[-1],executed_chunks=executed,robot_material=material,
             original_terminal_support_contract_preserved=True,actual_intermediate_feedback=feedback)

