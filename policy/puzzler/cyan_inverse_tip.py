"""RGB/FK/public-CAD shallow CYAN tip pinch and low vertical release.

The static grasp profile is a body-relative geometric transform. It is rebound
to current source RGB on every execution. The actual held transform is measured
again after lift, rather than treating this desired transform as a measurement.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from . import cyan_geometry as g,vision_controller as c,conditional_scene,model,visual_memory
from .finite_release_scene import finite_clearance
from .scene_reference import select_reference
from .gripper_geometry import hand_primitives,clearance
from .gripper_geometry import Primitive

_BOARD_QUERY=None


def held_board_clearance(body_center,body_rotation,scene,*,terminal=False):
 """All79 public components versus the complete held CAD sphere cover."""
 global _BOARD_QUERY
 from .public_board_clearance import PublicBoardQuery
 if _BOARD_QUERY is None:_BOARD_QUERY=PublicBoardQuery()
 local,radii=conditional_scene.local_sphere_cover('cyan');board=np.asarray(scene['board_for_planning']['pose']);rb=Rotation.from_quat(board[3:]).as_matrix();cloud=local@np.asarray(body_rotation).T+np.asarray(body_center);centers=(cloud-board[:3])@rb
 def query(indices,inflate):
  shapes=tuple(Primitive('sphere',np.zeros(3),np.eye(3),np.array([radii[i]+inflate]),'cyan_public_cover:'+str(i)) for i in indices)
  return _BOARD_QUERY.query_shapes(shapes,centers[indices],np.tile([0.,0.,0.,1.],(len(indices),1)))
 full=query(np.arange(len(local)),.006);board_margin=float(scene['board_for_planning'].get('uncertainty_m',0.))
 if not terminal:return dict(passed=bool(full['clearance_m']-board_margin>=.0005),envelope=full,board_margin_m=board_margin)
 bead=model.local_beads('cyan')[1];allow=np.linalg.norm(local-bead,axis=1)+radii<=model.RADIUS+1.e-8
 forbidden=query(np.flatnonzero(~allow),.006);nominal=query(np.flatnonzero(allow),0.)
 edge=min(full['groups'][name]['clearance_m'] for name in ('rim','hinge','lid_border'))-board_margin
 passed=bool(forbidden['clearance_m']-board_margin>=.0005 and nominal['clearance_m']>=-.003 and edge>=.0005)
 return dict(passed=passed,envelope=full,non_contact_cover=forbidden,allowed_low_bead1_nominal=nominal,edge_envelope_m=edge,board_margin_m=board_margin,allowed_board_groups=['floor','socket_post'],allowed_public_region='Only public lowest Cyan bead1 sphere cover along aligned final vertical descent')


def _groups(width):
 shapes=hand_primitives(float(width));fingers=tuple(s for s in shapes if '_lf_down' in s.name or '_rf_down' in s.name)
 names={s.name for s in fingers};return fingers,tuple(s for s in shapes if s.name not in names)


def source_contact_clearance(position,rotation,width,scene,parameters):
 """Selected public distal-bead contact with an RGB-rebound approach axis."""
 body=np.asarray(scene['measurement']['poses']['cyan']);br=Rotation.from_quat(body[3:]).as_matrix()
 spec=parameters['inverse_grasp_profile'];endpoint=body[:3]+br@np.asarray(spec['hand_in_body_position'])
 target=np.asarray(position);dz=float(target[2]-endpoint[2]);zone=bool(np.linalg.norm(target[:2]-endpoint[:2])<=.0025 and -.0025<=dz<=.015)
 local,radii=conditional_scene.local_sphere_cover('cyan');cloud=local@br.T+body[:3]
 fingers,forbidden=_groups(width)
 retained=finite_clearance(position,rotation,width,scene,exclude=('cyan',))
 forbidden_gap=clearance(position,rotation,width,cloud,radius=radii,primitives=forbidden)
 finger_gap=clearance(position,rotation,width,cloud,radius=radii,primitives=fingers)
 # Connector cover fully inside bead0 is the same permitted public sphere.
 selected_indices=parameters.get('source_contact_bead_indices',[int(spec['bead'])])
 permitted=np.zeros(len(local),dtype=bool)
 for selected_index in selected_indices:
  selected=model.local_beads('cyan')[int(selected_index)];permitted|=np.linalg.norm(local-selected,axis=1)+radii<=model.RADIUS+1.e-8
 other=clearance(position,rotation,width,cloud[~permitted],radius=radii[~permitted],primitives=fingers)
 passed=(retained['clearance_m']>=.0005 and forbidden_gap['clearance_m']>=.0005 and other['clearance_m']>=.0005 and finger_gap['clearance_m']>=(-.003 if zone else .0005))
 return dict(passed=bool(passed),source_contact_axis=dict(endpoint=endpoint.tolist(),direction_world=[0,0,1],source='Current RGB body × public body-relative TCP',maximum_contact_zone_m=.015,actual_encoder_xy_residual_bound_m=.0025),in_contact_zone=zone,retained=retained,non_distal_target=forbidden_gap,distal_target=finger_gap,distal_unselected_target=other,explicit_permitted_public_bead_indices=list(selected_indices),allowed_contact='Only explicitly named target bead regions against distal lf_down/rf_down; max3mm near computed endpoint; no nonfinger or non-target allowance')


def observe_and_plan(ctx,state):
 observed=c.state(ctx);scene=g.planning_scene(observed,state);p=state['parameters'];spec=p['inverse_grasp_profile']
 body=np.asarray(observed['poses']['cyan']);R=Rotation.from_quat(body[3:]).as_matrix();grasp=body[:3]+R@np.asarray(spec['hand_in_body_position']);hand_R=R@np.asarray(spec['hand_in_body_rotation']);q=c.qdict(Rotation.from_matrix(hand_R).as_quat());arm=0
 checks=[source_contact_clearance(grasp+[0,0,dz],hand_R,p['source_open_width'],scene,p) for dz in np.linspace(0,p['source_hover'],16)]
 if not all(x['passed'] for x in checks):raise c.PlanningRefusal('Inverse tip RGB-rebound source contact CAD refused: '+str(next(x for x in checks if not x['passed'])))
 iks=[ctx.tool('puzzle.plan_pose',position=(grasp+[0,0,dz]).tolist(),rotation=q,arm_id=arm) for dz in (0.,p['source_hover'],p['lift'])]
 if not all(x['position_error_m']<.002 and x['rotation_error_rad']<.03 and x['joint_limit_margin_rad']>=.035 for x in iks):raise c.PlanningRefusal('Inverse tip source native IK/joint reserve refused')
 center,target_R=g.goals('pyramid',p['pyramid_plan'])['cyan']
 plan=dict(body=body.tolist(),grasp=grasp.tolist(),rotation=q,index=int(spec['bead']),target_center=np.asarray(center).tolist(),target_R=np.asarray(target_R).tolist(),heading=spec['goal_heading'],lean=spec['goal_lean'],grasp_height_m=float(grasp[2]-body[2]),source_contact_checks=checks,source_native_ik=iks,source_grasp_only=True,target_release_validated=False,s=g.rgb_record(observed),inverse_grasp_profile=spec)
 parameters={**p,'historical_complete_rgb':g.rgb_record(observed)}
 return g._plain(dict(route='planned',state={**state,'arm':arm,'parameters':parameters,'plan':plan,'planning_scene':scene,'observation':g.rgb_record(observed),'memory':visual_memory.observe(state['memory'],observed,p)},evidence=dict(kind='modular_geometry_plan',timestamp=observed['timestamp'],frame_sha256=observed['frame_sha256'],public_relative_grasp_rebound=True)))


def grasp_and_bind(ctx,state):
 # The shared source module executes exactly state.plan.grasp/rotation, which
 # the preceding node rebuilt and checked using this RGB epoch.
 from .cyan_original_transport import grasp_and_bind as acquire
 out=acquire(ctx,state)
 if out['route']=='verified':
  st=out['state'];observed=st['observation']
  # Keep an authentic later packet, not an initial packet with a new stamp.
  try:scene=g.planning_scene(observed,st)
  except c.PlanningRefusal:pass
  else:
   st['planning_scene']=scene;st['parameters']['historical_complete_rgb']=g.rgb_record(observed)
 return out


def _release_scene(state,observed):
 p=state['parameters'];prior=p['historical_complete_rgb'];raw=observed['poses'];fresh_board='board' in raw
 age=float(observed['timestamp']-prior['timestamp'])
 reference=select_reference(observed,prior,now=observed['timestamp'],max_age_s=30.,board_limit_m=.002,refusal=c.PlanningRefusal)
 b=np.asarray(raw['board'] if fresh_board else prior['poses']['board']);br=Rotation.from_quat(b[3:]).as_matrix();margin=0. if fresh_board else .004
 public=g.goals('pyramid',p['pyramid_plan']);oc={}
 for name in model.CELLS:
  if name in p['conditional_prefix']:
   center,rr=public[name];poses=[np.r_[b[:3]+br@center,Rotation.from_matrix(br@rr).as_quat()].tolist()]
   if name in raw:poses.append(raw[name])
   elif name in prior['poses']:poses.append(prior['poses'][name])
   source='explicit_prefix5_public_goal4mm_union_observed_history';inflation=.004+margin
  elif name in raw:poses=[raw[name]];source='current_rgb_measurement';inflation=0.
  elif name in prior['poses']:poses=[prior['poses'][name]];source='explicit_historical_unplaced4mm';inflation=.004
  else:raise c.PlanningRefusal('No owned bounded CAD occupancy for '+name)
  oc[name]=dict(source=source,pose_hypotheses=poses,inflation_m=inflation,shape='public CAD sphere cover')
 return dict(target='cyan',reference=reference,measurement=g.rgb_record(observed),occupancy=oc,board_for_planning=dict(pose=b.tolist(),uncertainty_m=margin,source='current_RGB' if fresh_board else 'explicit_recent_RGB_persistence')),b,margin


def _execute_path(ctx,state,position,rotation,scene,width,*,belief=None,terminal=False,joint=False,path_fraction=1.):
 arm=state['arm'];plan=ctx.tool('puzzle.plan_joint_path' if joint else 'puzzle.plan_path',position=np.asarray(position).tolist(),rotation=rotation,arm_id=arm,min_joint_limit_margin_rad=.035)
 if not plan.get('feasible'):raise c.PlanningRefusal('Inverse tip continuous native IK path refused: '+str(plan.get('errors',[])[-1:]))
 if path_fraction!=1.:
  if not joint or terminal or belief is None or not 0.<path_fraction<1.:raise c.PlanningRefusal('Partial carry requires a bounded owned nonterminal joint path')
  if len(plan.get('path',()))!=len(plan.get('tcp_path',())) or len(plan['path'])<3:raise c.PlanningRefusal('Partial carry requires matching native FK path samples')
  count=len(plan['path']);last=max(1,min(count-2,int(np.ceil((count-1)*path_fraction))))
  plan={**plan,'path':plan['path'][:last+1],'tcp_path':plan['tcp_path'][:last+1],'executed_path_fraction':last/(count-1),'original_path_sample_count':count}
  position=np.asarray(plan['tcp_path'][-1]['position']);rotation=plan['tcp_path'][-1]['rotation']
 checks=[]
 if belief is not None:
  body=np.asarray(belief['predicted_body_pose']);tcp,q=c.hand(ctx,arm);hr=Rotation.from_quat(c.qarr(q)).as_matrix();rp=hr.T@(body[:3]-tcp);rR=hr.T@Rotation.from_quat(body[3:]).as_matrix();local,lr=conditional_scene.local_sphere_cover('cyan');centers,sizes=conditional_scene.spheres(scene,exclude=('cyan',));board=np.asarray(scene['board_for_planning']['pose']);rb=Rotation.from_quat(board[3:]).as_matrix();public=g.goals('pyramid',state['parameters']['pyramid_plan']);gc,_=public['cyan'];gR=np.asarray(state['plan']['target_R']);goal=board[:3]+rb@gc
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
   current=ctx.tool('puzzle.propagate_pyramid_held_belief',source=state['parameters']['owned_held_belief_source'],arm_id=arm)
   enc=current['current_robot_encoder_values'];off=0 if arm==0 else 8
   if float(enc[off+6]-enc[off+7])<.004:raise c.PlanningRefusal('Actual empty jaws invalidate inverse-tip held state')
  moved=ctx.tool('robot.execute_trajectory',trajectory=dict(waypoints=waypoints[index:index+8]),arm_id=arm,max_steps_per_waypoint=2)
  if moved.get('status')!='ok':raise RuntimeError('Inverse tip motor trajectory refused')
  executed.append(moved)
 c.settle_motion(ctx,position,rotation,arm,False,.0025)
 return dict(native_ik=plan,checks=checks,motor=executed[-1],executed_chunks=executed)


def _select_low_target(ctx,state,belief,scene):
 # Goal transforms and their strict insertion-axis guard must share the same
 # measured board reference. This is a separate planning reference, never a
 # current board pose pasted into an older RGB packet.
 state={**state,'parameters':{**state['parameters'],'current_board_for_inverse_planning':deepcopy(scene['board_for_planning'])}}
 _,choices,_=g.carry_target(ctx,state);arm=state['arm'];feasible=[]
 for choice in choices:
  rr=Rotation.from_quat(c.qarr(choice['rotation'])).as_matrix()
  for height in (.003,.006,.010,.012,.014):
   if height>state['parameters'].get('max_release_height',.010):continue
   pos=np.asarray(choice['target_tcp'])+[0,0,height];sweep=[finite_clearance(pos,rr,float(w),scene,exclude=('cyan',)) for w in np.linspace(0.,.055,12)]
   ik=ctx.tool('puzzle.plan_pose',position=pos.tolist(),rotation=choice['rotation'],arm_id=arm)
   if min(x['clearance_m'] for x in sweep)>=.0005 and ik['position_error_m']<.002 and ik['rotation_error_rad']<.03 and ik['joint_limit_margin_rad']>=.035 and choice['native_ik']['joint_limit_margin_rad']>=.035:
    feasible.append({**choice,'height':height,'release_tcp':pos.tolist(),'opening_sweep':sweep,'release_ik':ik});break
 if not feasible:raise c.PlanningRefusal('Actual postlift transform has no3–10mm full55mm-opening low release')
 selected=min(feasible,key=lambda x:(x['height'],-x['release_ik']['joint_limit_margin_rad']));selected['board_reference']=deepcopy(scene['board_for_planning'])
 return selected


def carry_and_release(ctx,state):
 p=state['parameters'];arm=state['arm']
 if state.get('held_status')!='CURRENT_RGB_VERIFIED' or 'owned_held_belief_source' not in p:raise c.PlanningRefusal('Inverse tip release requires actual fresh postlift RGB/q16 transform')
 from . import cyan_original_transport as tr
 belief=tr._belief(ctx,state);fresh=c.state(ctx,{'cyan':belief['predicted_body_pose']});scene,board,margin=_release_scene(state,fresh)
 # New accepted held RGB replaces the postlift epoch if available; UNKNOWN
 # stays a prediction from the previous owned source and is never filled.
 if 'cyan' in fresh['poses']:
  source=g._source(fresh,'cyan');p={**p,'owned_held_belief_source':source};state={**state,'parameters':p};belief=tr._belief(ctx,state)
 target=_select_low_target(ctx,state,belief,scene);state={**state,'planning_scene':scene,'plan':{**state['plan'],'target_R':target['relative_target_R']}}
 # Joint interpolation avoids the near-limit wrist passage of Cartesian
 # orientation Slerp. Both endpoint IK and every FK sample are recomputed
 # from this actual RGB/q16 attachment, never from saved joint arrays.
 width=float(belief['current_robot_encoder_values'][6]-belief['current_robot_encoder_values'][7])
 carry=_execute_path(ctx,state,target['hover'],target['rotation'],scene,width,belief=belief,joint=True)
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
 execution=dict(contract='inverse_tip_actual_rgb_fk_low_release_v1',execution_mark=mark,held_belief=belief,planning_scene={**scene,'measurement':g.rgb_record(preopen)},preopen_rgb=g.rgb_record(preopen),requested_release_tcp=requested.tolist(),requested_rotation=q,actual_preopen_tcp=actual.tolist(),actual_preopen_rotation=actual_q,release_height_m=target['height'],actual_preopen_width_m=actual_width,open_width_m=.055,open_result=opening,home_result=home,new_visual_release_receipt=receipt,receipt_refusal=reason,carry=carry,descent=recordmove,retreat=retreat,full_open_sweep=target['opening_sweep'],support_cleanup=dict(arm=support,open_result=opened,home_result=shome),source=p['owned_held_belief_source'])
 return g._plain(dict(route='executed',state={**state,'memory':memory,'release_execution':execution,'observation':g.rgb_record(after),'last_stage':'conditional_segment_release_execution','last_outcome':'executed'},evidence=dict(kind='conditional_segment_release_execution',timestamp=after['timestamp'],frame_sha256=after['frame_sha256'])))
