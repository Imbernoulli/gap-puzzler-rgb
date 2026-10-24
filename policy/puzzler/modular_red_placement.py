"""Live RED held-to-goal modules. No source regrasp or recorded motion input."""
from copy import deepcopy
import itertools,json
import numpy as np
from scipy.spatial.transform import Rotation
from . import model,conditional_scene,visual_memory,vision_controller as c
from .modular_controller import ModularState,ModularResult,save_observation,scene_belief,result
from .modular_red_contact import encoder_epoch
from .red_source_geometry import target_nonfinger_clearance
from .public_board_clearance import PublicBoardQuery
from .gripper_geometry import clearance,scene_clearance
from .vision_observer import goals
from .vision_stages import _plain,StageRefusal

_BOARD=None
def _board_query():
 global _BOARD
 if _BOARD is None:_BOARD=PublicBoardQuery()
 return _BOARD

def proprio(ctx,arm=1):
 row=ctx.tool('robot.read_proprioception',arm_id=arm)
 if row['width_coordinates']!=([6,7]if arm==0 else[14,15]):raise StageRefusal('Complete actual mirrored jaw coordinates required')
 return row

def public_symmetries(name):
 local=model.local_beads(name);out=[]
 for order in itertools.permutations(range(3)):
  for signs in itertools.product((-1.,1.),repeat=3):
   rr=np.eye(3)[:,order]@np.diag(signs)
   if np.linalg.det(rr)<.999:continue
   d=np.linalg.norm((local@rr.T)[:,None]-local[None],axis=2)
   if np.max(np.min(d,axis=1))<1.e-7:out.append(rr)
 return out

def bind_current_held(ctx,state:ModularState)->ModularResult:
 """Authenticate new actual RGB/q16 after test-only initialization."""
 if set([*state.get('initialized_prefix',[]),*state['executed_placements']])!=set(model.CELLS)-{'red'}:raise StageRefusal('RED final module requires all11 named public predecessor conditions')
 old=state['source_contact_evidence']['postlift_rgb']['poses']['red']
 if state['parameters'].get('owned_held_belief_source'):
  old=ctx.tool('puzzle.propagate_pyramid_held_belief',source=state['parameters']['owned_held_belief_source'],arm_id=1)['predicted_body_pose']
 observed=c.state(ctx,{'red':old});epoch=encoder_epoch(observed)
 if not {'board','red'}.issubset(observed['poses']):raise StageRefusal('Restored RED needs newly accepted current board/RED RGB')
 current=proprio(ctx);pose=observed['poses']['red'];gap=c.part_hand_gap('red',pose,np.asarray(current['tcp'][:3]))
 source_z=state['source_contact_evidence']['original_source_rgb']['poses']['red'][2]
 if pose[2]-source_z<.025 or gap>.045:raise StageRefusal('Fresh RGB contradicts lifted near-hand RED')
 state=save_observation(state,observed);public_center,public_R=goals('pyramid',state['parameters']['pyramid_plan'])['red']
 source=dict(mode='pyramid',piece='red',pose=pose,timestamp=observed['timestamp'],frame_sha256=observed['frame_sha256'],observation_source=observed['observation_source'],robot_encoder_values=epoch['values'],robot_encoder_sha256=epoch['sha256'],source_step=epoch['step'],source_role='New actual restored-held RGB and its same-image q16; no time mixing')
 p={**state['parameters'],'owned_held_belief_source':source,'historical_complete_rgb':deepcopy(observed),'historical_rgb_records':deepcopy(state['rgb_history']),'historical_max_age_s':30.,'actual_executed_prefix':list(state['executed_placements']),'planning_prefix':[*state['initialized_prefix'],*state['executed_placements']],'given_prefix_names':list(state['initialized_prefix']),'transport':.030,'release_height':.003,'release_open_width':.090}
 # Keep the idle public-home hand open for the final task postcondition.
 p['open_idle_hand_after_release']=proprio(ctx,0)['width_m']<.045
 ctx.tool('robot.wait_steps',steps=6)
 state={**state,'parameters':p,'arm':1,'plan':dict(target_center=public_center.tolist(),target_R=public_R.tolist()),'prediction':pose}
 return result('verified',state,observed,'grasp',rise_m=pose[2]-source_z,hand_bead_gap_m=gap,fresh_source=source,test_initialization_only=True)

def _target_centers(position,rr,belief):
 local,radii=conditional_scene.local_sphere_cover('red')
 return local@(rr@np.asarray(belief['object_in_tcp_rotation'])).T+position+rr@np.asarray(belief['object_in_tcp_position']),radii

def _guard(pos,rr,width,scene,board,*,target_centers,target_radii,all_target=False):
 centers,radii=conditional_scene.spheres(scene,exclude=('red',))
 other=clearance(pos,rr,width,centers,radius=radii)
 fixed=scene_clearance(pos,rr,width,[],table_z=float(board[2]-.003),plane_z=float(board[2]-.003),board=board)
 finite=_board_query().query(board,pos,rr,width)
 # Intended finger/target grasp contact is the only exclusion during opening.
 # Wrist/camera housing is ALWAYS tested against the target's public CAD.
 target=(clearance if all_target else target_nonfinger_clearance)(pos,rr,width,target_centers,radius=target_radii+.006)
 guards=dict(non_target=other,table_bases_hinge=fixed,finite_board=finite,target_all_hand=target)if all_target else dict(non_target=other,table_bases_hinge=fixed,finite_board=finite,target_wrist_camera=target)
 key=min(guards,key=lambda k:guards[k]['clearance_m'])
 return dict(clearance_m=float(guards[key]['clearance_m']),witness=dict(component=key,detail=guards[key].get('witness')),components={k:float(v['clearance_m'])for k,v in guards.items()})

def choose_target(ctx,state,belief,scene,current,*,for_transport=False):
 board=np.asarray(scene['board_for_planning']['pose']);br=Rotation.from_quat(board[3:]).as_matrix();center,goal=goals('pyramid',state['parameters']['pyramid_plan'])['red'];target=board[:3]+br@center
 relative_p=np.asarray(belief['object_in_tcp_position']);relative_R=np.asarray(belief['object_in_tcp_rotation']);rows=[]
 for index,sym in enumerate(public_symmetries('red')):
  desired=br@goal@sym;hand_R=desired@relative_R.T;zero=target-hand_R@relative_p;q=c.qdict(Rotation.from_matrix(hand_R).as_quat());release=zero+[0,0,.003];hover=zero+[0,0,.030]
  target_centers,target_radii=_target_centers(release,hand_R,belief)
  sweep=[_guard(release,hand_R,float(w),scene,board,target_centers=target_centers,target_radii=target_radii)for w in np.linspace(current['width_m'],.090,19)]
  hover_guard=_guard(hover,hand_R,float(current['width_m']),scene,board,target_centers=target_centers,target_radii=target_radii)
  row=dict(symmetry_index=index,public_symmetry=sym.tolist(),target_R=(goal@sym).tolist(),target_center=center.tolist(),release_tcp=release.tolist(),hover_tcp=hover.tolist(),rotation=q,actual_width_m=current['width_m'],width_sweep_m=[current['width_m'],.090],release_height_m=.003,sweep_min_clearance_m=min(r['clearance_m']for r in sweep),sweep=sweep)
  if (row['sweep_min_clearance_m']<.0005 and not for_transport):
   row.update(feasible=False,refusal='drop3 actualjaw-to90 CAD sweep' if row['sweep_min_clearance_m']<.0005 else 'high-hover CAD guard',hover_guard=hover_guard);rows.append(row);continue
  checks=[] if for_transport else [ctx.tool('puzzle.plan_pose',position=p.tolist(),rotation=q,arm_id=1)for p in (hover,release,release+[0,0,.030])]
  row.update(native_IK=checks,feasible=True if for_transport else all(r['position_error_m']<.002 and r['rotation_error_rad']<.03 and r['joint_limit_margin_rad']>=.035 for r in checks));rows.append(row)
 feasible=[r for r in rows if r['feasible']]
 if not feasible:raise c.PlanningRefusal('RED actualheld publicgoal symmetry/drop3 preflight refused: '+json.dumps(_plain(rows),separators=(',',':')))
 selected=min(feasible,key=lambda r:(max(v['position_error_m']+.05*v['rotation_error_rad']for v in r['native_IK']) if r['native_IK'] else 0.))
 return selected,rows

def _named_obstacles(state,scene,board,final_rotation):
 """Public bead names authorize contact; current obstacles never create it."""
 br=Rotation.from_quat(board[3:]).as_matrix();public=goals('pyramid',state['parameters']['pyramid_plan']);red_center,red_R=state['plan']['target_center'],state['plan']['target_R'];red_nominal=model.local_beads('red')@(br@np.asarray(red_R)).T+board[:3]+br@np.asarray(red_center)
 centers=[];sizes=[];nominal_sizes=[];allowed=[];names=[]
 for name,item in scene['occupancy'].items():
  if name=='red':continue
  local,radii=conditional_scene.local_sphere_cover(name);center,rr=public[name];gp=board[:3]+br@center;gr=br@rr;nominal_beads=model.local_beads(name)@gr.T+gp;goalpose=np.r_[gp,Rotation.from_matrix(gr).as_quat()]
  map_only=(item['source']=='conditional_executed_goal_MAP_persistence_current_rgb'
      and name=='maroon' and item.get('conditional_MAP_public_goal_only') is True
      and len(item['pose_hypotheses'])==1)
  valid_source=(item['source']in('conditional_given_stage_prefix_goal_plus_raw','conditional_executed_goal_persistence_plus_current_raw')or map_only)and item['inflation_m']>=.004
  for hypothesis,pose in enumerate(item['pose_hypotheses']):
   pose=np.asarray(pose);actual=local@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]
   valid_shape=valid_source and visual_memory.shape_displacement(name,goalpose,pose)<=(1.e-9 if map_only else .0040001)
   for index,(point,radius)in enumerate(zip(actual,radii)):
    mask=np.zeros(len(conditional_scene.local_sphere_cover('red')[0]),bool)
    nominal_index=None
    if valid_shape and index<len(model.local_beads(name)):
     nominal_index=int(np.argmin(np.linalg.norm(nominal_beads-point,axis=1)))
     if np.linalg.norm(nominal_beads[nominal_index]-point)<=.0040001:
      distance=np.linalg.norm(red_nominal-nominal_beads[nominal_index],axis=1)
      mask[:len(red_nominal)]=np.abs(distance-2*model.RADIUS)<1.e-6
    centers.append(point);sizes.append(radius+item['inflation_m']);nominal_sizes.append(radius);allowed.append(mask);names.append(dict(piece=name,hypothesis=hypothesis,cover_index=index,public_bead_index=nominal_index,named_support_scope=bool(mask.any())))
 return np.asarray(centers),np.asarray(sizes),np.asarray(nominal_sizes),np.asarray(allowed).T,names

def _screened_path(ctx,state,endpoint,rotation,belief,scene,*,descent=False):
 current=proprio(ctx);board=np.asarray(scene['board_for_planning']['pose']);path=ctx.tool('puzzle.plan_path',position=np.asarray(endpoint).tolist(),rotation=rotation,arm_id=1,min_joint_limit_margin_rad=.035)
 if not path.get('feasible')or len(path.get('tcp_path',()))!=len(path.get('path',()))or len(path['path'])<2:raise c.PlanningRefusal('RED continuous currentFK Cartesian IK refused')
 local,radii=conditional_scene.local_sphere_cover('red');checks=[]
 final_R=Rotation.from_quat(c.qarr(rotation)).as_matrix();final_centers,_=_target_centers(np.asarray(endpoint),final_R,belief)
 # Only public nominal support pairs at the destination may enter contact in
 # the final12mm of a downward placement. Every other held collision remains.
 obstacles,sizes,nominal_sizes,support_pairs,support_names=_named_obstacles(state,scene,board,final_R)
 start=np.asarray(path['tcp_path'][0]['position']);end=np.asarray(endpoint)
 descending=bool(end[2]<start[2]-.001 and np.linalg.norm(end[:2]-start[:2])<=.001)
 for sample in path['tcp_path']:
  point=np.asarray(sample['position']);rr=Rotation.from_quat(c.qarr(sample['rotation'])).as_matrix();centers,rs=_target_centers(point,rr,belief)
  hand=_guard(point,rr,current['width_m'],scene,board,target_centers=centers,target_radii=rs)
  pair=np.linalg.norm(centers[:,None]-obstacles[None],axis=2)-rs[:,None]-sizes[None]-.006
  nominal=np.linalg.norm(centers[:,None]-obstacles[None],axis=2)-rs[:,None]-nominal_sizes[None]
  relative_height=point[2]-end[2];constant_orientation=(Rotation.from_matrix(final_R).inv()*Rotation.from_matrix(rr)).magnitude()<=.04
  terminal=bool(descent and descending and -.00025<=relative_height<=.012 and np.linalg.norm(point[:2]-end[:2])<=.001 and constant_orientation)
  allowed=support_pairs if terminal else np.zeros_like(support_pairs)
  bad=np.where(allowed,nominal<-.001,pair<.0005)
  row=dict(position=point.tolist(),hand=hand,held_min_envelope_clearance_m=float(pair.min()),held_min_nominal_clearance_m=float(nominal.min()),declared_public_support_pairs=int(allowed.sum()),public_support_contact_approach=bool(descent and allowed.any()))
  checks.append(row)
  if hand['clearance_m']<.0005 or bad.any():raise c.PlanningRefusal('RED sampled hand/held path refused: '+json.dumps(_plain(row),separators=(',',':')))
 waypoints=[]
 for aa,bb in zip(np.asarray(path['path'])[:-1],np.asarray(path['path'])[1:]):
  count=max(1,int(np.ceil(np.max(np.abs(bb-aa))/.016)))
  waypoints.extend(dict(positions=(aa+(bb-aa)*u).tolist())for u in np.linspace(0,1,count+1)[1:])
 segments=[];width_reference=current['width_m']
 for begin in range(0,len(waypoints),8):
  actual_before=proprio(ctx)
  if actual_before['width_m']<max(.030,width_reference-.0025):raise StageRefusal('Actual jaw suddenly closed during carry; possible lost attachment, no further blind motion authorized')
  width_reference=actual_before['width_m']
  moved=ctx.tool('robot.execute_trajectory',trajectory=dict(waypoints=waypoints[begin:begin+8]),arm_id=1,max_steps_per_waypoint=3)
  if moved.get('status')!='ok':raise StageRefusal('RED live motor trajectory failed')
  segments.append(dict(actual_proprio_before=actual_before,motor_result=moved))
 moved=dict(status='ok',segments=segments,feedback_scope='Complete actualq16/jaw between every8 motorwaypoints;16mrad joint increments, stop on2.5mm closure since previous chunk or actualwidth<30mm. Width is only a loss warning, never a measured object pose or grasp proof.')
 c.settle_motion(ctx,np.asarray(endpoint),rotation,1,False,.0025)
 return dict(contract='RED_live_actualq16_nativeIK_sampled_hand_held_path_v1',actual_proprio_before=current,native_path=path,checks=checks,motor_result=moved,named_public_supports=[r for r in support_names if r['named_support_scope']],public_support_scope='Named public nominal bead-neighbors at exactly25mm only; compatible prefix shape displacement<=4mm; downward constantorientation final12mm within1mm lateral corridor, nominal penetration<=1mm. All non-target hand/table/base/finiteboard and targethousing retained. Conditional6mm held/4mm prefix envelope remains for every non-support pair.'if descent else'All held pairs separated with6mm held/4mm prefix assumptions. Target wrist/camera and all other hand/table/board guards retained.')

def transport(ctx,state:ModularState)->ModularResult:
 p=state['parameters'];belief=ctx.tool('puzzle.propagate_pyramid_held_belief',source=p['owned_held_belief_source'],arm_id=1);scene=scene_belief(state,state['observation']);current=proprio(ctx)
 selected,candidates=choose_target(ctx,state,belief,scene,current)
 state={**state,'plan':{**state['plan'],'target_R':selected['target_R'],'target_center':selected['target_center'],'red_target_selection':selected,'red_target_candidates':candidates}}
 try:
  motion=_screened_path(ctx,state,selected['hover_tcp'],selected['rotation'],belief,scene)
 except c.PlanningRefusal as direct_error:
  # These via poses are regenerated from current FK and the public target.
  # Raise before rotating over the occupied pyramid; retain every clearance
  # and obtain fresh robot feedback before planning each subsequent segment.
  current=proprio(ctx);source=np.asarray(current['tcp'][:3]);source_q=c.qdict(np.asarray(current['tcp'][3:]));target=np.asarray(selected['hover_tcp']);legs=[];level_trials=[]
  if source[2]<target[2]+.030:
   high=max(source[2],target[2])+.060
   legs.append(_screened_path(ctx,state,[source[0],source[1],high],source_q,belief,scene))
  for extra in(.060,.040,.030,.025,.020,.015,.010):
   via=target+[0,0,extra];ik=ctx.tool('puzzle.plan_pose',position=via.tolist(),rotation=selected['rotation'],arm_id=1);trial=dict(extra_above_public_hover_m=extra,native_IK=ik)
   if not(ik['position_error_m']<.002 and ik['rotation_error_rad']<.03 and ik['joint_limit_margin_rad']>=.035):trial['refusal']='nativeIK';level_trials.append(trial);continue
   try:leg=_screened_path(ctx,state,via,selected['rotation'],belief,scene)
   except c.PlanningRefusal as error:trial['refusal']=str(error);level_trials.append(trial);continue
   legs.append(leg);trial['executed']=True;level_trials.append(trial);break
  else:raise c.PlanningRefusal('No currentFK/CAD/nativeIK feasible raised RED rotation layer: '+json.dumps(_plain(level_trials),separators=(',',':')))
  legs.append(_screened_path(ctx,state,target,selected['rotation'],belief,scene))
  motion=dict(contract='actualFK_raise_rotate_vertical_carry_v1',direct_path_refusal=str(direct_error),runtime_rotation_layer_trials=level_trials,segments=legs,scope='Runtime currentFK+publicCAD via poses, each new nativeIK path fully screened before execution; same held/prefix bounds retained.')
 actual=proprio(ctx);state['transport_record']=dict(source=belief,selected_target=selected,all_public_symmetry_candidates=candidates,actual_robot=actual,motion=motion)
 return result('carried',state,state['observation'],'transport',transport_record=state['transport_record'])

def release(ctx,state:ModularState)->ModularResult:
 p=state['parameters'];belief=ctx.tool('puzzle.propagate_pyramid_held_belief',source=p['owned_held_belief_source'],arm_id=1);scene=scene_belief(state,state['observation']);selected=state['plan']['red_target_selection'];board=np.asarray(scene['board_for_planning']['pose']);q=selected['rotation'];rr=Rotation.from_quat(c.qarr(q)).as_matrix()
 # Recompute the actualheld grasp offset from new FK, retaining the selected
 # public symmetry. No preflight worldTCP is replayed as a fixed template.
 br=Rotation.from_quat(board[3:]).as_matrix();desired=br@np.asarray(state['plan']['target_R']);target=board[:3]+br@np.asarray(state['plan']['target_center']);hand_R=desired@np.asarray(belief['object_in_tcp_rotation']).T
 requested=target-hand_R@np.asarray(belief['object_in_tcp_position'])+[0,0,.003];q=c.qdict(Rotation.from_matrix(hand_R).as_quat());rr=hand_R
 motion=_screened_path(ctx,state,requested,q,belief,scene,descent=True);actual=proprio(ctx);actual_R=Rotation.from_quat(actual['tcp'][3:]).as_matrix();centers,radii=_target_centers(np.asarray(actual['tcp'][:3]),actual_R,belief)
 sweep=[_guard(np.asarray(actual['tcp'][:3]),actual_R,float(w),scene,board,target_centers=centers,target_radii=radii)for w in np.linspace(actual['width_m'],.090,25)]
 if min(r['clearance_m']for r in sweep)<.0005:raise c.PlanningRefusal('Actual reached drop3 jaw-to90 sweep refused; no release authorized')
 opening=ctx.tool('robot.set_grip',arm_id=1,width_m=.090,ramp_steps=18,settle_steps=15)
 opened=proprio(ctx)
 if opened['width_m']<.0895:raise StageRefusal('Actual opening has not reached bounded90mm release width')
 origin=np.asarray(opened['tcp'][:3]);opened_R=Rotation.from_quat(opened['tcp'][3:]).as_matrix();opened_q=c.qdict(np.asarray(opened['tcp'][3:]));withdraw=origin+[0,0,.030]
 # After opening, all hand primitives are screened against released target,
 # including every finger. The target remains a conditional placed hypothesis.
 withdrawal=[_guard(point,opened_R,opened['width_m'],scene,board,target_centers=centers,target_radii=radii,all_target=True)for point in np.linspace(origin,withdraw,31)]
 if min(r['clearance_m']for r in withdrawal)<.0005:raise c.PlanningRefusal('Opened withdrawal intersects target or another retained CAD guard')
 c.move(ctx,withdraw,opened_q,1,cartesian=True,accuracy=.0025)
 if p.get('open_idle_hand_after_release'):c.grip(ctx,0,.055)
 home=ctx.tool('robot.go_home',arm_id=1);ctx.tool('robot.wait_steps',steps=10)
 hint={'red':[*target.tolist(),*Rotation.from_matrix(desired).as_quat().tolist()]};first=c.state(ctx,hint);ctx.tool('robot.wait_steps',steps=50);second=c.state(ctx,{'red':first['poses']['red']}if'red'in first['poses']else hint)
 memory=state['memory'];receipt=None;refusal=None
 try:
  receipt=visual_memory.release_receipt('red',first,second,np.asarray(proprio(ctx)['tcp'][:3]),p);memory=visual_memory.mint_placement(visual_memory.observe(memory,second,p),receipt,second,p)
 except visual_memory.MemoryRefusal as error:refusal=str(error)
 record=dict(contract='red_dynamic_actualheld_goal_release_v1',owned_source=p['owned_held_belief_source'],runtime_held_belief=belief,selected_public_symmetry=selected['public_symmetry'],release_height_m=.003,requested_release_tcp=requested.tolist(),actual_preopen=actual,actual_width_sweep=[actual['width_m'],.090],sweep=sweep,motion=motion,opening_result=opening,actual_postopen=opened,withdrawal=withdrawal,home_result=home,new_visual_release_receipt=receipt,receipt_refusal=refusal,release_rgb_pair=[first,second],target_3d_status='measured_in_new_release_pair'if'red'in first['poses']and'red'in second['poses']else'UNKNOWN',scope='All actions recomputed from new ownedRGB/actualq16/publicCAD/nativeIK. No recorded commands aftertestinit. Independent publicaxis physical CP still required.')
 state={**state,'memory':memory,'release_execution':record,'execution_tokens':{**state['execution_tokens'],'red':record},'executed_placements':[*state['executed_placements'],'red']}
 state=save_observation(save_observation(state,first),second)
 # Independent full12 CP requires2s stable; nativeRSS may terminate first.
 ctx.tool('robot.wait_steps',steps=45)
 return result('placed'if receipt else'uncertain',state,second,'release',release_execution=record)
