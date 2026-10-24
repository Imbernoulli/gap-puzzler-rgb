"""New RGB/FK/CAD release after exact test-only own-policy held recovery."""
import json
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from . import vision_controller as c,conditional_scene
from .vision_stages import StageState,StageResult,StageRefusal,_pause,_plain
from .vision_observer import goals
from .modular_maroon_geometry import profile
from .modular_maroon_scene import _scene,_owned,_output
from .public_board_clearance import PublicBoardQuery
from .progressive_release_feedback import next_release_step
from .gripper_geometry import clearance,hand_primitives,scene_clearance

def _gap(pos,R,width,scene,board,query):
 from .gripper_geometry import clearance,scene_clearance
 cs,rs=conditional_scene.spheres(scene,exclude=('maroon',));values=[clearance(pos,R,width,cs,radius=rs),scene_clearance(pos,R,width,[],table_z=board[2]-.003,plane_z=board[2]-.003,board=board),query.query(board,pos,R,width)]
 return min(values,key=lambda a:a['clearance_m'])

def _contact_gap(pos,R,width,scene,board,query,spec):
 """Closed-jaw support-contact screen for the named public support piece.

 Intended contact is restricted to distal finger primitives against one
 declared support body. All other occupancy, board, base, nonfinger and
 support-nonfinger clearances retain the normal positive 0.5 mm guard.
 """
 support=str(spec.get('support_piece','cyan'))
 names=tuple(scene.get('occupancy',{}))
 if support not in names: return dict(passed=False,reason='support body absent')
 other=conditional_scene.spheres(scene,exclude=('maroon',support))
 support_spheres=conditional_scene.spheres(scene,exclude=tuple(n for n in names if n!=support))
 centers,radii=other;sc, sr=support_spheres
 margin=float(spec.get('non_target_margin_m',.0005));penetration=float(spec.get('maximum_support_penetration_m',.003))
 all_hand=clearance(pos,R,width,centers,radius=radii)
 board_check=scene_clearance(pos,R,width,[],table_z=float(board[2]-.003),plane_z=float(board[2]-.003),board=board)
 fingers=tuple(s for s in hand_primitives(float(width)) if any(x in s.name.split(':')[0] for x in ('_lf_down','_rf_down')))
 finger_names={s.name for s in fingers}
 nonfingers=tuple(s for s in hand_primitives(float(width)) if s.name not in finger_names)
 sf=clearance(pos,R,width,sc,radius=sr,primitives=fingers)
 sn=clearance(pos,R,width,sc,radius=sr,primitives=nonfingers)
 query_gap=query.query(board,pos,R,width)
 non_target=min(all_hand['clearance_m'],board_check['clearance_m'],query_gap['clearance_m'])
 passed=non_target>=margin and sn['clearance_m']>=margin and sf['clearance_m']>=-penetration
 return dict(passed=bool(passed),clearance_m=float(min(non_target,sn['clearance_m'],sf['clearance_m'])),non_target_clearance_m=float(non_target),support_finger_clearance_m=float(sf['clearance_m']),support_nonfinger_clearance_m=float(sn['clearance_m']),support_piece=support,maximum_support_penetration_m=penetration,witnesses=dict(non_target=all_hand.get('witness'),board=board_check.get('witness'),support_finger=sf.get('witness'),support_nonfinger=sn.get('witness')))

def _rgb_ray_cad_support(observed,piece='maroon'):
 """Persist calibrated RGB ray/CAD support without depth/GT claims."""
 row=observed.get('detections',{}).get(piece,{})
 return dict(piece=piece,visible=row.get('visible'),views=row.get('views',[]),center_support=row.get('center_support',[]),median_ray_fit_residual_m=row.get('median_residual_m'),rms_ray_fit_residual_m=row.get('rms_residual_m'),independent_rgb_support=row.get('independent_rgb_support',{}),timestamp=observed.get('timestamp'),frame_sha256=observed.get('frame_sha256'),robot_geometry_provenance=observed.get('robot_geometry_provenance'),evidence_source='calibrated RGB camera-plane/ray fit against public CAD; no depth, GT or evaluator state')

def resume_release(ctx,state:StageState)->StageResult:
 p=state['parameters'];pr=profile();source=state['held_belief_source'];belief=ctx.tool('puzzle.propagate_pyramid_held_belief',source=source,arm_id=1)
 priors=deepcopy(state['latest_rgb']['poses']);priors['maroon']=belief['predicted_body_pose']
 seen=c.state(ctx,priors);history=[*state['historical_rgb_records'],deepcopy(seen)]
 if 'board'not in seen['poses']:
  # One camera-independent public-board-relative inspection move. Historical
  # RGB remains a conditional planning prior; it never fills raw current poses.
  inspection_scene,inspection_board=_scene(seen,history,p,'maroon',belief)
  ib=np.asarray(inspection_board);ir=Rotation.from_quat(ib[3:]).as_matrix();current=ctx.tool('robot.read_proprioception');ip=np.array(current['tcp'][:3]);iq=c.qdict(current['tcp'][3:]);iR=Rotation.from_quat(current['tcp'][3:]).as_matrix();endpoint=ib[:3]+ir@np.array([.02,-.19,.15]);query=PublicBoardQuery()
  tests=[_gap(x,iR,current['width_m'],inspection_scene,ib,query)for x in np.linspace(ip,endpoint,25)]
  if min(x['clearance_m']for x in tests)<.0005:raise StageRefusal('Inspection path publicCAD refused')
  check=ctx.tool('puzzle.plan_pose',position=endpoint.tolist(),rotation=iq,arm_id=1)
  if check['position_error_m']>.002 or check['rotation_error_rad']>.03 or check['joint_limit_margin_rad']<.035:raise StageRefusal('Inspection endpoint nativeIK refused')
  c.move(ctx,endpoint,iq,1,cartesian=True,accuracy=.0025)
  belief=ctx.tool('puzzle.propagate_pyramid_held_belief',source=source,arm_id=1);priors['maroon']=belief['predicted_body_pose'];seen=c.state(ctx,priors);history.append(deepcopy(seen))
  if 'board'not in seen['poses']:raise StageRefusal('Inspection currentboard remains UNKNOWN; no raw pose filled')
 if'maroon'in seen['poses']:
  source=_owned(seen);belief=ctx.tool('puzzle.propagate_pyramid_held_belief',source=source,arm_id=1)
 scene,board=_scene(seen,history,p,'maroon',belief);rb=Rotation.from_quat(board[3:]).as_matrix();center,goal=goals('pyramid',p['pyramid_plan'])['maroon'];bias=np.array(pr['release_translation_bias_board_m']);biasR=np.array(pr['release_rotation_bias_board']);relp=np.array(belief['object_in_tcp_position']);relR=np.array(belief['object_in_tcp_rotation']);query=PublicBoardQuery();actual=ctx.tool('robot.read_proprioception');w=actual['width_m'];screens=[];chosen_candidates=[];contact_spec=p.get('release_support_contact')
 # Evaluate both exact public goal aliases; the desired body translation is
 # board-relative and is not reversed by an equivalent body-frame alias.
 for frame in(np.eye(3),np.diag([-1.,-1.,1.])):
  desired=rb@biasR@goal@frame;R=desired@relR.T;zero=board[:3]+rb@(center+bias)-R@relp;q=c.qdict(Rotation.from_matrix(R).as_quat());orientation_screens=[]
  for extraZ in np.arange(0,.0161,.001):
   pos=zero+[0,0,float(extraZ)];start=_contact_gap(pos,R,w,scene,board,query,contact_spec) if contact_spec else _gap(pos,R,w,scene,board,query);partial=_contact_gap(pos,R,.028,scene,board,query,contact_spec) if contact_spec else _gap(pos,R,.028,scene,board,query)
   row=dict(extraZ_m=float(extraZ),actual_jaw_gap=start,partial28_gap=partial)
   if contact_spec:
    row['contact_screen']=dict(start=start,partial28=partial)
    if not start.get('passed') or not partial.get('passed'):orientation_screens.append(row);continue
   elif min(start['clearance_m'],partial['clearance_m'])<.0025:orientation_screens.append(row);continue
   if contact_spec:
    c0=_contact_gap(pos,R,.028,scene,board,query,contact_spec);path=[c0]+[_gap(pos+[0,0,dz],R,.028,scene,board,query)for dz in np.linspace(.004,.05,24)]
    if not c0.get('passed') or min(x['clearance_m']for x in path[1:])<.0025:row['retreat_refused']=min(path,key=lambda x:x['clearance_m']);orientation_screens.append(row);continue
   else:path=[_gap(pos+[0,0,dz],R,.028,scene,board,query)for dz in np.linspace(0,.05,26)]
   if (not contact_spec) and min(x['clearance_m']for x in path)<.0025:row['retreat_refused']=min(path,key=lambda x:x['clearance_m']);orientation_screens.append(row);continue
   ik=ctx.tool('puzzle.plan_pose',position=pos.tolist(),rotation=q,arm_id=1);row['native_IK']=ik
   if ik['position_error_m']>.001 or ik['rotation_error_rad']>.03 or ik['joint_limit_margin_rad']<.035:orientation_screens.append(row);continue
   chosen_candidates.append((pos,R,q,desired,float(extraZ),frame.tolist()));orientation_screens.append({**row,'selected_candidate':True})
  screens.append(dict(goal_alias=frame.tolist(),candidates=orientation_screens))
 if not chosen_candidates:raise StageRefusal('No separated low release/partial28/vertical exit candidate: '+json.dumps(_plain(screens)))
 # Evaluate both public symmetry aliases before committing. The former
 # first-feasible selection accepted an unnecessarily high (+15 mm) body
 # release. Choose the lowest certified closed-jaw approach while retaining
 # all CAD, IK, retreat and 0.5 mm guards.
 target,R,q,desired,extraZ,selected_alias=min(chosen_candidates,key=lambda row:row[4]);mark=ctx.tool('robot.execution_journal_cursor');descent=[]
 # Keep the actual closed jaw during descent. Every step starts from a newly
 # sampled actual FK/jaw; the endpoint is a geometric boundary, not a claim
 # that the nominal goal+3 command was reached.
 for _ in range(40):
  actual=ctx.tool('robot.read_proprioception');pos=np.array(actual['tcp'][:3]);error=target-pos
  if np.linalg.norm(error)<.0025:break
  nxt=pos+error*min(1.,.025/max(np.linalg.norm(error),1e-9));points=list(np.linspace(pos,nxt,51));segment=[]
  for u,z in enumerate(points):
   if contact_spec and u>=max(0,len(points)-12):segment.append(_contact_gap(z,R,actual['width_m'],scene,board,query,contact_spec))
   else:segment.append(_gap(z,R,actual['width_m'],scene,board,query))
  if contact_spec:
   if any(x.get('clearance_m',-1.)<.0005 for x in segment[:-12]) or any(not x.get('passed') for x in segment[-12:]):raise StageRefusal('Actual closed-jaw contact descent guard refused '+json.dumps(_plain(segment[-1])))
  elif min(x['clearance_m']for x in segment)<.0005:raise StageRefusal('Actual closed-jaw descent sample lost separation '+json.dumps(_plain(min(segment,key=lambda x:x['clearance_m']))))
  c.move(ctx,nxt,q,1,cartesian=True,accuracy=.0025);descent.append(dict(actual_before=actual,target=nxt.tolist(),minimum_gap_m=min(x['clearance_m']for x in segment)))
 else:raise StageRefusal('Closed-jaw descent feedback did not converge')
 actual=ctx.tool('robot.read_proprioception');origin=np.array(actual['tcp'][:3]);release_start=deepcopy(actual);steps=[];contact_rgb_pair=[];contact_settle=[]
 if contact_spec:
  # Keep the jaw closed at the certified support-contact boundary, settle with
  # measured q16/FK, and sample fresh calibrated RGB before opening.  The
  # vision pair is an observation of the support/roll outcome, never a pose
  # fill or depth/GT claim.
  contact_before=c.state(ctx,{'maroon':[*list(board[:3]+rb@center),*Rotation.from_matrix(rb@goal).as_quat().tolist()]});_pause(ctx,50);contact_after=c.state(ctx,{'maroon':[*list(board[:3]+rb@center),*Rotation.from_matrix(rb@goal).as_quat().tolist()]});contact_rgb_pair=[contact_before,contact_after]
  for _ in range(10):
   actual=ctx.tool('robot.read_proprioception');cg=_contact_gap(np.asarray(actual['tcp'][:3]),Rotation.from_quat(actual['tcp'][3:]).as_matrix(),actual['width_m'],scene,board,query,contact_spec);contact_settle.append(dict(kind='closed_contact',proprio=actual,gap=cg))
   if not cg.get('passed'):raise StageRefusal('Closed-jaw support contact guard refused '+json.dumps(_plain(cg)))
   if actual['width_m']>=.04-.0005:break
   ctx.tool('robot.wait_steps',steps=5)
  # Try only bounded board-tangent micro-slides while still closed.  A slide
  # is issued only when every sampled point retains the named support-contact
  # contract and improves the distal-finger support margin; otherwise the
  # zero-motion settle remains the conservative choice.
  actual=ctx.tool('robot.read_proprioception');base_pos=np.asarray(actual['tcp'][:3]);base_R=Rotation.from_quat(actual['tcp'][3:]).as_matrix();best_gap=_contact_gap(base_pos,base_R,actual['width_m'],scene,board,query,contact_spec);best=None
  for delta in (rb[:,0]*.0005,-rb[:,0]*.0005,rb[:,1]*.0005,-rb[:,1]*.0005):
   candidate=base_pos+delta;sample=[_contact_gap(z,base_R,actual['width_m'],scene,board,query,contact_spec)for z in np.linspace(base_pos,candidate,13)]
   if all(x.get('passed') for x in sample) and sample[-1]['support_finger_clearance_m']>best_gap['support_finger_clearance_m']+1.e-6:
    best=(candidate,sample[-1],sample)
  if best is not None:
   c.move(ctx,best[0],c.qdict(actual['tcp'][3:]),1,cartesian=True,accuracy=.0025);contact_settle.append(dict(kind='closed_contact_micro_slide',offset_m=(best[0]-base_pos).tolist(),gap=best[1],samples=best[2]))
  else:contact_settle.append(dict(kind='closed_contact_settle',offset_m=[0.,0.,0.],gap=best_gap))
  # Open only at the settled contact pose, then retreat while rechecking
  # support contact until the normal positive-separation guard resumes.
  for _ in range(10):
   actual=ctx.tool('robot.read_proprioception');cg=_contact_gap(np.asarray(actual['tcp'][:3]),Rotation.from_quat(actual['tcp'][3:]).as_matrix(),actual['width_m'],scene,board,query,contact_spec)
   if not cg.get('passed'):raise StageRefusal('Support contact drifted before opening '+json.dumps(_plain(cg)))
   if actual['width_m']<.04-.0005:
    width=min(.04,actual['width_m']+.002);ctx.tool('robot.set_grip',arm_id=1,width_m=width,ramp_steps=8,settle_steps=4);steps.append(dict(status='command',operation='open',command=dict(width_m=width),contact_gap=cg));continue
   break
  for _ in range(40):
   actual=ctx.tool('robot.read_proprioception');pos=np.asarray(actual['tcp'][:3]);target_exit=origin+np.array([0.,0.,.003]);
   if pos[2]>=origin[2]+.05:break
   nxt=pos+np.array([0.,0.,min(.003,max(0.,origin[2]+.05-pos[2]))]);samples=[_contact_gap(z,Rotation.from_quat(actual['tcp'][3:]).as_matrix(),actual['width_m'],scene,board,query,contact_spec) if z[2]-origin[2]<.004 else _gap(z,Rotation.from_quat(actual['tcp'][3:]).as_matrix(),actual['width_m'],scene,board,query) for z in np.linspace(pos,nxt,13)]
   if not all(x.get('passed',x.get('clearance_m',-1.)>=.0005) for x in samples):raise StageRefusal('Support contact retreat guard refused '+json.dumps(_plain(samples[-1])))
   c.move(ctx,nxt,c.qdict(actual['tcp'][3:]),1,cartesian=True,accuracy=.0025);steps.append(dict(status='command',operation='withdraw',command=dict(position=nxt.tolist()),samples=samples))
  else:raise StageRefusal('Closed-jaw support retreat exhausted')
  steps.append(dict(status='completed',operation='contact_release',contact_rgb_pair=[_observation(x) for x in contact_rgb_pair],settle=contact_settle))
 else:
  for _ in range(160):
   actual=ctx.tool('robot.read_proprioception');step=next_release_step(actual_position=actual['tcp'][:3],actual_rotation=Rotation.from_quat(actual['tcp'][3:]).as_matrix(),actual_width=actual['width_m'],release_origin=origin,retreat_vector=[0,0,.05],final_width=.04,min_release_width=.0275,scene=scene,piece='maroon',board=board,proprio_provenance=actual,board_query=query)
   steps.append(step)
   if step['status']=='completed':break
   if step['status']!='command':raise StageRefusal('Actual progressive release refused '+json.dumps(_plain(step)))
   if step['operation']=='open':ctx.tool('robot.set_grip',arm_id=1,width_m=step['command']['width_m'],ramp_steps=8,settle_steps=4)
   else:c.move(ctx,step['command']['position'],c.qdict(Rotation.from_matrix(step['command']['rotation']).as_quat()),1,cartesian=True,accuracy=.0025)
  else:raise StageRefusal('Progressive release feedback exhausted')
 # Both hands leave only after the actual measured opening/withdrawal complete.
 ctx.tool('robot.set_grip',arm_id=1,width_m=.055,ramp_steps=18,settle_steps=15);ctx.tool('robot.go_home',arm_id=1)
 cp,cq=c.hand(ctx,0);ctx.tool('robot.set_grip',arm_id=0,width_m=.055,ramp_steps=18,settle_steps=15);c.move(ctx,cp+[0,0,.06],cq,0,cartesian=True);ctx.tool('robot.go_home',arm_id=0);_pause(ctx,100)
 final=c.state(ctx,{'maroon':[*list(board[:3]+rb@center),*Rotation.from_matrix(rb@goal).as_quat().tolist()]});record=dict(mark=mark,actualheld_source=source,target_screens=screens,selected_goal_alias=selected_alias,release_support_contact=contact_spec,contact_rgb_pair=[_observation(x) for x in contact_rgb_pair],contact_rgb_ray_cad_support=[_rgb_ray_cad_support(x) for x in contact_rgb_pair],contact_settle=contact_settle,extraZ_over_learned_preopen_m=extraZ,tracking_clearance_reserve_m=.0025,nominal_requested_drop_m=.003,learned_board_preopen_z_m=float(bias[2]),geometric_release_body_z_over_goal_m=float(bias[2]+extraZ),release_start=release_start,descent=descent,feedback_steps=steps,scope='Geometrically separated progressive commands from actual RGB/FK/jaw. Goal+3 remained desired; actual release higher and finalplacement is UNKNOWN without external score.')
 return _output('executed',{**state,'release_execution':record,'execution_mark':mark},final,'release_execution',target_3d_status='UNKNOWN',visual_release_receipt=False,actual_release_body_z_over_goal_m=float(bias[2]+extraZ))

def checkpoint(kind,world,outputs):
 try:
  s=outputs['state'];e=outputs['evidence'];seen=world.sim_state['vision'];r=s['release_execution'];mark=r['mark'];records=world.sim_state['robot_command_journal'];nonce=world.sim_state['robot_command_session']
  if outputs['decision']!='executed'or e['target_3d_status']!='UNKNOWN'or e['visual_release_receipt']is not False:return False
  if mark['session_nonce']!=nonce or seen['timestamp']!=e['timestamp']or seen['frame_sha256']!=e['frame_sha256']:return False
  calls={x['call_id']:x for x in records if x.get('event')=='call'and x.get('session_nonce')==nonce and x['call_id']>mark['call_id']};returned={x['call_id']:x for x in records if x.get('event')=='return'and x.get('session_nonce')==nonce}
  if not calls or any(i not in returned for i in calls):return False
  for x in calls.values():
   ret=returned[x['call_id']].get('result');tool=x['tool']
   if tool=='robot.execute_trajectory'and ret.get('status')!='ok':return False
   if tool=='robot.go_home'and isinstance(ret,dict)and(ret.get('homed')is False or ret.get('status')in['failed','error']):return False
   if tool=='robot.set_grip'and ret.get('clamped')is not False:return False
  return r['feedback_steps'][-1]['status']=='completed'and world.sim_state['robot_tcp_poses']['1'][2]>1.05
 except(KeyError,ValueError,TypeError):return False

def resume_exit(ctx,state:StageState)->StageResult:
 """Only finish withdrawal after actual opening, never rebind/regrasp target."""
 seen=c.state(ctx,{**state['latest_rgb']['poses']});history=[*state['historical_rgb_records'],deepcopy(seen)];actual=ctx.tool('robot.read_proprioception');p=state['parameters'];prior=next((r for r in reversed(history)if'board'in r['poses']),None)
 if prior is None:raise StageRefusal('No genuine board RGB prior for exit')
 board=np.array(seen['poses'].get('board',prior['poses']['board']));historical='board'not in seen['poses'];board_margin=.004 if historical else 0.
 if historical and actual['timestamp']-prior['timestamp']>30.:raise StageRefusal('Historicalboard expired before safeexit')
 rb=Rotation.from_quat(board[3:]).as_matrix();targets=goals('pyramid',p['pyramid_plan']);occupancy={}
 for n in targets:
  if n=='maroon':
   occupancy[n]=dict(source='UNKNOWN releasedtarget excluded by explicit target-contact scope',pose_hypotheses=[seen['poses'][n]]if n in seen['poses']else[],inflation_m=0.);continue
  if n in p['conditional_prefix']:
   center,R=targets[n];options=[[*list(board[:3]+rb@center),*Rotation.from_matrix(rb@R).as_quat().tolist()]]
   if n in seen['poses']:options.append(seen['poses'][n])
   occupancy[n]=dict(source='Originalprefix8 public4mm precondition',pose_hypotheses=options,inflation_m=.004+board_margin)
  elif n in seen['poses']:occupancy[n]=dict(source='CurrentRGB',pose_hypotheses=[seen['poses'][n]],inflation_m=0.)
  else:
   old=next((r for r in reversed(history)if n in r['poses']),None)
   if old is None or actual['timestamp']-old['timestamp']>30.:raise StageRefusal('No bounded real prior '+n)
   occupancy[n]=dict(source='Actual recent unplacedRGB conditional4mm',pose_hypotheses=[old['poses'][n]],inflation_m=.004)
 scene=dict(contract='Remainingexit publicprefix8, releasedtarget UNKNOWN; no targetpose imputed',measurement=seen,occupancy=occupancy,board_for_planning=dict(pose=board.tolist(),uncertainty_m=board_margin));origin=np.array(state['actual_release_origin']);mark=ctx.tool('robot.execution_journal_cursor');query=PublicBoardQuery();steps=[];positive_egress=None
 # Existing positive sub-margin geometry is not a negative-clearance waiver.
 # Only a fixed-jaw escape that monotonically increases every sampled worst
 # clearance is admissible, and its destination restores the usual0.5mm.
 actual=ctx.tool('robot.read_proprioception');ep=np.array(actual['tcp'][:3]);eR=Rotation.from_quat(actual['tcp'][3:]).as_matrix();initial=_gap(ep,eR,actual['width_m'],scene,board,query)
 if 0<initial['clearance_m']<.0005:
  escape=ep+[0,0,.006];samples=[dict(position=x.tolist(),**_gap(x,eR,actual['width_m'],scene,board,query))for x in np.linspace(ep,escape,13)];gs=[x['clearance_m']for x in samples]
  if min(gs)<=0 or any(bb<aa-1.e-8 for aa,bb in zip(gs,gs[1:]))or gs[-1]<.003:raise StageRefusal('No monotone positive fixedjaw escape')
  c.move(ctx,escape,c.qdict(actual['tcp'][3:]),1,cartesian=True,accuracy=.0025)
  actual_after=ctx.tool('robot.read_proprioception');measured_gap=_gap(actual_after['tcp'][:3],Rotation.from_quat(actual_after['tcp'][3:]).as_matrix(),actual_after['width_m'],scene,board,query)
  if measured_gap['clearance_m']<=initial['clearance_m']or measured_gap['clearance_m']<.0005:raise StageRefusal('Positivebuffer retreat didnot measurably restore margin; noretry')
  positive_egress=dict(actual_after=actual_after,actual_gap_after=measured_gap,tracking_reserve_m=.0025,scope='Existingpositive sub0.5mm margin only; nonegative initial/samples, fixedactualjaw, every0.5mm sample monotone, end>=0.5mm. Normalguard resumes after thissinglewithdrawal.',actual_before=actual,samples=samples)
 for _ in range(120):
  actual=ctx.tool('robot.read_proprioception');step=next_release_step(actual_position=actual['tcp'][:3],actual_rotation=Rotation.from_quat(actual['tcp'][3:]).as_matrix(),actual_width=actual['width_m'],release_origin=origin,retreat_vector=[0,0,.05],final_width=.04,min_release_width=.0275,scene=scene,piece='maroon',board=board,proprio_provenance=actual,board_query=query);steps.append(step)
  if step['status']=='completed':break
  if step['status']!='command':raise StageRefusal('Remainingexit refused '+json.dumps(_plain(step)))
  if step['operation']=='open':ctx.tool('robot.set_grip',arm_id=1,width_m=step['command']['width_m'],ramp_steps=8,settle_steps=4)
  else:c.move(ctx,step['command']['position'],c.qdict(Rotation.from_matrix(step['command']['rotation']).as_quat()),1,cartesian=True,accuracy=.0025)
 else:raise StageRefusal('Remainingexit feedback exhausted')
 actual=ctx.tool('robot.read_proprioception');R=Rotation.from_quat(actual['tcp'][3:]).as_matrix();tests=[_gap(actual['tcp'][:3],R,w,scene,board,query)for w in np.linspace(actual['width_m'],.055,31)]
 if min(r['clearance_m']for r in tests)<.0005:raise StageRefusal('Finalparkwidth sweep refused')
 ctx.tool('robot.set_grip',arm_id=1,width_m=.055,ramp_steps=18,settle_steps=15);ctx.tool('robot.go_home',arm_id=1);cp,cq=c.hand(ctx,0);ctx.tool('robot.set_grip',arm_id=0,width_m=.055,ramp_steps=18,settle_steps=15);c.move(ctx,cp+[0,0,.06],cq,0,cartesian=True);ctx.tool('robot.go_home',arm_id=0);_pause(ctx,150);gc,gR=targets['maroon'];final=c.state(ctx,{'maroon':[*list(board[:3]+rb@gc),*Rotation.from_matrix(rb@gR).as_quat().tolist()]})
 record=dict(mark=mark,feedback_steps=steps,positive_egress=positive_egress,source_actual_release_origin=origin.tolist(),target_3d_status='UNKNOWN',scope='Only completion of existingactualrelease exit. No regrasp, no targetgoal4mmprecondition, all originalprefix8 guarded. Physicalplacement externally audited.')
 return _output('executed',{**state,'release_execution':record,'execution_mark':mark},final,'release_execution',target_3d_status='UNKNOWN',visual_release_receipt=False)
