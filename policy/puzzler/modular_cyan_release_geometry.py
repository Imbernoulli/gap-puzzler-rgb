"""CYAN public geometry and measured robot path helpers, adapted from shared actual runtime.
No initializer, teacher coordinates, action tape or evaluator inputs.
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from . import model,conditional_scene,vision_controller as c
from .vision_observer import goals
from .gripper_geometry import clearance,scene_clearance

def plain(x):return json.loads(json.dumps(x,default=lambda v:v.tolist()if isinstance(v,np.ndarray)else float(v),allow_nan=False))
def robot(ctx,arm):
 row=ctx.tool('robot.read_proprioception',arm_id=arm)
 return {**row,'actual_jaw_width_m':row['width_m']}
def read(ctx,state):
 row=robot(ctx,state['arm']);return row,np.asarray(row['tcp'][:3]),Rotation.from_quat(row['tcp'][3:]).as_matrix(),row['actual_jaw_width_m']
def geometry(scene,board,query,p,r,w):
 margin=scene.get('board_for_planning',{}).get('uncertainty_m',0.)
 centers,radii=conditional_scene.spheres(scene,exclude=('cyan',))
 finite=query.query(board,p,r,w);finite={**finite,'clearance_m':finite['clearance_m']-margin}
 checks=dict(other=clearance(p,r,w,centers,radius=radii),table=scene_clearance(p,r,w,[],table_z=float(board[2]-.003+margin),plane_z=float(board[2]-.003+margin),board=board),board=finite)
 key=min(checks,key=lambda k:checks[k]['clearance_m']);return dict(clearance_m=float(checks[key]['clearance_m']),worst=key,witness=checks[key].get('witness'))

def target_candidates(state,scene,board,record,query):
 a=state['release_held_belief'];relative_r=np.asarray(a['object_in_tcp_rotation']);relative_p=np.asarray(a['object_in_tcp_position']);br=Rotation.from_quat(board[3:]).as_matrix();center,goalr=goals('pyramid',state['parameters']['pyramid_plan'])['cyan'];local=model.local_beads('cyan');actual_w=record['actual_jaw_width_m'];minimum=max(2*model.RADIUS+.0025,actual_w+.003);rows=[]
 for turn in range(4):
  for flip in(False,True):
   sym=model.rotation(turn*np.pi/2)@(np.diag([1,-1,-1])if flip else np.eye(3))
   if np.max(np.min(np.linalg.norm((local@sym.T)[:,None]-local[None],axis=2),axis=1))>1e-7:continue
   r=br@goalr@sym@relative_r.T;zero=board[:3]+br@center-r@relative_p;widths=np.linspace(actual_w,minimum,max(2,int(np.ceil((minimum-actual_w)/.0005))+1))
   def check(h):
    hand=min([geometry(scene,board,query,zero+[0,0,h],r,float(w))for w in widths],key=lambda q:q['clearance_m'])
    cover,cover_sizes=conditional_scene.local_sphere_cover('cyan');oc,os=conditional_scene.spheres(scene,exclude=('cyan',));points=cover@(r@relative_r).T+zero+[0,0,h]+r@relative_p;pair=np.linalg.norm(points[:,None]-oc[None],axis=2)-cover_sizes[:,None]-os[None]-state['release_held_uncertainty_m'];gap=min(float(pair.min()),float(np.min(points[:,2]-cover_sizes-state['release_held_uncertainty_m']-(board[2]-.003))))
    held=dict(clearance_m=gap,worst='conditional_held8mm',witness=dict(scope='Retained 8mm conditional held CAD versus all other retained occupancy/table before opening'))
    return min((hand,held),key=lambda x:x['clearance_m'])
   found=None;scans=[]
   # These are geometry-search parameters, not a replayed world-coordinate path.
   for h in np.arange(.003,.081,.002):
    v=check(float(h));scans.append(dict(height_m=float(h),**v))
    if v['clearance_m']>=.0005:
     lo=max(.003,float(h)-.002);hi=float(h)
     for _ in range(7):
      mid=(lo+hi)/2
      if check(mid)['clearance_m']>=.0005:hi=mid
      else:lo=mid
     final=hi+.0025;found=dict(height_m=final,closest_swept_height_m=hi,tracking_reserve_m=.0025,clearance=check(final));break
   if found:
    rows.append(dict(symmetry=dict(turn=turn,flip=flip),zero_tcp=zero.tolist(),rotation=r.tolist(),rotation_q=c.qdict(Rotation.from_matrix(r).as_quat()),endpoint_tcp=(zero+[0,0,found['height_m']]).tolist(),hover=(zero+[0,0,max(.100,found['height_m']+.060)]).tolist(),minimum_release_width_m=minimum,minimum_release_width_basis='max(public25mm bead diameter+2.5mm clearance, measured held jaw+3mm); command heuristic, not detached proof',refinement=found,scanned_closed_to_partial_sweeps=scans))
 if not rows:raise c.PlanningRefusal('No public-CAD symmetry supports closedjaw plus partial-release width')
 return sorted(rows,key=lambda q:q['refinement']['height_m'])

def execute_screened_path(ctx,state,scene,board,query,target,rotation,*,held=True,accuracy=.0025):
 arm=state['arm'];record,p,r,width=read(ctx,state);plan=ctx.tool('puzzle.plan_path',position=plain(target),rotation=rotation,arm_id=arm,min_joint_limit_margin_rad=.035)
 if not plan.get('feasible'):plan=ctx.tool('puzzle.plan_joint_path',position=plain(target),rotation=rotation,arm_id=arm,min_joint_limit_margin_rad=.035)
 if not plan.get('feasible'):raise c.PlanningRefusal('Current native IK path refused '+str(plan.get('errors')))
 if len(plan.get('tcp_path',()))!=len(plan['path'])or len(plan['path'])<2:raise c.PlanningRefusal('Missing native robot-only FK path')
 a=state['release_held_belief'];rp=np.asarray(a['object_in_tcp_position']);rr=np.asarray(a['object_in_tcp_rotation']);local,sizes=conditional_scene.local_sphere_cover('cyan');oc,os=conditional_scene.spheres(scene,exclude=('cyan',));checks=[]
 for sample in plan['tcp_path']:
  pp=np.asarray(sample['position']);sr=Rotation.from_quat([sample['rotation'][k]for k in 'xyzw']).as_matrix();check=geometry(scene,board,query,pp,sr,width)
  if held:
   body=local@(sr@rr).T+pp+sr@rp;pair=np.linalg.norm(body[:,None]-oc[None],axis=2)-sizes[:,None]-os[None]-state['release_held_uncertainty_m'];held_gap=min(float(pair.min()),float(np.min(body[:,2]-sizes-state['release_held_uncertainty_m']-(board[2]-.003))))
  else:held_gap=None
  checks.append(dict(hand=check,conditional_held8mm_clearance_m=held_gap))
 if any(x['hand']['clearance_m']<.0005 or held and x['conditional_held8mm_clearance_m']<.0005 for x in checks):
  raise c.PlanningRefusal('Live IK path failed retained hand/table/board or conditionalheld8mm CAD '+json.dumps(plain(min(checks,key=lambda x:min(x['hand']['clearance_m'],x['conditional_held8mm_clearance_m']if held else 100)))))
 waypoints=[]
 for aa,bb in zip(np.asarray(plan['path'])[:-1],np.asarray(plan['path'])[1:]):
  count=max(1,int(np.ceil(np.max(np.abs(bb-aa))/.018)));waypoints.extend(dict(positions=(aa+(bb-aa)*u).tolist())for u in np.linspace(0,1,count+1)[1:])
 result=ctx.tool('robot.execute_trajectory',trajectory=dict(waypoints=waypoints),arm_id=arm,max_steps_per_waypoint=2)
 if result.get('status')!='ok':raise RuntimeError('Dynamic native path motor failed')
 # Feedback arrival is required; corrections may not bypass geometry. A
 # geometric adjustment is replanned in the next graph invocation instead.
 c.await_motion(ctx,target,rotation,arm,accuracy=accuracy)
 return dict(plan=plan,cad_checks=checks,motor_result=result,after=robot(ctx,arm))
