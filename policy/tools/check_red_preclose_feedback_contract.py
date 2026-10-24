"""Offline authentic-source feedback ordering; no Env/physics/motor execution."""
from pathlib import Path
import sys,json
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import puzzler
assert Path(puzzler.__file__).resolve().parent == ROOT/'puzzler'
from puzzler import red_bimanual_handoff as h
checks=[]
def check(name,passed):
 if not passed:raise AssertionError(name)
 checks.append(name)
old=dict(object_in_tcp_position=[0,0,0],object_in_tcp_rotation=np.eye(3).tolist(),uncertainty_m=.006)
new={**old,'object_in_tcp_position':[0,.009,0]}
scene={};receiver_point=np.array([.4,0,.95]);moves=[]
def proprio(ctx,arm):
 return dict(tcp=[*([.4,0,.95] if arm==0 else receiver_point),0,0,0,1],width_m=.055,robot_encoder_values=[0]*16)
def observe(ctx,hints):return dict(poses={'red':[.4,.009,.95,0,0,0,1]},timestamp=20.,frame_sha256={'top':'actual'},observation_source='rgb_shape_geometry')
def source(observed,role):return dict(piece='red',pose=observed['poses']['red'],timestamp=observed['timestamp'])
def rebound(ctx,state,observed,sender,belief,role):
 return state,new,dict(source=source(observed,role),measured_source_supersedes_old_rigid_attachment=True)
def screen(ctx,state,scene,belief,held,moving,ways,contract,**kw):
 moves.append(ways[-1]['position']);return dict(endpoint=ways[-1])
def execute(ctx,state,motion,**kw):
 receiver_point[:]=motion['endpoint']['position'];return dict(actual_after={'timestamp':21.})
state=dict(piece='red',arm=0,parameters={},observation={},rgb_history=[],belief_ledger={},initialized_prefix=[],executed_placements=[])
with patch.object(h.base,'proprio',side_effect=proprio),patch.object(h.c,'state',side_effect=observe),patch.object(h,'_source',side_effect=source),patch.object(h,'_rebind_sender_observation',side_effect=rebound),patch.object(h,'scene_for_motion',side_effect=lambda ctx,state,belief:(state,scene,{})),patch.object(h,'_screen',side_effect=screen),patch.object(h,'_execute',side_effect=execute):
 changed,bound,sc,preclose,records=h._recenter_preclose(object(),state,0,1,old,scene,dict(hand_in_body_position=[0,0,0],hand_in_body_rotation=np.eye(3).tolist()),{0:[1],1:[4]},10.)
check('Preclose real9mm body displacement recomputes open receiver point',len(moves)==1 and np.allclose(moves[0],[.4,.009,.95]))
check('Latest measured sender attachment supersedes stale rigid source',bound['object_in_tcp_position']==[0,.009,0])
check('Receiver alignment converges from actual RGB rather than desired source',preclose['timestamp']==20. and len(records)==2)

# Exercise the actual receiver_close function with separate observation,
# binding, geometry and command events. Sender has no opening instruction.
events=[];width=.055;tick=20.;bound_count=0
class Context:
 def tool(self,name,**kwargs):
  global width
  if name=='robot.set_grip':
   events.append(('grip',kwargs['arm_id'],kwargs['width_m']));width=max(.0242,kwargs['width_m']);return dict(status='ok')
  if name=='puzzle.propagate_pyramid_held_belief':return {'actual':'receiver source binding'}
  if name=='robot.wait_steps':return dict(status='ok')
  raise AssertionError(name)
def current(ctx,arm):
 q=np.zeros(16);q[14:16]=[width/2.,-width/2.]
 return dict(tcp=[.4,0,.95,0,0,0,1],width_m=width if arm==1 else .0242,robot_encoder_values=q.tolist())
def observed(ctx,hints):
 global tick
 tick+=1.;events.append(('observe',tick))
 return dict(poses={'red':[.4,.001*(tick-20),.95,0,0,0,1]},timestamp=tick,frame_sha256={'top':str(tick)},observation_source='rgb_shape_geometry')
def bind(ctx,state,obs,sender,prior,role):
 events.append(('rebind',obs['timestamp']));fresh={**old,'object_in_tcp_position':[0,.001*(obs['timestamp']-20),0]}
 return state,fresh,{'source':source(obs,role)}
class Geometry:
 def screen_q16_path(self,*a,**kw):
  events.append(('guard',kw['object_in_tcp_position'][1]));return dict(safe=True)
prepared={**state,'handoff':dict(sender=0,receiver=1,stage='prepared',preclose_rgb={'timestamp':20.},sender_arrival_timestamp=10.,contract={0:[1],1:[4]})}
with patch.object(h.base,'proprio',side_effect=current),patch.object(h,'_belief',return_value=old),patch.object(h.c,'state',side_effect=observed),patch.object(h,'_source',side_effect=source),patch.object(h,'_rebind_sender_observation',side_effect=bind),patch.object(h,'scene_belief',return_value=scene),patch.object(h,'scene_for_motion',side_effect=lambda ctx,state,belief:(state,scene,{})),patch.object(h,'geometry',return_value=Geometry()),patch.object(h,'_guard_passes',return_value=True):
 out=h.receiver_close(Context(),prepared)
commands=[e for e in events if e[0]=='grip']
check('Progressive close requests public ladder then real fullclose',[(e[1],e[2])for e in commands]==[(1,.030),(1,.026),(1,.0242),(1,0.)])
check('Sender stays closed throughout receiver feedback',all(e[1]==1 for e in commands))
check('Each actual close step gets new observation and source binding',sum(e[0]=='observe'for e in events)==4 and sum(e[0]=='rebind'for e in events)==4)
for i,e in enumerate(events):
 if e[0]=='rebind':
  following=next(x for x in events[i+1:] if x[0]=='guard');assert following[1]>.0
check('Postclose material gates use rebound current RGB not oldzero source',True)
check('Receiver still returns dualheld route for independent physical CP',out['route']=='dual_held' and out['state']['handoff']['stage']=='dual_held')
check('No measured receiver attachment is minted from the grasp recipe',out['state']['handoff']['receiver_attachment']['source']['timestamp']==24.)
report=dict(status='PASS',cases=len(checks),checks=checks,scope='Offline control ordering/contracts only; independent physical dual-hold still mandatory')
report['packaged_source']=str(Path(h.__file__).resolve())
(ROOT/'verification/red_feedback_contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
