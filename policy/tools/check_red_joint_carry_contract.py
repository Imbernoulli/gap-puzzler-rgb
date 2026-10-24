"""New dynamic joint carry tool and refresh contracts, without physics."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import puzzler
assert Path(puzzler.__file__).resolve().parent == ROOT/'puzzler'
from puzzler import red_bimanual_goal as m
checks=[]
def check(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)
belief=dict(uncertainty_m=.008,object_in_tcp_position=[0,0,0],object_in_tcp_rotation=np.eye(3).tolist())
board=[.45,0,.753,0,0,0,1]
scene=dict(board_for_planning=dict(pose=board,uncertainty_m=.004))
calls=[]
class Ctx:
    def tool(self,name,**kwargs):
        calls.append((name,kwargs));return dict(feasible=True,min_joint_limit_margin_rad=.1,path=[[0]*6,[.012]*6])
class G:
    scratch=None
    robot=SimpleNamespace(iks={'right':SimpleNamespace(ee_pose=lambda state:([.4,0,.9],[0,0,0,1]))})
    def densify_q16_path(self,q):return np.asarray(q)
    def fk(self,q):pass
    def screen_q16_path(self,q,**kwargs):
        calls.append(('full9',kwargs))
        return dict(safe=True,rows=[dict(groups={k:dict(clearance_m=.010)for k in('robot_board79','held_envelope_board79')})])
def actual(ctx,arm):return dict(tcp=[.4,0,.9,0,0,0,1],robot_encoder_values=[0]*16,width_m=.024)
def hand(*args,**kwargs):calls.append(('local_hand',args[-1]['uncertainty_m']));return dict(safe=True)
way=dict(position=[.45,-.1,.9],rotation=dict(x=0,y=0,z=0,w=1))
with patch.object(m.base,'proprio',side_effect=actual),patch.object(m,'geometry',return_value=G()),patch.object(m,'_body_at',return_value=np.array([.4,0,.9,0,0,0,1])),patch.object(m,'_hand_guard',side_effect=hand):
    motion=m._plan(Ctx(),dict(piece='red',arm=1),scene,belief,[way],joint_endpoint=True)
check('Joint carry uses only online endpoint pose tool',calls[0][0]=='puzzle.plan_joint_path' and calls[0][1]['position']==way['position'])
check('Joint carry keeps original35mrad reserve',calls[0][1]['min_joint_limit_margin_rad']==.035)
check('Joint carry goes through full9 with original8mm envelope',calls[1][0]=='full9' and calls[1][1]['held_envelope_m']==.008)
check('Each dense joint sample retains original local hand guards',sum(e[0]=='local_hand'for e in calls)==len(motion['q16_path']))
check('Joint proposal is labelled separately from Cartesian/terminal',motion['planning_space']=='live_IK_endpoint_joint_interpolation' and not motion['terminal_contract'])
try:
    with patch.object(m.base,'proprio',side_effect=actual):
        m._plan(Ctx(),dict(piece='red',arm=1),scene,belief,[way],joint_endpoint=True,terminal=True)
    raise AssertionError('Expected terminal joint refusal')
except m.StageRefusal:pass
check('Terminal descent cannot use joint endpoint interpolation',True)

# Exercise carry ordering with an actual scene update between the two legs.
br=Rotation.from_euler('z',90,degrees=True).as_matrix();q=Rotation.from_matrix(br).as_quat()
initial_board=np.r_[[.45,0,.753],q];fresh_board=initial_board.copy();fresh_board[1]+=.010
first_scene=dict(board_for_planning=dict(pose=initial_board.tolist(),uncertainty_m=0.))
fresh_scene=dict(board_for_planning=dict(pose=fresh_board.tolist(),uncertainty_m=0.))
initial_belief={**belief,'epoch':'before'};fresh_belief={**belief,'epoch':'fresh_after_outside'}
state=dict(piece='red',arm=1,parameters=dict(pyramid_plan='public',assembly_order=['red']),plan={},
    handoff=dict(stage='verified',hop=2,candidate=dict(bead=4)),observation=dict(timestamp=1.,frame_sha256={'top':'actual'}))
plans=[];events=[]
def refresh(ctx,s,label,**kwargs):
    events.append(('refresh',label));first=label=='goal_entry'
    return s,initial_belief if first else fresh_belief,first_scene if first else fresh_scene,dict(attempts=[dict(status='actual_RGB_q16_rebound')])
def plan(ctx,s,scene,b,ways,**kwargs):
    plans.append(dict(ways=ways,belief_epoch=b['epoch'],**kwargs));events.append(('plan',b['epoch']))
    return dict(endpoint=ways[-1])
def drive(ctx,s,motion,**kwargs):events.append(('drive',motion['endpoint']));return {}
def pose(ctx,**kwargs):return dict(position_error_m=0.,rotation_error_rad=0.,joint_limit_margin_rad=.4)
with patch.object(Ctx,'tool',side_effect=lambda name,**kwargs:pose(None,**kwargs)),patch.object(m.base,'proprio',side_effect=actual),patch.object(m.base,'public_symmetries',return_value=[np.eye(3)]),patch.object(m,'goals',return_value={'red':(np.array([.1,0,.05]),np.eye(3))}),patch.object(m,'_hand_guard',return_value=dict(safe=True)),patch.object(m,'_terminal_external',return_value=dict(safe=True)),patch.object(m,'_refresh',side_effect=refresh),patch.object(m,'_plan',side_effect=plan),patch.object(m,'_drive',side_effect=drive),patch.object(m,'_align_goal_hover',side_effect=lambda ctx,s,b,sc,r,**kwargs:(s,b,sc,dict(status='actual_RGB_goal_hover_aligned'))):
    out=m.carry(Ctx(),state)
expected_pretranslate=np.array([.4,0,.9])+br@np.array([0,-.04,0])
expected_first=fresh_board[:3]+br@np.array([.1,0,.05])+br@np.array([0,-.1,0])+[0,0,.085]
expected_second=fresh_board[:3]+br@np.array([.1,0,.05])+[0,0,.036]
check('Current-attitude pretranslation follows actual board frame',np.allclose(plans[0]['ways'][0]['position'],expected_pretranslate)and not plans[0].get('joint_endpoint',False))
check('Outside point follows actual board/goal rotation instead of world coordinates',np.allclose(plans[1]['ways'][0]['position'],expected_first))
check('Fresh board and attachment recompute ingress point',plans[2]['belief_epoch']=='fresh_after_outside' and np.allclose(plans[2]['ways'][-1]['position'],expected_second))
check('Joint outside leg retains original Cartesian ingress afterwards',len(plans)==3 and plans[1]['joint_endpoint'] and not plans[2].get('joint_endpoint',False))
check('Receiver RGB refresh separates every actual motor leg',[e[0]for e in events]==['refresh','plan','drive','refresh','plan','drive','refresh','plan','drive','refresh'])
check('Transport remains routed to original independent carried checkpoint',out['route']=='carried' and out['state']['transport_record']['route_proposal']['saved_world_or_joint_path']is False)

# The final empty left hand has a same-arm Cartesian-home grazing witness.
# A new online joint proposal still uses the unchanged dated release guards.
home_calls=[]
class HomeGeometry:
    scratch=None
    robot=SimpleNamespace(iks={'left':SimpleNamespace(_home_q=[0]*6,ee_pose=lambda state:([.3,.2,1.],[0,0,0,1]))})
    def fk(self,q):pass
def empty_plan(ctx,s,sc,arm,ways,released,uncertainty,**kwargs):
    home_calls.append(dict(arm=arm,ways=ways,uncertainty=uncertainty,**kwargs))
    if not kwargs.get('joint_endpoint',False):raise m.c.PlanningRefusal('Original same-arm clearance gate refused')
    return dict(endpoint=ways[-1])
with patch.object(m.base,'proprio',side_effect=actual),patch.object(m,'geometry',return_value=HomeGeometry()),patch.object(m,'_empty_plan',side_effect=empty_plan),patch.object(m,'_empty_execute',return_value={'new_path_executed':True}):
    home=m._home(None,state,scene,0,np.array([.45,0,.814,0,0,0,1]),.009)
check('Three refused Cartesian home proposals retain a current-joint alternative',len(home_calls)==4 and home_calls[-1]['joint_endpoint']and all(not x.get('joint_endpoint')for x in home_calls[:-1]))
check('Empty-hand home keeps original dated9mm released-body envelope',all(x['uncertainty']==.009 for x in home_calls))
check('Joint home endpoint is recomputed from public home FK',home_calls[-1]['ways'][0]['position']==[.3,.2,1.]and home['planning_space']=='live_IK_endpoint_joint_interpolation')
report=dict(status='PASS',cases=len(checks),checks=checks,
    scope='Offline tool selection, exact existing guards and fresh-source control order; no physical completion claim')
report['packaged_source']=str(Path(m.__file__).resolve())
(ROOT/'verification/red_joint_carry_contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
