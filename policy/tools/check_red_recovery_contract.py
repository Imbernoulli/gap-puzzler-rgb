"""Control conflict recovery and stable intermediate contracts, no physics."""
from pathlib import Path
from unittest.mock import patch
from copy import deepcopy
import json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import puzzler
assert Path(puzzler.__file__).resolve().parent == ROOT/'puzzler'
from puzzler import red_bimanual_handoff as h,red_bimanual_goal as m
checks=[]
def check(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)
old_source={'dated':'original'}
old=dict(object_in_tcp_position=[0,0,0],object_in_tcp_rotation=np.eye(3).tolist(),uncertainty_m=.008,source=old_source,source_age_s=6.,predicted_body_pose=[.4,0,.95,0,0,0,1])
state=dict(piece='red',arm=1,parameters=dict(owned_held_belief_source=old_source),
    handoff=dict(sender=0,receiver=1,hop=2,stage='verified',contract={0:[1],1:[4]},candidate={'bead':4}),
    observation=dict(timestamp=1.,frame_sha256={'top':'old'}),executed_placements=[],execution_tokens={},
    rgb_history=[dict(timestamp=1.,poses={'board':[.45,0,.753,0,0,0,1]})])
def actual(ctx,arm):return dict(timestamp=2.,tcp=[.4,0,.95,0,0,0,1],width_m=.024 if arm==1 else .055,robot_encoder_values=[0]*16)
bad=dict(poses={'red':[.4,0,.95,0,0,1,0]},timestamp=2.,frame_sha256={'top':'authentic-conflicting-image'},observation_source='rgb_shape_geometry')
def source(obs,role):return dict(pose=obs['poses']['red'],timestamp=obs['timestamp'],frame_sha256=obs['frame_sha256'])
def saved(s,o):return {**s,'observation':o}
calls=[]
seen_hints=[]
class Ctx:
    def tool(self,name,**kwargs):calls.append((name,kwargs));return {}
def retained(ctx,s,arm):calls.append(('retained_full_scene',arm));return s,old,{'held_envelope_m':.008}
with patch.object(h,'_belief',return_value=old),patch.object(h.base,'proprio',side_effect=actual),patch.object(h.c,'state',side_effect=lambda ctx,hints:seen_hints.append(hints)or bad),patch.object(h,'save_observation',side_effect=saved),patch.object(h,'_source',side_effect=source),patch.object(h.visual_memory,'shape_displacement',return_value=.028),patch.object(h.c,'part_hand_gap',return_value=.014),patch.object(h,'_retained_receiver_scene_check',side_effect=retained):
    updated,bound,record=h._fresh_sender(Ctx(),deepcopy(state),1)
check('Conflicting actual RGB is retained as raw evidence, never installed as owned source',updated['observation']==bad and updated['parameters']['owned_held_belief_source']==old_source)
check('Final receiver visual conflict keeps original8mm dated conditional control',bound is old and record['status']=='RGB_CONFLICT_OLD_BOUND_RETAINED' and record['current_target_measured']is False)
check('Retained conflict route requires full current scene validation',calls==[('retained_full_scene',1)])
check('Dated board is only a current-fit hint and never fills raw observation',seen_hints[0]['board']==state['rgb_history'][0]['poses']['board']and 'board'not in updated['observation']['poses'])

with patch.object(h,'_belief',side_effect=ValueError('outside bounded30s')),patch.object(h.base,'proprio',side_effect=actual),patch.object(h.c,'state',return_value=bad),patch.object(h,'save_observation',side_effect=saved),patch.object(h,'_source',side_effect=source),patch.object(h.visual_memory,'shape_displacement',return_value=.028),patch.object(h.c,'part_hand_gap',return_value=.014),patch.object(h,'_retained_receiver_scene_check',side_effect=AssertionError('Expired control cannot survive')):
    try:h._fresh_sender(Ctx(),deepcopy(state),1,optimizer_belief=old);raise AssertionError('Expected expired refusal')
    except h.StageRefusal:pass
check('Conflicting image cannot extend the old source TTL',True)

calls=[];transferred={**state,'handoff':{**state['handoff'],'stage':'transferred'}}
with patch.object(h,'_belief',return_value=old),patch.object(h.base,'proprio',side_effect=actual),patch.object(h.c,'state',return_value=bad),patch.object(h,'save_observation',side_effect=saved),patch.object(h,'_source',side_effect=source),patch.object(h.visual_memory,'shape_displacement',return_value=.028),patch.object(h.c,'part_hand_gap',return_value=.014),patch.object(h,'_retained_receiver_scene_check',side_effect=retained):
    out=h.receiver_bind(Ctx(),deepcopy(transferred))
rows=out['state']['handoff']['receiver_bind_record']['attempts']
check('Receiver conflict makes two actual observation attempts and rejects both control candidates',len(rows)==2 and all(r['status']=='RGB_CONFLICT_REJECTED_FOR_CONTROL'for r in rows))
check('Physical verified route labels visual UNKNOWN with unchanged source and no receipt',out['route']=='verified' and not out['state']['handoff']['receiver_bind_record']['current_target_measured']and out['state']['parameters']['owned_held_belief_source']==old_source and not out['state']['execution_tokens'])

scene={'board_for_planning':{'pose':[.45,0,.753,0,0,0,1],'uncertainty_m':.004}}
seen=[]
class G:
    def screen_q16_path(self,path,**kwargs):
        seen.append(kwargs);return {'safe':True,'rows':[{'groups':{k:{'clearance_m':.010}for k in('robot_board79','held_envelope_board79')}}]}
with patch.object(h,'_belief',return_value=old),patch.object(h.base,'proprio',side_effect=actual),patch.object(h,'scene_for_motion',side_effect=lambda ctx,s,b:(s,scene,{})),patch.object(h,'geometry',return_value=G()):
    _,b,r=h._retained_receiver_scene_check(Ctx(),deepcopy(state),1)
check('Retained current scene keeps8mm and grants material contact only to receiver',seen[0]['held_envelope_m']==.008 and seen[0]['finger_target_contract']=={1:[4]}and r['source_or_bound_widened']is False)

# Reuse original <=6mm/.03rad stableactual full9 twice, then original local
# hand guards. This permission is supplied only for outside/rotation points.
execute_calls=[];local=[]
def executed(ctx,s,motion,**kwargs):
    execute_calls.append(kwargs)
    return {'stable_actual_endpoint':dict(actual_initial_robot=actual(ctx,1),actual_robot=actual(ctx,1))}
with patch.object(m.handoff,'_execute',side_effect=executed),patch.object(m,'_hand_guard',side_effect=lambda *a,**k:local.append(a)or{'safe':True}):
    ex=m._drive(Ctx(),deepcopy(state),{'terminal_contract':False},intermediate=True,scene=scene,belief=old)
check('Only declared intermediate passes reviewed stableactual goal purpose',execute_calls[0]['stable_actual_handoff']['purpose']=='intermediate_goal_carry' and execute_calls[0]['retained_arm']==1)
check('Both stable actual q epochs retain original local hand checks',len(local)==2 and ex['stable_actual_endpoint']['fresh_RGB_required_before_next_motion'])
try:m._drive(Ctx(),state,{'terminal_contract':True},intermediate=True,scene=scene,belief=old);raise AssertionError('Expected terminal adaptation refusal')
except m.StageRefusal:pass
check('Terminal motion cannot request intermediate adaptation',True)

terminal_state={**state,'last_outcome':'carried'}
for refresh,b in (({'attempts':[{'status':'RGB_CONFLICT_OLD_BOUND_RETAINED'}]},old),({'attempts':[{'status':'actual_RGB_q16_rebound'}]},old)):
    with patch.object(m,'_refresh',return_value=(terminal_state,b,scene,refresh)),patch.object(m,'_plan',side_effect=AssertionError('Terminal movement must not start')):
        try:m.release(Ctx(),terminal_state);raise AssertionError('Expected terminal fresh6 refusal')
        except m.StageRefusal:pass
check('Neither visual conflict nor8mm fresh source may authorize terminal descent',True)
report=dict(status='PASS',cases=len(checks),checks=checks,
    scope='Source lifecycle/control ordering only; prior remains conditional and original independent physical checkpoints are required')
report['packaged_source']=str(Path(m.__file__).resolve())
(ROOT/'verification/red_recovery_contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
