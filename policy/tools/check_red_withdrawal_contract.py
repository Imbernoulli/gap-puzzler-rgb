"""New source lifecycle and opened-hand control contracts; no physical claim."""
from pathlib import Path
import json,sys
from copy import deepcopy
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import puzzler
assert Path(puzzler.__file__).resolve().parent == ROOT/'puzzler'
from puzzler import red_bimanual_handoff as h
checks=[]
def check(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)

belief=dict(object_in_tcp_position=[0,0,0],object_in_tcp_rotation=np.eye(3).tolist(),uncertainty_m=.008)
state=dict(piece='red',arm=1,parameters={},observation=dict(timestamp=1.,frame_sha256={'top':'actual-test'}),handoff=dict(sender=0,receiver=1,
    stage='dual_held',hop=2,contract={0:[1],1:[4]},receiver_attachment=dict(kind='actual_RGB_q16',source={'dated':True})))
events=[]
class Ctx:
    def tool(self,name,**kwargs):
        events.append((name,kwargs))
        return dict(status='ok')
def actual(ctx,arm):
    return dict(tcp=[.4,0,.95,0,0,0,1],width_m=.055 if arm==0 else .024,robot_encoder_values=[0]*16)

families=h._withdrawal_families(actual(None,0),0,[.4,0,.95,0,0,0,1])
check('Second-hop first proposal extracts20mm then turns upward65mm',families[0][0]=='local_back_20_then_up65' and
      np.allclose(families[0][1][0]['position'],[.4,0,.93]) and np.allclose(families[0][1][1]['position'],[.4,0,.995]))

def screen(ctx,s,scene,b,held,moving,ways,contract,**kwargs):
    events.append(('screen',deepcopy(contract)))
    check('Empty sender has no old selected-bead contact permission',contract=={1:[4]})
    check('Withdraw keeps original8mm held uncertainty',h._held_envelope(b)==.008)
    return {'endpoint':ways[-1]}
def execute(ctx,s,motion,**kwargs):
    events.append(('execute',kwargs));return {'actual_after':actual(ctx,0)}
with patch.object(h.base,'proprio',side_effect=actual),patch.object(h,'_screen',side_effect=screen),patch.object(h,'_execute',side_effect=execute),patch.object(h,'_belief',return_value=belief),patch.object(h.c,'part_hand_gap',return_value=.08):
    out=h._withdraw_open_sender(Ctx(),deepcopy(state),belief,{})
check('Entire fresh withdrawal proposal is screened before first motor',events[0][0]=='screen' and events[1][0]=='execute')
check('Retained receiver is the execution retention arm',events[1][1]['retained_arm']==1)

events=[]
unknown=dict(poses={},timestamp=5.)
with patch.object(h.base,'proprio',side_effect=actual),patch.object(h.c,'state',return_value=unknown),patch.object(h,'save_observation',side_effect=lambda s,o:s),patch.object(h,'scene_for_motion',side_effect=lambda ctx,s,b:(s,{},{})),patch.object(h,'_rebind_sender_observation',side_effect=AssertionError('UNKNOWN must not bind')):
    _,unchanged,_,record=h._postopen_receiver_refresh(Ctx(),deepcopy(state),1,belief)
check('Two RGB-UNKNOWN observations cannot mint a new attachment',unchanged is belief and len(record['attempts'])==2 and all(not row['current_target_measured']for row in record['attempts']))
check('UNKNOWN preserves declared8mm and never sends a motion',record['held_envelope_m']==.008 and all(e[0]=='robot.wait_steps'for e in events))

events=[]
class Geometry:
    def screen_q16_path(self,*args,**kwargs):return {'safe':True}
def after_open(ctx,s,receiver,b):
    events.append(('fresh_after_open',receiver));return s,b,{},{}
def withdraw(ctx,s,b,scene):
    events.append(('withdraw',s['arm']));return dict(execution={},family='new_live_proposal',refusals=[])
with patch.object(h.base,'proprio',side_effect=actual),patch.object(h,'_belief',return_value=belief),patch.object(h,'scene_belief',return_value={}),patch.object(h,'geometry',return_value=Geometry()),patch.object(h,'_guard_passes',return_value=True),patch.object(h,'_postopen_receiver_refresh',side_effect=after_open),patch.object(h,'_withdraw_open_sender',side_effect=withdraw):
    out=h.sender_release(Ctx(),state)
check('Actual sender opening precedes new receiver RGB and withdrawal',[e[0]for e in events]==['robot.set_grip','fresh_after_open','withdraw'])
check('Formal transfer still needs preceding dual-held stage',out['route']=='transferred' and out['state']['handoff']['sender_old_source_revoked'])
try:
    h.sender_release(Ctx(),{**state,'handoff':{**state['handoff'],'stage':'prepared'}})
    raise AssertionError('Expected typed refusal')
except h.StageRefusal:pass
check('Prepared receiver cannot bypass independent dualhold checkpoint',True)

for contaminated in ({'owned_held_belief_source':{'old':True}}, {'conditional_attachment_id':'old'}):
    try:
        h.warm_prepare(Ctx(),{**state,'parameters':{'red_bimanual_warm_debug':True,**contaminated}})
        raise AssertionError('Expected stale warm source refusal')
    except h.StageRefusal:pass
check('Warm entry refuses old owned and conditional source epochs',True)
try:
    h.warm_withdraw(Ctx(),{**state,'parameters':{'red_bimanual_warm_debug':True}})
    raise AssertionError('Expected warm typed refusal')
except h.StageRefusal:pass
check('Warm tail cannot impersonate formal dual-held stage',True)

report=dict(status='PASS',cases=len(checks),checks=checks,
    scope='Offline observation/attachment/control-order contracts; no physical execution or earlier checkpoint earned')
report['packaged_source']=str(Path(h.__file__).resolve())
(ROOT/'verification/red_withdrawal_contract.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
