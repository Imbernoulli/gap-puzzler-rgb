"""Actual Cyan feedback/TTL/staged close/release APIs, offline fixtures only."""
import ast
from copy import deepcopy
import hashlib,json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path('/srv/home/bohanlyu/gap-puzzler-web');WORKSPACE=Path(__file__).resolve().parents[1];HERE=WORKSPACE/'verification'
PATCH=WORKSPACE/'puzzler'
sys.path[:0]=[str(WORKSPACE),str(ROOT/'third_party/graph-as-policy-runtime/gap-core/src')]
import puzzler;puzzler.__path__.insert(0,str(PATCH))
from puzzler import cyan_vertical_transfer_v2_adapter as m
checks={};events=[]
board=[.45,0,.753,0,0,0,1]
def state():
    measurement=dict(timestamp=10.,frame_sha256={'top':'EXPLICIT_OFFLINE_PACKET'},observation_source='rgb_shape_geometry',poses={'board':board})
    return dict(arm=0,piece='cyan',planning_scene=dict(measurement=deepcopy(measurement),occupancy={name:dict(inflation_m=0.)for name in m.model.CELLS}),
        parameters=dict(historical_complete_rgb=deepcopy(measurement),historical_max_age_s=30.,source_contact_bead_indices=[0],owned_held_belief_source={'old':'EXPLICIT_SOURCE'}),
        transfer=dict(sender=0,receiver=1,candidate=dict(bead_index=2,bead_in_tcp_m=[0.,0.,.014],permitted_distal_bead_indices=[2]),execution=[],preclose_source={'timestamp':10.},receiver_conditional_attachment=dict(attachment_id='EXPLICIT_RECEIVER_ATTACHMENT',close_proprio_epoch={'timestamp':20.})))
class Context:
    arbitrary='EXPLICIT_CONTEXT_FIELD'
    def __init__(self,now=20.,width=.024):self.now=now;self.width=width;self.calls=[]
    def tool(self,name,**kwargs):
        self.calls.append((name,deepcopy(kwargs)))
        if name=='robot.execution_journal_cursor':return dict(timestamp=self.now,call_id=len(self.calls),session_nonce='EXPLICIT_OFFLINE_NONCE')
        if name in ('puzzle.propagate_pyramid_held_belief','puzzle.propagate_conditional_attachment'):
            q=np.zeros(16);q[6:8]=[self.width/2,-self.width/2]
            return dict(current_robot_encoder_values=q.tolist())
        if name=='puzzle.capture_conditional_attachment':return dict(attachment_id='EXPLICIT_NEW_ATTACHMENT',close_proprio_epoch=dict(timestamp=self.now,actual_jaw_width_m=.022))
        return {'unchanged':'EXPLICIT_RETURN'}
def accepts(fn):
    try:fn();return True
    except (m.c.PlanningRefusal,ValueError):return False
for name in ('robot.execute_trajectory','robot.set_grip'):
    for age,ok in ((0.,True),(30.,True),(30.0001,False),(-.0001,False)):
        raw=Context(now=10.+age);wrapped=m._VerticalSourceContext(raw,state())
        result=accepts(lambda:wrapped.tool(name,arm_id=1,width_m=.03))
        checks[f'{name}_age_{age}']=result==ok and (ok or all(call[0]!=name for call in raw.calls))
for name in ('puzzle.propagate_pyramid_held_belief','puzzle.propagate_conditional_attachment'):
    for width,ok in ((.019999,False),(.020,True),(.024,True)):
        wrapped=m._VerticalSourceContext(Context(width=width),state())
        checks[f'{name}_donor_jaw_{width}']=accepts(lambda:wrapped.tool(name,arm_id=0))==ok
raw=Context(width=.001);wrapped=m._VerticalSourceContext(raw,state())
checks['receiver_propagation_not_donor_gate']=accepts(lambda:wrapped.tool('puzzle.propagate_conditional_attachment',arm_id=1))
s=state();raw=Context(now=41.);wrapped=m._VerticalSourceContext(raw,s)
s['planning_scene']['measurement']['timestamp']=40.;s['parameters']['historical_complete_rgb']['timestamp']=40.
checks['deepcopy_state_change_cannot_renew_bound_scene']=not accepts(lambda:wrapped.tool('robot.set_grip',arm_id=1,width_m=.03))
wrapped.bind_transfer_scene(s)
checks['explicit_new_complete_epoch_rebinds_scene']=accepts(lambda:wrapped.tool('robot.set_grip',arm_id=1,width_m=.03))
for key in ('timestamp','frame_sha256','poses'):
    s=state();s['parameters']['historical_complete_rgb'][key]=99. if key=='timestamp'else {'top':'OTHER'}if key=='frame_sha256'else {'board':[0]*7}
    checks['scene_historical_mismatch_'+key]=not accepts(lambda:m._VerticalSourceContext(Context(),s))
for limit,age,ok in ((60.,30.1,False),(10.,10.1,False),(10.,10.,True)):
    s=state();s['parameters']['historical_max_age_s']=limit
    checks[f'original_or_stricter_TTL_{limit}_{age}']=accepts(lambda:m._VerticalSourceContext(Context(now=10+age),s).tool('robot.set_grip',arm_id=1,width_m=.03))==ok

# Center a selected real CAD bead against the actual open-hand FK. The
# fixture makes all three correction components nonzero.
beads=m.model.local_beads('cyan');body=np.array([0,0,1,0,0,0,1.]);point=body[:3]+beads[2]
def center_case(delta,width=.055,depth=.014):
    s=state();s['transfer']['candidate']['bead_in_tcp_m'][2]=depth
    q=np.zeros(16);q[14:16]=[width/2,-width/2]
    belief=dict(current_robot_encoder_values=q.tolist(),predicted_body_pose=body.tolist())
    tcp=point-np.array([0.,0.,depth])-delta;calls=[]
    with patch.object(m.c,'hand',return_value=(tcp,dict(x=0.,y=0.,z=0.,w=1.))),patch.object(m.transfer,'_execute',side_effect=lambda *args,**kwargs:calls.append((args,kwargs))or{}),patch.object(m.c,'await_motion',side_effect=lambda *args,**kwargs:calls.append(('await',kwargs))),patch.object(m.transfer,'_belief',return_value=belief),patch.object(m.transfer,'_guard',side_effect=lambda *args,**kwargs:calls.append(('guard',kwargs))or{}):
        try:out=m._center_open_receiver(None,s,belief);accepted=True
        except m.c.PlanningRefusal:out=None;accepted=False
    return accepted,out,calls,tcp
ok,out,calls,tcp=center_case(np.array([.004,-.002,.001]))
checks['open_center_moves_all_three_actual_FK_components']=ok and np.allclose(out['correction_world_m'],[.004,-.002,.001]) and np.allclose(calls[0][0][3][0][0],tcp+[.004,-.002,.001])
checks['open_center_has_no_receiver_contact_exemption']=ok and calls[0][1]['receiver_contact']is False and calls[-1][1]['receiver_contact']is False
checks['open_center_submillimeter_actual_endpoint']=ok and calls[1]==('await',{'accuracy':.0007})
checks['open_center_over_six_mm_refused']=not center_case(np.array([.0061,0,0]))[0]
checks['open_center_closed_hand_refused']=not center_case(np.zeros(3),width=.0449)[0]
checks['open_center_invalid_public_depth_refused']=not center_case(np.zeros(3),depth=.017)[0]

# Exercise the real close controller. Mocks isolate source refresh ordering;
# original _grip's complete sweep and endpoint guards are separately unchanged.
stage_events=[];refreshes=0
def refresh(ctx,s):
    global refreshes
    refreshes+=1;s['transfer']['preclose_source']={'timestamp':float(refreshes),'frame_sha256':{'top':str(refreshes)}}
    stage_events.append(('refresh',refreshes));return s,{},dict(predicted_body_pose=body.tolist()),{'epoch':refreshes}
def grip(ctx,s,arm,command,**kwargs):
    stage_events.append(('grip',command,kwargs,s['transfer']['preclose_source']['timestamp']));return {}
raw=Context()
with patch.object(m,'_refresh_close',side_effect=refresh),patch.object(m,'_center_open_receiver',side_effect=lambda *args:stage_events.append(('center',))or{}),patch.object(m.transfer,'_grip',side_effect=grip),patch.object(m.transfer,'_belief',return_value={'predicted_body_pose':body.tolist()}),patch.object(m.c,'state',return_value={}),patch.object(m.transfer,'_out',side_effect=lambda route,s,o,**kw:dict(route=route,state=s,evidence=kw)):
    out=m.close_receiver(raw,state())
stages=[row for row in stage_events if row[0]=='grip']
checks['actual_four_stage_commands_and_loaded_screen_widths']=[row[1]for row in stages]==[.030,.026,.0242,0.] and [row[2]['loaded_screen_width']for row in stages]==[.030,.026,.0242,.022]
checks['fresh_rebind_precedes_every_next_stage']=stage_events[0][0]=='refresh' and stage_events[1][0]=='center' and all(stage_events[stage_events.index(row)-1][0]=='refresh'for row in stages)
capture=next(row for row in raw.calls if row[0]=='puzzle.capture_conditional_attachment')
checks['actual_close_uses_latest_dated_actual_source_and_original_six_mm']=capture[1]['source_rgb']['timestamp']==5. and capture[1]['assumption']['bead_displacement_bound_m']==.006 and out['route']=='receiver_closed'
checks['distinct_actual_close_epoch_preserved']=out['state']['transfer']['actual_receiver_close_proprio']['timestamp']==20.

# Invoke the new receiver-only guard without building public GPU geometry.
guard_calls=[]
class Geometry:
    def screen_q16_path(self,**kwargs):guard_calls.append(kwargs);return {'safe':True}
s=state();s['arm']=1;s['transfer']['donor_contact_revoked']=True
b=dict(predicted_body_pose=body.tolist(),object_in_tcp_position=[0,0,0],object_in_tcp_rotation=np.eye(3).tolist())
with patch.object(m,'_geometry',return_value=Geometry()):m._release_guard(s,[np.zeros(16)],0,b)
checks['release_guard_only_receiver_selected_bead_permission']=guard_calls[-1]['finger_target_contract']=={1:[2]} and guard_calls[-1]['held_arm']==1 and guard_calls[-1]['held_envelope_m']==.006
s['transfer']['donor_contact_revoked']=False
with patch.object(m,'_geometry',return_value=Geometry()):checks['release_guard_rejects_unrevoked_sender']=not accepts(lambda:m._release_guard(s,[np.zeros(16)],0,b))

release_events=[]
def release_grip(ctx,s,arm,width,**kwargs):release_events.append(('initial_open',width,s['arm'],s['parameters']['conditional_attachment_id']));return{}
def release_guard(s,*args):release_events.append(('guard',s['transfer'].get('donor_contact_revoked')));return{}
def release_exec(ctx,s,arm,p,q,phase):release_events.append(('move',phase,np.asarray(p).tolist(),s['transfer'].get('donor_contact_revoked')));return{}
def later_grip(ctx,s,arm,width):release_events.append(('later_open',width,s['transfer'].get('donor_contact_revoked')));return{}
s=state();before=deepcopy(s)
with patch.object(m.transfer,'_grip',side_effect=release_grip),patch.object(m,'_release_guard',side_effect=release_guard),patch.object(m,'_release_execute',side_effect=release_exec),patch.object(m,'_release_grip',side_effect=later_grip),patch.object(m.transfer,'_belief',return_value={'current_robot_encoder_values':[0]*16,'predicted_body_pose':body.tolist()}),patch.object(m.c,'hand',return_value=(np.array([.4,.2,1.]),dict(x=0.,y=0.,z=0.,w=1.))),patch.object(m.c,'state',return_value={}),patch.object(m.transfer,'_out',side_effect=lambda route,s,o,**kw:dict(route=route,state=s)):
    released=m.release_donor(Context(),s)
checks['release_switches_to_receiver_attachment_before_open']=release_events[0]==('initial_open',.030,1,'EXPLICIT_RECEIVER_ATTACHMENT')
checks['release_revokes_sender_contact_before_each_withdrawal']=release_events[1]==('guard',True) and all(row[-1]is True for row in release_events if row[0]in ('move','later_open'))
checks['release_public_relative_short_then_clear_proposals']=np.allclose(release_events[2][2],[.4,.2,.985]) and release_events[3][:2]==('later_open',.055) and np.allclose(release_events[4][2],[.34,.2,1.04])
checks['release_input_immutable_oldsource_no_longer_active']=s==before and 'owned_held_belief_source'not in released['state']['parameters'] and released['state']['transfer']['old_sender_source_active']is False
old=ast.parse((PATCH/'cyan_bimanual_near_stream_adapter.py').read_text());new=ast.parse((PATCH/'cyan_bimanual_feedback_adapter.py').read_text())
find=lambda tree,name:next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
checks['stream_dense_line_motor_batch_logic_AST_unchanged']=ast.dump(find(old,'_execute_joint_pre'))==ast.dump(find(new,'_execute_joint_pre'))
checks['original_full_grip_sweep_and_endpoint_function_reused']=m.transfer._grip.__module__=='puzzler.cyan_bimanual_adapter'
report=dict(schema='puzzler.cyan_feedback_v2.packaged_checks.v1',all_passed=all(checks.values()),checks=checks,
    source_sha256={name:hashlib.sha256((PATCH/name).read_bytes()).hexdigest()for name in ('cyan_bimanual_feedback_adapter.py','cyan_vertical_transfer_v2_adapter.py')},
    scope='Actual import/API fixtures: motor-side actual journal TTL and deepcopy rebind, donor jaw, actual openFK3D centering, staged fresh close ordering/latest dated source, receiver-only release permission/order; unchanged streaming AST/full _grip reuse. Guards/motors mocked; no Env/FK/GPU/native/solver or physical success claim. Separate physical CPs remain required.',native_success_claim=False)
(HERE/'cyan_feedback_v2_packaged_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
