"""Only new bounded goal feedback contracts; no Env, physics or motor."""
from pathlib import Path
import importlib.util,json,sys
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import puzzler
assert Path(puzzler.__file__).resolve().parent == ROOT/'puzzler'
from puzzler import red_bimanual_goal as m
checks=[]
def check(name,predicate):
 if not predicate:raise AssertionError(name)
 checks.append(name)

state={'piece':'red','arm':1};belief={'uncertainty_m':.008};scene={};fresh={'attempts':[{'status':'actual_RGB_q16_rebound'}]}
unknown={'attempts':[{'status':'RGB_UNKNOWN'}]}
def error(n,translation=(.008,0,0),rot=0.,ready=False):
 return {'bead_error_m':n,'actual_body_estimate':[*np.asarray(translation).tolist(),0,0,np.sin(rot/2),np.cos(rot/2)],
         'transport_six_mm_predicate':n<=.006,'ready_for_original_terminal':ready}
plans=[];drives=[]
def plan(ctx,state,scene,belief,ways,terminal=False):
 plans.append({'ways':ways,'terminal':terminal});return {'planned':len(plans)}
def drive(ctx,state,motion):drives.append(motion);return {'executed':motion}
with patch.object(m.base,'proprio',return_value={}),patch.object(m,'_hover_error',side_effect=[error(.008,rot=.2),error(.008,rot=.2),error(.0004,ready=True)]),patch.object(m,'_goal',return_value=(None,np.array([0.,0.,-.036]),np.eye(3),None,None)),patch.object(m,'_body_waypoint',side_effect=lambda p,r,b:{'position':np.asarray(p).tolist(),'rotation_matrix':np.asarray(r).tolist()}),patch.object(m,'_plan',side_effect=plan),patch.object(m,'_drive',side_effect=drive),patch.object(m,'_refresh',return_value=(state,belief,scene,fresh)):
 _,_,_,record=m._align_goal_hover(None,state,belief,scene,fresh,label='unit')
check('Changed attachment first replans exact unchanged goal',record['corrections'][0]['proposal']['strategy']=='new_actual_attachment_exact_goal' and np.allclose(plans[0]['ways'][0]['position'],[0,0,0]))
second=record['corrections'][1]['proposal']
check('Nonimproving residual switches to explicit screened command feedback',second['strategy']=='explicit_bounded_command_feedback')
check('Command translation compensation bounded at six millimetres',abs(np.linalg.norm(second['translation_compensation_m'])-.006)<1e-12)
check('Command rotation compensation bounded at .06 rad',abs(np.linalg.norm(second['rotation_compensation_rotvec_rad'])-.06)<1e-12)
check('Every correction uses original full normal path screen',len(plans)==len(drives)==2 and all(row['terminal'] is False for row in plans))
check('True goal remains unchanged and proposal never called measurement',all(c['proposal']['goal_is_unchanged'] and c['proposal']['proposal_is_measurement'] is False for c in record['corrections']))
check('Feedback returns only after authentic refreshed low residual',record['status']=='actual_RGB_goal_hover_aligned')
with patch.object(m,'_plan',side_effect=AssertionError('UNKNOWN cannot trigger correction')):
 _,_,_,r=m._align_goal_hover(None,state,belief,scene,unknown,label='unknown')
check('Unknown view never invents target error or a motion',r['status']=='visual_unknown_no_claim' and r['corrections']==[])
count=[]
with patch.object(m.base,'proprio',return_value={}),patch.object(m,'_hover_error',return_value=error(.007)),patch.object(m,'_goal',return_value=(None,np.array([0.,0.,-.036]),np.eye(3),None,None)),patch.object(m,'_body_waypoint',return_value={}),patch.object(m,'_plan',side_effect=lambda *a,**k:count.append(1) or {}),patch.object(m,'_drive',return_value={}),patch.object(m,'_refresh',return_value=(state,belief,scene,fresh)):
 try:m._align_goal_hover(None,state,belief,scene,fresh,label='bounded')
 except m.StageRefusal:pass
 else:raise AssertionError('Known bad target accepted')
check('Persisting actual error refuses after exactly three corrections',len(count)==3)
with patch.object(m.base,'proprio',return_value={}),patch.object(m,'_hover_error',return_value=error(.004)),patch.object(m,'_goal',return_value=(None,np.array([0.,0.,-.036]),np.eye(3),None,None)),patch.object(m,'_body_waypoint',return_value={}),patch.object(m,'_plan',return_value={}),patch.object(m,'_drive',return_value={}),patch.object(m,'_refresh',return_value=(state,belief,scene,fresh)):
 try:m._align_goal_hover(None,state,belief,scene,fresh,label='axis',require_terminal_axis=True)
 except m.StageRefusal:checks.append('Terminal alignment never widens original one-millimetre axis contract')
 else:raise AssertionError('Misaligned terminal permitted')
report={'status':'PASS','cases':len(checks),'checks':checks,'scope':'New bounded feedback code only; tool/motion mocks, no physical success or solver feasibility claim'}
report['packaged_source']=str(Path(m.__file__).resolve())
(ROOT/'verification/red_goal_feedback_contract.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
