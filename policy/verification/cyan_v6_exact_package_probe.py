"""Independent API/actual-handler review; no Env, render, GPU or native motors."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT=Path('/srv/home/bohanlyu/gap-puzzler-web')
WORK=Path(__file__).resolve().parents[1]
PATCH=WORK/'puzzler'
sys.path.insert(0,str(WORK))
sys.path.insert(0,str(ROOT/'third_party/graph-as-policy/gap-core/src'))
import puzzler
puzzler.__path__.insert(0,str(PATCH))
from puzzler import cyan_receiver_release_feedback_delivery as d
from puzzler import cyan_receiver_motion_feedback as m

checks={}
def check(name,value):checks[name]=bool(value)
def refused(call):
    try:call()
    except (RuntimeError,ValueError,AssertionError):return True
    return False

# Real module import and explicit API eligibility guards execute actual code.
for terminal,joint,fraction in ((True,True,.5),(False,False,.5),(False,True,1.),(False,True,0.)):
    check('intermediate_permission_phase_guard_'+str((terminal,joint,fraction)),refused(lambda:m.move_held(
        None,{},[0,0,1],dict(x=0,y=0,z=0,w=1),{},.024,terminal=terminal,joint=joint,
        path_fraction=fraction,allow_actual_intermediate=True)))

tree=ast.parse((PATCH/'cyan_receiver_motion_feedback.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='move_held')
handler=next(n for n in fn.body if isinstance(n,ast.Try)).handlers[0]
args=['ctx','state','arm','off','scene','allow_actual_intermediate','error']
probe=ast.FunctionDef(name='_review_handler',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n)for n in args],kwonlyargs=[],kw_defaults=[],defaults=[]),
    body=[ast.Assign(targets=[ast.Name(id='feedback',ctx=ast.Store())],value=ast.Constant(None)),
          ast.Try(body=[ast.Raise(exc=ast.Name(id='error',ctx=ast.Load()))],handlers=[deepcopy(handler)],orelse=[],finalbody=[]),
          ast.Return(value=ast.Name(id='feedback',ctx=ast.Load()))],decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[probe],type_ignores=[])),'actual_motion_handler','exec'),m.__dict__)
proof=lambda:dict(safe=True,rows=[dict(groups={n:dict(clearance_m=.02)for n in
    ('robot_board79','moving_robot_table','held_envelope_board79')})])
belief=dict(predicted_body_pose=[.5,.1,.9,0,0,0,1],current_robot_encoder_values=[0.]*16,
    object_in_tcp_position=[0,0,.014],object_in_tcp_rotation=np.eye(3).tolist(),
    current_tcp=[.5,.1,.886,0,0,0,1],uncertainty_m=.006)
belief['current_robot_encoder_values'][14:]=[.012,-.012]
state=dict(receiver_delivery=dict(selected_receiver_bead=2))
scene=dict(board_for_planning=dict(uncertainty_m=.004))
class Context:
    def __init__(self):self.calls=[]
    def tool(self,name,**kwargs):
        self.calls.append((name,kwargs))
        assert name=='robot.wait_steps' and kwargs==dict(steps=6)
        return dict(waited=6)
class Geometry:
    def __init__(self,kind):self.kind=kind;self.calls=[];self.safe=True
    def screen_q16_path(self,rows,scene,**kw):
        self.calls.append(kw);assert np.asarray(rows).shape==(1,16)
        assert kw['held_envelope_m']==.006 and kw['finger_target_contract']=={1:[2]}
        return dict(proof(),safe=self.safe)
    def screen_robot_material_path(self,rows,scene,**kw):
        self.calls.append(kw);assert np.asarray(rows).shape==(1,16)
        assert np.asarray(kw['target_body_poses']).shape==(7,)
        assert kw['finger_target_contract']=={1:[2]}
        return dict(proof(),safe=self.safe)

full=Geometry('full');material=Geometry('material')
m._geometry=lambda:full;m._material_geometry=lambda:material
m.finite_clearance=lambda *a,**kw:dict(clearance_m=.01)
poses=[np.array([.5,.1,.886]),np.array([.5,.1,.886])]
q=dict(x=0.,y=0.,z=0.,w=1.)
def run_handler(pe=.003013,re=.007095,allow=True,uncertainty=.006,drift=0.):
    sequence=[deepcopy(belief),deepcopy(belief)]
    for b in sequence:b['uncertainty_m']=uncertainty
    m.tr._belief=lambda ctx,state:sequence.pop(0)
    hands=[(poses[0].copy(),q),(poses[1]+[drift,0,0],q)]
    m.c.hand=lambda ctx,arm:hands.pop(0)
    error=m.c.MotorAccuracyError('test',np.array([.5,.1,.886]),q,pe,re)
    ctx=Context()
    result=m._review_handler(ctx,state,1,8,scene,allow,error)
    return result,ctx

result,ctx=run_handler()
check('actual_handler_positive',result['original_partial_waypoint_reached']is False and
      result['terminal_goal_reached']is False and result['fresh_actual_RGB_required_before_replanning']is True)
check('actual_handler_wait6',len(ctx.calls)==1)
check('actual_handler_both_complete_and_material_calls',len(full.calls)==2 and len(material.calls)==2)
for name,kw in [('permission_false',dict(allow=False)),('pe_over6',dict(pe=.00601)),
    ('rotation_over03',dict(re=.03001)),('drift_over04',dict(drift=.000401)),
    ('actual_uncertainty_over6',dict(uncertainty=.00601)),('actual_uncertainty_nan',dict(uncertainty=float('nan')))]:
    check('actual_handler_reject_'+name,refused(lambda kw=kw:run_handler(**kw)))
full.safe=False;check('actual_handler_complete_geometry_failure',refused(run_handler));full.safe=True
material.safe=False;check('actual_handler_material_geometry_failure',refused(run_handler));material.safe=True

# Direct released-body API; all fabricated second frames are explicit mocks,
# never claimed as native/actual evidence. Actual first112.52 packet is used.
base=ROOT/'runs/cyan_vertical_handoff_native_v4/actual_receiver_release_failure'
first=json.loads((base/'latest_actual_RGB.json').read_text())
assert first['timestamp']==112.52 and {'board','cyan'}.issubset(first['poses'])
original_empty=d._refresh_empty_scene
scene_events=[]
def fake_refresh(ctx,state,packet):
    scene_events.append(packet['timestamp']);return {**state,'planning_scene':{}},{}
d._refresh_empty_scene=fake_refresh
original_goal=d.visual_memory.goal_error
def release_case(missing_first=False,missing_second=False,missing_board=False,interval=1.2,drift=0.,goal=.001,fallback_shift=0.):
    a=deepcopy(first);b=deepcopy(first);b['timestamp']+=interval
    b['poses']['cyan'][0]+=drift
    if missing_first:a['poses'].pop('cyan')
    if missing_second:b['poses'].pop('cyan')
    if missing_board:b['poses'].pop('board')
    d.c.state=lambda ctx:b
    d.visual_memory.goal_error=lambda *args:goal
    class C:
        def __init__(self):self.calls=[]
        def tool(self,name,**kw):
            self.calls.append((name,kw));assert name=='robot.wait_steps'and kw==dict(steps=60)
    fallback=np.asarray(first['poses']['cyan']).copy();fallback[2]+=fallback_shift
    ctx=C();out=d._released_body(ctx,{'parameters':{}},a,fallback,.016)
    return out,ctx
out,ctx=release_case()
check('released_actual_first_timestamp_preserved',first['timestamp']==112.52)
check('released_stable_pair_bound6_plus_goal',abs(out[3]-.007)<1e-12)
check('released_pair_not_receipt',out[4]['placement_receipt_minted']is False)
check('released_wait60',ctx.calls==[('robot.wait_steps',dict(steps=60))])
for name,kw in [('first_unknown',dict(missing_first=True)),('second_unknown',dict(missing_second=True)),
    ('board_unknown',dict(missing_board=True)),('interval_short',dict(interval=.98)),
    ('displacement_over15',dict(drift=.0016)),('goal_over3',dict(goal=.0031)),('goal_nan',dict(goal=float('nan')))]:
    out,_=release_case(**kw);check('released_preserve_old16_'+name,out[3]==.016)
fallback_case,_=release_case(missing_second=True,fallback_shift=.010)
check('released_unknown_keeps_preopen_reference',abs(fallback_case[2][2]-first['poses']['cyan'][2]-.010)<1e-12)
d._refresh_empty_scene=original_empty;d.visual_memory.goal_error=original_goal

# Execute the ACTUAL postretreat reference-union block, not a copied formula.
dtree=ast.parse((PATCH/'cyan_receiver_release_feedback_delivery.py').read_text())
rf=next(n for n in dtree.body if isinstance(n,ast.FunctionDef)and n.name=='_segment_conditional_release')
union_node=next(n for n in rf.body if isinstance(n,ast.If)and
    any(isinstance(z,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='released_bound'for t in z.targets)for z in ast.walk(n)))
pargs=['body','released_bound','current','target','state','released_evidence']
pf=ast.FunctionDef(name='_review_postretreat_union',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n)for n in pargs],kwonlyargs=[],kw_defaults=[],defaults=[]),
    body=[deepcopy(union_node),ast.Return(value=ast.Tuple(elts=[ast.Name(id=n,ctx=ast.Load())for n in ('body','released_bound','released_evidence')],ctx=ast.Load()))],decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[pf],type_ignores=[])),'actual_postretreat_union','exec'),d.__dict__)
anchor=np.asarray(first['poses']['cyan']).copy()
def union_case(delta=0.001,board=True):
    current=deepcopy(first);current['poses']['cyan'][0]+=delta
    if not board:current['poses'].pop('board')
    d.visual_memory.goal_error=lambda *args:.002
    return d._review_postretreat_union(anchor.copy(),.007,current,{'height':.010},{'parameters':{}},{})
body,bound,evidence=union_case()
check('postretreat_reference_center_preserved',np.array_equal(body,anchor))
check('postretreat_bound_union_enlarges',abs(bound-.009)<1e-10)
check('postretreat_over6mm_contradiction_refused',refused(lambda:union_case(delta=.00601)))
_,bound,_=union_case(board=False)
check('postretreat_unknown_board_retains_full_drop',bound>=.017-1e-10)
d.visual_memory.goal_error=original_goal

# Entire transport function with tool/motion primitives mocked: recovery must
# obtain fresh actual Cyan and replan, and only the first partial leg opts in.
def transport_case(middle_measured):
    events=[];count=[0]
    start=dict(piece='cyan',arm=1,transfer=dict(sender=0,receiver=1),plan={},parameters={},
               receiver_delivery=dict(selected_receiver_bead=2))
    bb=deepcopy(belief)
    def refresh(ctx,state):
        count[0]+=1;events.append('refresh'+str(count[0]))
        measured=not(count[0]==3 and not middle_measured)
        observed=dict(timestamp=float(count[0]),poses={'cyan':bb['predicted_body_pose']}if measured else{})
        ss=dict(reference=dict(reference_timestamp=float(count[0])))
        return state,observed,ss,bb,{'refresh':count[0]}
    target=dict(hover=[.5,.1,.9],rotation=q,relative_target_R=np.eye(3).tolist(),
                authored_bound_estimate_public_hover_error_m=0.)
    moves=[]
    def motion(*args,**kwargs):
        events.append('move'+str(len(moves)+1));moves.append(kwargs)
        return dict(actual_intermediate_feedback={'qualified':True}if len(moves)==1 else None)
    d._refresh=refresh;d._home_donor_while_held=lambda *args:events.append('home')
    d._select_target=lambda *args:target;d._hover=lambda *args:target;d.move_held=motion
    d.result=lambda route,state,observed,phase,**kw:dict(route=route)
    class C:
        def tool(self,name,**kw):assert name=='robot.wait_steps'and kw==dict(steps=6)
    try:out=d._execute_transport(C(),start);error=None
    except d.c.PlanningRefusal as exc:out=None;error=str(exc)
    return out,error,events,moves
out,error,events,moves=transport_case(True)
check('full_transport_recovered_middle_fresh_then_replan',out['route']=='carried'and events.index('refresh3')<events.index('move2'))
check('full_transport_terminal_not_opted_in',moves[0].get('allow_actual_intermediate')is True and 'allow_actual_intermediate'not in moves[1])
out,error,events,moves=transport_case(False)
check('full_transport_missing_current_Cyan_refuses_before_next_motor',out is None and len(moves)==1 and 'fresh actual Cyan RGB'in error)

# AST invariants for unchanged terminal precision and exact new entry points.
check('terminal_default_accuracy_still2_5mm',any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)
    and n.func.attr=='await_motion'and any(k.arg=='accuracy'and isinstance(k.value,ast.Constant)and k.value.value==.0025 for k in n.keywords)
    for n in ast.walk(fn)))
enabled=[n for n in ast.walk(dtree)if isinstance(n,ast.Call)and any(k.arg=='allow_actual_intermediate'for k in n.keywords)]
check('only_one_explicit_actual_intermediate_call',len(enabled)==1)
check('explicit_intermediate_is_half_joint',all(any(k.arg=='path_fraction'and isinstance(k.value,ast.Constant)and k.value.value==.5 for k in n.keywords)for n in enabled))
release_fn=next(n for n in dtree.body if isinstance(n,ast.FunctionDef)and n.name=='_segment_conditional_release')
opening_index=next(i for i,n in enumerate(release_fn.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='opening'for t in n.targets))
post=ast.Module(body=release_fn.body[opening_index+1:],type_ignores=[])
check('no_expired_held_belief_after_open',not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='_belief'for n in ast.walk(post)))
report=dict(status='PASS'if all(checks.values())else'BLOCKED',checks=checks,
    scope='Actual imports and actual extracted handler/direct helper mocks; no Env/render/GPU/native motor/physics success',
    failed=[name for name,value in checks.items()if not value])
outpath=WORK/'verification/cyan_v6_exact_package_review.json'
outpath.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'failed':report['failed']}))
