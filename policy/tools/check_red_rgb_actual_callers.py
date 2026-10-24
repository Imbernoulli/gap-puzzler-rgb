"""Saved-pixel VisualObserver replay through both actual candidate estimators."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from puzzler import model,vision,shape_vision
from puzzler.vision_observer import VisualObserver,goals
from puzzler.red_rgb_recovery import recover_missing_red
from puzzler.azure_rgb_attribution import refuse_unattributed_original_azure

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2]
SRC=REPO/'runs/red_bimanual_goal_runner_gpu_v6/workspace/runs/red_bimanual_goal_given_prefix11_gpu_v6/episode/rgb'
OUT=ROOT/'verification/red_rgb_actual_callers_replay.json'
rows=json.loads((SRC/'observations.json').read_text())
journal=[json.loads(line)for line in (SRC/'tools.jsonl').read_text().splitlines()]
calls=[row for row in journal if row.get('tool')=='vision.observe'and row.get('event')=='call']
cameras=vision.calibration(vision.rig_specs())
prefix=('azure','orange','yellow','teal','purple','cyan')
remaining=tuple(n for n in model.CELLS if n not in prefix)
checks={};records=[];negatives=[]

def extract(path):
    tree=ast.parse(path.read_text())
    return next(node for node in ast.walk(tree)if isinstance(node,ast.FunctionDef)and node.name=='estimate')

def estimator(relative,recover):
    node=extract(ROOT/relative)
    namespace=dict(estimate_shapes=shape_vision.estimate_shapes,model=model,np=np,Rotation=Rotation,
        fit_board=vision.fit_board,given=prefix,remaining=remaining,
        self=SimpleNamespace(visual=SimpleNamespace(targets=goals('pyramid','pyramid_robot6_plan.json'))),
        recover_missing_red=recover,refuse_unattributed_original_azure=refuse_unattributed_original_azure)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/relative),'exec'),namespace)
    return namespace['estimate']

def observe(function,images,index,hints,previous=None):
    class Capture:
        robot_geometry_provenance=deepcopy(rows[index]['robot_geometry_provenance'])
        def capture(self):return images,rows[index]['timestamp']
    observer=VisualObserver(Capture().capture,cameras,{},estimator=function)
    observer.previous=deepcopy(rows[index-1]['poses']if previous is None else previous)
    start=time.monotonic();result=observer.observe(hints=deepcopy(hints))
    return result,time.monotonic()-start

for label,relative in (('canonical','tools/unified_native.py'),('suffix','tools/conditional_suffix_native.py')):
    original=extract(ROOT/'source_archives/red_rgb_callers_baseline'/relative)
    current=deepcopy(extract(ROOT/relative))
    current.body=[node for node in current.body if not(isinstance(node,ast.Assign)and isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Name)and node.value.func.id in ('recover_missing_red','refuse_unattributed_original_azure'))]
    checks[label+'_preexisting_estimator_AST_preserved']=ast.dump(original,include_attributes=False)==ast.dump(current,include_attributes=False)
    before=estimator(relative,lambda strict,*a:strict)
    after=estimator(relative,recover_missing_red)
    positives=[]
    for index in (25,26):
        with np.load(SRC/f'{index:04}.npz')as archive:images={key:archive[key]for key in archive.files}
        hints=calls[index]['arguments'].get('hints',{})
        old,old_seconds=observe(before,images,index,hints)
        new,new_seconds=observe(after,images,index,hints)
        red=new['detections']['red'];visible='red'in new['poses']
        checks[f'{label}_{index}_current_timestamp_hash']=new['timestamp']==rows[index]['timestamp']and new['frame_sha256']==rows[index]['frame_sha256']
        checks[f'{label}_{index}_board_unchanged']=old['detections']['board']==new['detections']['board']
        checks[f'{label}_{index}_other_detections_unchanged']={k:v for k,v in old['detections'].items()if k!='red'}=={k:v for k,v in new['detections'].items()if k!='red'}
        evidence=shape_vision.raw_image_evidence({key:shape_vision.foreground(rgb)for key,rgb in images.items()},cameras)
        supported,proof=shape_vision.bead_image_support('red',new['poses']['red'],evidence,cameras)if visible else(False,{})
        checks[f'{label}_{index}_Red_current_bead_support']=visible and supported
        if index==26:
            checks[f'{label}_{index}_fresh_recovery_executed']=red.get('shape_hypothesis_consistency_checked')is True and red.get('bounded_rgb_source',{}).get('target_pose')==new['poses'].get('red')
        else:
            checks[f'{label}_{index}_valid_strict_Red_preserved']=red==old['detections']['red'] and new['poses']['red']==old['poses']['red']
        checks[f'{label}_{index}_proposal_not_copied']=visible and shape_vision.shape_distance('red',new['poses']['red'],hints['red'])>1.e-6
        record=dict(caller=label,index=index,actual_observe_call=calls[index],baseline=old,candidate=new,current_bead_support=proof,baseline_seconds=old_seconds,candidate_seconds=new_seconds)
        records.append(record);positives.append(record)
        OUT.with_name('red_rgb_actual_callers_partial.json').write_text(json.dumps(dict(checks=checks,records=records,negatives=negatives),indent=2)+'\n')
        print(label,index,'Red visible/support/recovery',visible,supported,red.get('shape_hypothesis_consistency_checked'),round(new_seconds,3),flush=True)
    with np.load(SRC/'0026.npz')as archive:current_images={key:archive[key]for key in archive.files}
    good=positives[-1]['candidate']['poses'].get('red')
    bad=rows[26]['poses']['red']
    erased={key:rgb.copy()for key,rgb in current_images.items()}
    fork=Rotation.from_quat(good[3:]).apply(shape_vision.local_beads('red')[0])+good[:3]
    for key,camera in cameras.items():
        pixel=camera.project(fork[None])[0]
        depth=((fork-camera.camera_to_world[:3,3])@camera.camera_to_world[:3,:3])[2]
        radius=int(np.ceil(camera.K[0,0]*shape_vision.RADIUS/depth))+6
        cv2.circle(erased[key],tuple(np.rint(pixel).astype(int)),radius,(64,64,64),-1)
    fixtures=[('blank_with_valid_history',{key:np.full_like(rgb,127)for key,rgb in current_images.items()},calls[26]['arguments']['hints'],None),
        ('without_red_proposal',current_images,{}, {'board':rows[25]['poses']['board']}),
        ('bad_red_proposal',current_images,{'red':bad},None),
        ('fork_pixels_erased',erased,calls[26]['arguments']['hints'],None)]
    for name,images,hints,previous in fixtures:
        new,seconds=observe(after,images,26,hints,previous)
        visible='red'in new['poses'];passed=not visible
        if visible and name in('without_red_proposal','bad_red_proposal'):
            evidence=shape_vision.raw_image_evidence({key:shape_vision.foreground(rgb)for key,rgb in images.items()},cameras)
            supported,proof=shape_vision.bead_image_support('red',new['poses']['red'],evidence,cameras)
            passed=supported and shape_vision.shape_distance('red',good,new['poses']['red'])<=.003
        checks[f'{label}_negative_{name}']=bool(passed)
        negatives.append(dict(caller=label,fixture=name,passed=bool(passed),result=new,seconds=seconds))
        print(label,'negative',name,passed,'visible',visible,round(seconds,3),flush=True)
report=dict(schema='puzzler.red_rgb_actual_callers_replay.v1',policy_input=False,checks=checks,all_passed=all(checks.values()),records=records,negatives=negatives,
    scope='Actual nested canonical/suffix estimator AST through actual VisualObserver, saved native RGB and same-capture robot provenance, original hints as optimizer proposals; no Env/render/GPU/GT. Red-only reviewed recovery with complete current bead/alias support. No physical success claim.')
OUT.write_text(json.dumps(report,indent=2)+'\n')
print('all_passed',report['all_passed'],'checks',len(checks),flush=True)
assert report['all_passed'],[key for key,value in checks.items()if not value]
