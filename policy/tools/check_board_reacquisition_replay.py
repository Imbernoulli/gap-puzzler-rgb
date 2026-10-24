"""Actual saved-pixel VisualObserver -> complete estimate_shapes replay."""
import ast
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
sys.dont_write_bytecode=True

ROOT=Path('/srv/home/bohanlyu/gap-puzzler-web')
OUT=Path(__file__).resolve().parents[1]/'verification'
BASE=Path(__file__).resolve().parents[1]
ARCHIVE=BASE/'verification/integration_sources/sync_009_board_rgb_reacquisition'
SRC=ROOT/'runs/red_bimanual_handoff_capacity_gpu_v3/workspace/runs/red_bimanual_capacity_given_prefix11_gpu_v3/episode/rgb'
rows=json.loads((SRC/'observations.json').read_text())


def package(name,paths):
    module=types.ModuleType(name)
    module.__path__=[str(path) for path in paths]
    sys.modules[name]=module
    return {key:importlib.import_module(name+'.'+key) for key in ('vision','shape_vision','vision_observer')}


baseline=package('_board_integration_base',[ARCHIVE/'puzzler',BASE/'puzzler'])
candidate=package('_board_integration_candidate',[BASE/'puzzler'])
cameras=baseline['vision'].calibration(baseline['vision'].rig_specs())


def observe(modules,images,timestamp,previous):
    # This is the actual observer and complete shape estimator. Only Red is
    # in the declared unplaced identity scope; no fit/support step is mocked.
    def estimator(images,cameras,seeds):
        return modules['shape_vision'].estimate_shapes(images,cameras,seeds,known_unplaced_names=('red',))
    observer=modules['vision_observer'].VisualObserver(lambda:(images,timestamp),cameras,{},estimator=estimator)
    observer.previous=deepcopy(previous)
    start=time.monotonic();result=observer.observe()
    return result,time.monotonic()-start


checks={};records=[]
for index in (7,8,9,10):
    source=SRC/f'{index:04}.npz'
    with np.load(source) as archive:images={key:archive[key] for key in ('top','front','rear','side')}
    historical_index=6 if index==7 else 7
    previous={key:rows[historical_index]['poses'][key] for key in ('board','red')}
    old,old_seconds=observe(baseline,images,rows[index]['timestamp'],previous)
    new,new_seconds=observe(candidate,images,rows[index]['timestamp'],previous)
    checks[f'actual_{index}_board_current_visible']=new['detections']['board'].get('visible') is True
    checks[f'actual_{index}_source_unchanged']=old['observation_source']==new['observation_source']=='rgb_shape_geometry'
    checks[f'actual_{index}_timestamp_current']=new['timestamp']==rows[index]['timestamp']
    checks[f'actual_{index}_hashes_current']=new['frame_sha256']==rows[index]['frame_sha256']
    checks[f'actual_{index}_nonboard_detections_unchanged']={k:v for k,v in old['detections'].items() if k!='board'}=={k:v for k,v in new['detections'].items() if k!='board'}
    if index==7:
        checks['good_top_keeps_original_detection']=new['detections']['board']==old['detections']['board']
        checks['good_top_does_not_call_fallback']='top_outline_diagnostic' not in new['detections']['board']
    else:
        board=new['detections']['board']
        checks[f'actual_{index}_top_refused_then_fallback']=not old['detections']['board'].get('visible') and board['top_outline_diagnostic']==old['detections']['board']
        checks[f'actual_{index}_two_current_views']=set(board['views'])=={'front','rear'} and all(v['visible'] and v['median_outline_residual_m']<=.0015 for v in board['views'].values()) and board['independent_corner_difference_m']<.0015
        checks[f'actual_{index}_proposal_explicit_history']=board['optimizer_proposal_key']=="previous['board']; historical RGB optimizer proposal only"
    record=dict(index=index,source_archive=str(source),archive_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                historical_rgb_proposal=dict(index=historical_index,timestamp=rows[historical_index]['timestamp'],previous=previous),
                baseline=old,candidate=new,baseline_seconds=old_seconds,candidate_seconds=new_seconds)
    records.append(record)
    (OUT/'board_reacquisition_partial_replay.json').write_text(json.dumps(dict(records=records,checks=checks),indent=2)+'\n')
    print('actual',index,'old/new board',old['detections']['board']['visible'],new['detections']['board']['visible'],new_seconds,flush=True)

with np.load(SRC/'0010.npz') as archive:current={key:archive[key] for key in ('top','front','rear','side')}
previous={key:rows[7]['poses'][key] for key in ('board','red')}
negatives=[]
fixtures=[(f'constant_{value}',{key:np.full_like(rgb,value) for key,rgb in current.items()},previous) for value in (0,127,255)]
fixtures.extend([
    ('front_blank',dict(current,front=np.full_like(current['front'],255)),previous),
    ('rear_blank',dict(current,rear=np.full_like(current['rear'],255)),previous),
    ('rear_shift_2px',dict(current,rear=np.roll(current['rear'],2,axis=1)),previous),
    ('rear_shift_4px',dict(current,rear=np.roll(current['rear'],4,axis=1)),previous),
    ('no_historical_board',current,{'red':previous['red']}),
])
for name,images,seeds in fixtures:
    old,old_seconds=observe(baseline,images,rows[10]['timestamp'],seeds)
    new,new_seconds=observe(candidate,images,rows[10]['timestamp'],seeds)
    passed=not new['detections']['board'].get('visible') and 'board' not in new['poses']
    checks[f'negative_{name}_board_refused']=passed
    checks[f'negative_{name}_base_refused']=not old['detections']['board'].get('visible')
    checks[f'negative_{name}_freshness_from_input']=new['timestamp']==rows[10]['timestamp'] and new['frame_sha256']=={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in images.items()}
    checks[f'negative_{name}_nonboard_unchanged']={k:v for k,v in old['detections'].items() if k!='board'}=={k:v for k,v in new['detections'].items() if k!='board'}
    if name=='no_historical_board':checks['no_history_original_board_detection_unchanged']=old['detections']['board']==new['detections']['board']
    negatives.append(dict(fixture=name,synthetic=name!='no_historical_board',passed=passed,baseline=old,candidate=new,
                          baseline_seconds=old_seconds,candidate_seconds=new_seconds))
    print('negative',name,passed,new_seconds,flush=True)

def function_ast(path,name):
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.dump(node,include_attributes=False)
checks['top_fit_AST_unchanged']=function_ast(ARCHIVE/'puzzler/board_vision.py','fit_board_outline')==function_ast(BASE/'puzzler/board_vision.py','fit_board_outline')
checks['board_API_frozen_hash_unchanged']=hashlib.sha256((BASE/'puzzler/board_vision.py').read_bytes()).hexdigest()=='f5331a028dfb52c4032ea600e05d7bef93ddc86ef33f5bb1f393b769d8d728bb'
report=dict(schema='puzzler.board_reacquire_shape_integration.v1',policy_input=False,
            scope='Actual VisualObserver and complete estimate_shapes, Red target-only identity scope, saved real RGB. Current timestamps and hashes come from saved captures. Historical RGB pose proposes optimization only. No Env/render/GT or mocked fits.',
            base_workspace=str(ARCHIVE),packaged_workspace=str(BASE),checks=checks,all_passed=all(checks.values()),records=records,negatives=negatives,
            source_sha256={name:hashlib.sha256((BASE/'puzzler'/name).read_bytes()).hexdigest() for name in ('board_vision.py','shape_vision.py')})
(OUT/'board_reacquisition_replay.json').write_text(json.dumps(report,indent=2)+'\n')
print('all_passed',report['all_passed'],'checks',len(checks),flush=True)
