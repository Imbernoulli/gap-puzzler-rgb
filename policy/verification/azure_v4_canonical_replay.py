"""Execute the actual immutable canonical estimator closure on saved RGB."""
import ast,hashlib,importlib.util,json,sys,time
from copy import deepcopy
from pathlib import Path
import numpy as np
ROOT=Path('/srv/home/bohanlyu/gap-puzzler-web');OUT=Path(__file__).parent;BASE=OUT.parents[0];sys.path.insert(0,str(BASE))
from puzzler import model,shape_vision
from puzzler.vision import calibration,rig_specs
from puzzler.vision_observer import VisualObserver,goals
from puzzler.red_rgb_recovery import recover_missing_red
from puzzler.modular_controller import save_observation
from puzzler.azure_rgb_attribution import refuse_unattributed_original_azure as gate
def caller(path):
 tree=ast.parse(path.read_text());init=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='__init__');f=next(n for n in ast.walk(init)if isinstance(n,ast.FunctionDef)and n.name=='estimate');scope=dict(model=model,estimate_shapes=shape_vision.estimate_shapes,recover_missing_red=recover_missing_red,refuse_unattributed_original_azure=gate);exec(compile(ast.Module(body=[f],type_ignores=[]),str(path),'exec'),scope);return scope['estimate']
old=caller(ROOT/'runs/pyramid_full12_bimanual_v3/workspace/tools/unified_native.py');new=caller(BASE/'tools/unified_native.py');cams=calibration(rig_specs());targets=goals('pyramid','pyramid_robot6_plan.json')
checks={};cases={}
def check(k,v):
 checks[k]=bool(v)
 if not v:raise AssertionError(k)
inputs=[]
for version,index in((6,41),(7,44)):
 ep=ROOT/f'runs/pyramid_suffix10_12_bimanual_gpu_v{version}/workspace/runs/navy_pink_red_given_prefix9_gpu_v{version}/episode/rgb';a=json.loads((ep/'observations.json').read_text());previous={}
 for p in a[:index]:previous.update(p['poses'])
 inputs.append((f'V{version}_{index}',ep/f'{index:04d}.npz',a[index],previous,False))
ep=ROOT/'runs/pyramid_staged_robot_imitation_v1/workspace/runs/full12_staged_robot_imitation_v2/episode/rgb';p=json.loads((ep/'observations.json').read_text())[0];inputs.append(('separated_positive',ep/'0000.npz',p,p['poses'],True))
for label,path,packet,previous,accepted in inputs:
 images=dict(np.load(path));check(label+'_actual_RGB_hashes',{k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in images.items()}==packet['frame_sha256']);start=time.perf_counter();before=old(images,cams,deepcopy(previous));after=new(images,cams,deepcopy(previous));check(label+'_old_visible','azure'in before['poses']);check(label+'_new_expected_acceptance',('azure'in after['poses'])==accepted);check(label+'_non_Azure_poses_equal',{k:v for k,v in before['poses'].items()if k!='azure'}=={k:v for k,v in after['poses'].items()if k!='azure'});check(label+'_non_Azure_detections_equal',{k:v for k,v in before['detections'].items()if k!='azure'}=={k:v for k,v in after['detections'].items()if k!='azure'})
 if not accepted:check(label+'_raw_candidate_exact',after['detections']['azure']['raw_shape_candidate']==before['detections']['azure'])
 else:check(label+'_positive_pose_exact',before['poses']['azure']==after['poses']['azure'])
 # Exercise the real VisualObserver binding/cache around the real closure output.
 class Sensor:
  def __init__(self):self.robot_geometry_provenance=deepcopy(packet.get('robot_geometry_provenance',{}))
  def capture(self):return images,packet['timestamp']
 sensor=Sensor();observer=VisualObserver(sensor.capture,cams,targets,estimator=lambda images,cameras,seeds:deepcopy(after));observer.previous=deepcopy(previous);bound=observer.observe();check(label+'_current_epoch_preserved',bound['timestamp']==packet['timestamp']and bound['frame_sha256']==packet['frame_sha256']);check(label+'_q16_preserved',bound.get('robot_geometry_provenance')==packet.get('robot_geometry_provenance',{}))
 if not accepted:check(label+'_old_Azure_cache_not_refreshed',observer.previous.get('azure')==previous.get('azure'))
 for category in('initialized','executed'):
  state=dict(rgb_history=[],belief_ledger={'azure':dict(status='current_rgb_conflict_goal_envelope_invalidated',source_timestamp=1.)},initialized_prefix=['azure']if category=='initialized'else[],executed_placements=['azure']if category=='executed'else[],parameters={'mode':'pyramid','pyramid_plan':'pyramid_robot6_plan.json'})
  if not accepted:
   updated=save_observation(state,bound);check(label+'_'+category+'_preexisting_invalidation_preserved',updated['belief_ledger']==state['belief_ledger']);check(label+'_'+category+'_raw_history_retained',updated['rgb_history'][-1]['detections']['azure']['raw_shape_candidate']==before['detections']['azure'])
 cases[label]=dict(source=str(path),timestamp=packet['timestamp'],actual_current_packet=bound,previous_proposals=previous,before=before,wall_s=time.perf_counter()-start)
for label,path,packet,previous,accepted in inputs[:2]:
 images=dict(np.load(path));blank={k:np.zeros_like(v)for k,v in images.items()};out=gate(deepcopy(packet),blank,cams);check(label+'_blank_refuses','azure'not in out['poses'])
report=dict(all_checks_passed=all(checks.values()),checks=checks,cases=cases,scope='Exact canonical UnifiedPyramidGraph estimator closure AST with immutable v3 real shape/Red recovery; actual saved RGB only. Historical accepted RGB poses are optimizer proposals. No Env, renderer, GT or native success claim. The VisualObserver binding subcheck wraps the already computed genuine closure output; no new fit is claimed there.')
(OUT/'azure_v4_canonical_replay.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(all_checks_passed=report['all_checks_passed'],checks=len(checks),path=str(OUT/'azure_v4_canonical_replay.json')),indent=2),flush=True)
