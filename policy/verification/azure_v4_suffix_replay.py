"""Actual prefix9 estimator closure on saved native RGB; no Env or motion."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
from puzzler import model, shape_vision, vision
from puzzler.vision_observer import goals
from puzzler.red_rgb_recovery import recover_missing_red
from puzzler.azure_rgb_attribution import refuse_unattributed_original_azure

WORK = Path(__file__).resolve().parents[1]
REPO = WORK.parents[2]
BASE = REPO/'runs/pyramid_full12_bimanual_v3/workspace'
given = ('azure', 'orange', 'yellow', 'teal', 'purple', 'cyan', 'mint', 'lime', 'maroon')
remaining = tuple(n for n in model.CELLS if n not in given)
checks = {}


def caller(path):
    tree = ast.parse(path.read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'estimate')
    scope = dict(model=model, np=np, Rotation=Rotation, given=given, remaining=remaining,
                 self=SimpleNamespace(visual=SimpleNamespace(targets=goals('pyramid', 'pyramid_robot6_plan.json'))),
                 estimate_shapes=shape_vision.estimate_shapes, fit_board=vision.fit_board,
                 recover_missing_red=recover_missing_red,
                 refuse_unattributed_original_azure=refuse_unattributed_original_azure)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
    return scope['estimate']


old = caller(BASE/'tools/conditional_suffix_native.py')
new = caller(WORK/'tools/conditional_suffix_native.py')
cameras = vision.calibration(vision.rig_specs())
records = []
for version, index in ((6, 41), (7, 44)):
    rgb = REPO/f'runs/pyramid_suffix10_12_bimanual_gpu_v{version}/workspace/runs/navy_pink_red_given_prefix9_gpu_v{version}/episode/rgb'
    packets = json.loads((rgb/'observations.json').read_text())
    packet = packets[index]
    images = dict(np.load(rgb/f'{index:04d}.npz'))
    previous = {}
    for earlier in packets[:index]:
        previous.update(earlier['poses'])
    before = old(images, cameras, deepcopy(previous))
    after = new(images, cameras, deepcopy(previous))
    prefix = f'V{version}_{index}_'
    checks[prefix+'actual_RGB_hashes'] = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in images.items()} == packet['frame_sha256']
    checks[prefix+'old_alias_visible_new_attribution_refused'] = 'azure' in before['poses'] and 'azure' not in after['poses']
    checks[prefix+'non_Azure_poses_unchanged'] = {k:v for k,v in before['poses'].items() if k != 'azure'} == {k:v for k,v in after['poses'].items() if k != 'azure'}
    checks[prefix+'non_Azure_detections_unchanged'] = {k:v for k,v in before['detections'].items() if k != 'azure'} == {k:v for k,v in after['detections'].items() if k != 'azure'}
    checks[prefix+'exact_raw_candidate_retained'] = after['detections']['azure']['raw_shape_candidate'] == before['detections']['azure']
    checks[prefix+'public_given_hypotheses_preserved'] = before['public_given_prefix_optimizer_hypotheses'] == after['public_given_prefix_optimizer_hypotheses']
    records.append(dict(version=version, index=index, timestamp=packet['timestamp'], before=before, after=after))
report = dict(all_passed=all(checks.values()), checks=checks, records=records,
              scope=__doc__, native_launched=False, physical_success_claim=False)
(WORK/'verification/azure_v4_suffix_replay.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(all_passed=report['all_passed'], checks=len(checks))))
assert report['all_passed'], [name for name, passed in checks.items() if not passed]
