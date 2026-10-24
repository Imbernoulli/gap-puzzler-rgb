"""Prepare/run an explicit given-prefix6 or9 conditional suffix; never canonical."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation
from puzzler import task,model
from tools.unified_native import UnifiedPyramidGraph

ROOT=Path(__file__).resolve().parents[1]
PREFIX=('azure','orange','yellow','teal','purple','cyan')
ORDER=('mint','lime','maroon','navy','pink','red')
ORIGINAL=PREFIX+ORDER

class GivenPrefixGraph(UnifiedPyramidGraph):
    def __init__(self,env,**kwargs):
        super().__init__(env,**kwargs)
        from puzzler.shape_vision import estimate_shapes
        from puzzler.red_rgb_recovery import recover_missing_red
        from puzzler.azure_rgb_attribution import refuse_unattributed_original_azure
        from puzzler.vision import fit_board
        given=tuple(self._verification_prefix)
        remaining=tuple(n for n in model.CELLS if n not in given)
        def estimate(images,cameras,previous=None):
            seeds=dict(previous or {});board=fit_board(images['top'],cameras['top'])
            proposed=[]
            if board.get('visible'):
                bp=np.asarray(board['pose']);br=Rotation.from_quat(bp[3:]).as_matrix()
                seeds['board']=bp.tolist()
                for name in given:
                    center,rr=self.visual.targets[name]
                    seeds[name]=np.r_[bp[:3]+br@center,Rotation.from_matrix(br@rr).as_quat()].tolist();proposed.append(name)
            observed=estimate_shapes(images,cameras,seeds,known_unplaced_names=remaining)
            observed=recover_missing_red(observed,images,cameras,seeds)
            observed=refuse_unattributed_original_azure(observed,images,cameras)
            for name in proposed:
                if name in observed.get('poses',{}):observed['detections'][name]['identity_source']='Public GIVEN goal optimizer hypothesis independently refined and validated by current RGB'
            observed['public_given_prefix_optimizer_hypotheses']=dict(names=proposed,seed_only=True,source='Current calibrated top RGB board × public given-prefix CAD; no raw-pose fill or previous-cache insertion')
            if 'cyan' in observed.get('poses',{}):
                pose=np.asarray(observed['poses']['cyan']);rr=Rotation.from_quat(pose[3:]).as_matrix()
                if rr[2,2]<-.8:
                    points=model.local_beads('cyan')
                    for turn in range(4):
                        symmetry=model.rotation(turn*np.pi/2)@np.diag([1.,-1.,-1.]);error=float(np.linalg.norm((points@symmetry.T)[:,None]-points[None],axis=2).min(axis=1).max())
                        if error<1.e-7 and (rr@symmetry)[2,2]>.8:
                            canonical=np.r_[pose[:3],Rotation.from_matrix(rr@symmetry).as_quat()].tolist();observed['poses']['cyan']=canonical;observed['detections']['cyan']['pose']=canonical
                            observed['detections']['cyan']['source_plane_CAD_symmetry_canonicalization']=dict(turn=turn,flip=True,world_bead_set_error_m=error,no_new_measurement=True);break
            return observed
        self.visual.estimator=estimate
        from puzzler.episode import snapshot_sources
        protocol_dir=self.output.parent
        self.source_hashes.update(snapshot_sources(ROOT,self.output,[Path(__file__),protocol_dir/'protocol.json',protocol_dir/'native_jobs.py']))
        launch=json.loads((self.output/'launch.json').read_text())
        launch.update(source_sha256=self.source_hashes,qualification_only=True,full12_acceptance=False,
            conditional_suffix_scope=f'Public given-prefix{len(self._verification_prefix)}, empty earned/history; only new suffix physical checkpoints count')
        self._write('launch.json',launch)

def _validated(path):
    protocol=json.loads(Path(path).read_text())
    count=len(protocol['verification_prefix'])
    prefix,order=ORIGINAL[:count],ORIGINAL[count:]
    if (count not in (6,9) or protocol['initialization']!=f'conditional_public_given_prefix{count}'
            or tuple(protocol['verification_prefix'])!=prefix
            or tuple(protocol['ordered_stages'])!=order
            or protocol.get('full12_acceptance') or not protocol.get('qualification_only')
            or protocol['device']!='cuda:0' or protocol['contact_capacity']!=4096):
        raise ValueError('Explicit GPU conditional prefix6 or9 suffix protocol required')
    from puzzler.pool_policy import validated_prefix_context
    validated_prefix_context(protocol['prefix_context'])
    if tuple(protocol['prefix_context']['initialized_prefix'])!=prefix:
        raise ValueError('Given fixture context and verification prefix disagree')
    graph=Path(protocol['graph']);document=json.loads((graph/'workflow.json').read_text())
    if (tuple(document['meta']['ordered_stages'])!=order
            or tuple(document['meta']['initialized_prefix'])!=prefix
            or document['meta']['full_assembly_eligible']):
        raise ValueError('Suffix graph cannot claim canonical full12')
    literals=[node['inputs']['state']for node in document['nodes'].values()
              if isinstance(node.get('inputs',{}).get('state'),dict)
              and '$ref'not in node['inputs']['state']]
    if len(literals)!=1:raise ValueError('Exactly one suffix initial state required')
    state=literals[0]
    if (tuple(state['initialized_prefix'])!=prefix or state['executed_placements']
            or state['execution_tokens'] or state['rgb_history'] or state['memory']['placements']):
        raise ValueError('Given prefix must not import earned state or observation history')
    return protocol


def prepare(output,*,prefix_count=6,start_poses=None):
    from tools.build_unified_graph import build
    if prefix_count not in (6,9):raise ValueError('Only reviewed prefix6 and prefix9 fixtures are supported')
    prefix,order=ORIGINAL[:prefix_count],ORIGINAL[prefix_count:]
    if start_poses is None:
        if prefix_count==9:raise ValueError('Explicit public prefix9 fixture path required')
        start_poses=ROOT.parents[2]/'runs/pyramid_suffix7_12_gpu_v3/workspace/contexts/prefix6/public_task_initialization.json'
    public=task.validate_start_poses(start_poses)
    if tuple(json.loads(public.read_text()).get('initialized_prefix',()))!=prefix:
        raise ValueError('Public fixture must declare the exact requested given prefix')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    graph=build(ROOT/'graphs'/output.name,pieces=order,given_prefix=prefix)
    start=output/f'public_given_prefix{prefix_count}.json';shutil.copy2(public,start)
    context=dict(kind='prefix-conditioned-initialization',initialized_prefix=list(prefix),
        start_poses_path=str(start),start_poses_sha256=hashlib.sha256(start.read_bytes()).hexdigest(),
        scope=f'Public catalog prefix{prefix_count} fixture only; fresh robot and empty earned ledger, no saved CP or RGB history imported')
    protocol=dict(schema='puzzler.conditional_bimanual_suffix_native.v1',
        initialization=f'conditional_public_given_prefix{prefix_count}',scope=f'Conditional sameEnv{prefix_count+1}–12 only; public given-prefix{prefix_count} is not earned; no canonical/full12 acceptance',
        graph=str(graph),episode=str(output/'episode'),native_root=str(output/'native_pool'),
        jobfile=str(output/'native_jobs.py'),suite=output.name,seed=0,max_steps=40000,
        device='cuda:0',contact_capacity=4096,prefix_context=context,verification_prefix=list(prefix),
        ordered_stages=list(order),initial_earned_empty=True,initial_rgb_history_empty=True,
        imported_checkpoints=False,visual_history_seed=None,qualification_only=True,
        full12_acceptance=False,flat_acceptance=False,native_run_started=False,
        source_plan='Frozen Cyan v6 delivery/ANY2 board; original non-Red phases/Maroon conditional-MAP actual history; fixed Red V19 plus public V7 four-field source overlay; each new physical CP required',
        complete_native_logs_required_for_capacity_audit=True)
    path=output/'protocol.json';path.write_text(json.dumps(protocol,indent=2)+'\n')
    (output/'native_jobs.py').write_text('from tools.conditional_suffix_native import jobs\nJOBS = jobs('+repr(str(path))+')\n')
    _validated(path)
    return path


def jobs(path):
    from robosimstudio.specs import JobSpec,CaptureSpec
    protocol=_validated(path)
    variant=task.Variant(mode='pyramid',pyramid_plan='pyramid_robot6_plan.json',preset_pieces=(),seed=protocol['seed'],appearance='original')
    spec=task.task_spec(variant,protocol['max_steps'],suite=protocol['suite'],
        start_poses=protocol['prefix_context']['start_poses_path'],terminate_on_success=False,contact_capacity=4096)
    kwargs=dict(graph=protocol['graph'],output_dir=protocol['episode'],variant=asdict(variant),
        max_steps=protocol['max_steps'],observation='rgb-shape',prefix_context=protocol['prefix_context'],
        verification_prefix=protocol['verification_prefix'],independent_verifier_path=str(ROOT/'verification/bimanual_independent.py'))
    return [JobSpec(name=f'conditional_suffix{len(protocol["verification_prefix"])+1}_12',task=spec.full_name,
        solution='tools.conditional_suffix_native:GivenPrefixGraph',solution_kwargs=tuple(kwargs.items()),
        episodes=(0,1),seed=variant.seed,max_steps=protocol['max_steps'],device='cuda:0',
        control_mode='joint_position',renderer='',capture=CaptureSpec(enabled=True,every=1,contacts=True,tool_spans=True))]


def run_prepared(path):
    path=Path(path).resolve();protocol=_validated(path)
    if protocol['native_run_started']or Path(protocol['episode']).exists():
        raise RuntimeError('Conditional candidate already started; prepare a new suffix')
    from tools.launch_unified import validate
    validate(Path(protocol['graph']))
    protocol['native_run_started']=True;path.write_text(json.dumps(protocol,indent=2)+'\n')
    from tools.unified_native import run
    return run(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run-prepared'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--prefix-count',type=int,choices=(6,9),default=6)
    parser.add_argument('--start-poses',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':
        if args.output is None:parser.error('--output required')
        print(prepare(args.output,prefix_count=args.prefix_count,start_poses=args.start_poses))
    else:
        if args.protocol is None:parser.error('--protocol required')
        print(run_prepared(args.protocol))
