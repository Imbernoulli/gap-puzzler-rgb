"""Target one diagnosed piece using native RSS recorded-pose initialization.

This is a fresh prefix-conditioned episode, not an exact continuation: object
poses come from one native capture point; robot state, velocities and visual
history are reset. RSS owns construction, reset, multiprocessing and rollout.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path

from tools.explore_staged_pieces import (ROOT, ENVIRONMENT, changed_sources, digest,
                                         native_capture_pointer, run_process, write_json)


def extract_prefix(baseline, sequence, prefix, target):
    """Use one actual native point, without projecting any part onto its goal."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from robosimstudio.record import EpisodeCapture
    from puzzler import model
    from tools.audit_staged_episode import native_frame, matched_error

    baseline=Path(baseline).resolve()
    result=json.loads((baseline/'result.json').read_text())
    pointer=json.loads((baseline/'development/native_capture_pointer.json').read_text())
    for item in pointer['files'].values():
        if digest(item['path'])!=item['sha256']:
            raise ValueError('Baseline native capture hash mismatch')
    records=[json.loads(line) for line in (baseline/'development/native_checkpoint_indices.jsonl').read_text().splitlines()]
    rows=[row for row in records if row['sequence']==sequence and row['status']=='ok']
    if len(rows)!=1:
        raise ValueError('Choose one unique successful native checkpoint index')
    capture=EpisodeCapture(pointer['native_episode_dir'])
    frame=native_frame(capture.point(step=rows[0]['boundary_step']),pointer['gripper_readout'])
    initial=native_frame(capture.point(step=0),pointer['gripper_readout'])
    if not prefix or len(prefix)!=len(set(prefix)) or set(prefix)-set(model.CELLS) or target in prefix:
        raise ValueError('Prefix must contain unique known pieces other than the target')
    board=frame['poses']['board'];rb=Rotation.from_quat(board[3:]).as_matrix()
    errors={}
    for name in prefix:
        pose=frame['poses'][name]
        points=model.local_beads(name)@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]
        local=(points-np.asarray(board[:3]))@rb
        errors[name]=matched_error(local,np.asarray(result['target_beads_local'][name]))
        if errors[name]>=.004 or np.linalg.norm(frame['body_velocity'][name])>=.015:
            raise ValueError('The requested recorded prefix is not seated and stationary: '+name)
    target_points=model.local_beads(target)@Rotation.from_quat(frame['poses'][target][3:]).as_matrix().T+frame['poses'][target][:3]
    gaps=[float(np.linalg.norm(target_points-np.asarray(hand[:3]),axis=1).min()) for hand in frame['hands']]
    if min(gaps)<=.055 or abs(frame['poses'][target][2]-initial['poses'][target][2])>.01:
        raise ValueError('Chosen native point does not show a loose target separated from the hands')
    objects={name:dict(pose=pose[:3],quat=pose[3:],world_xform=True)
             for name,pose in frame['poses'].items()}
    return dict(schema='puzzler.recorded_prefix_poses.v1',objects=objects),dict(
        source_baseline=str(baseline),source_result_sha256=digest(baseline/'result.json'),
        source_native_capture=pointer['native_episode_dir'],
        source_native_capture_sha256=digest(Path(pointer['native_episode_dir'])/'capture.npz'),
        source_checkpoint_sequence=sequence,source_step=frame['step'],
        source_sensor_provenance=rows[0]['sensor_provenance'],
        prefix_goal_errors_m=errors,target_hand_gaps_m=gaps,
        source_pose_count=len(objects),scope='All recorded poses retained; no goal snapping. Robot joints, velocities and policy memory are not restored.')


def load_protocol(path):
    path=Path(path).resolve();protocol=json.loads(path.read_text())
    if digest(path)!=path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Prefix protocol hash mismatch')
    changed=changed_sources(ROOT,protocol['source_sha256'])
    if changed:
        raise ValueError('Frozen prefix sources changed: '+', '.join(changed))
    return protocol


def jobs_from_protocol(path):
    from robosimstudio.specs import JobSpec,CaptureSpec
    from puzzler import task

    protocol=load_protocol(path)
    variant=task.Variant(**{**protocol['variant'],'preset_pieces':()})
    spec=task.task_spec(variant,protocol['max_steps'],suite=protocol['task_suite'],
                        start_poses=protocol['prefix_context']['start_poses_path'])
    kwargs=dict(graph=protocol['graph'],output_dir=protocol['episode'],variant=asdict(variant),
                observation='rgb-shape',max_steps=protocol['max_steps'],
                prefix_context=protocol['prefix_context'],
                visual_history_seed=protocol.get('visual_history_seed'))
    return [JobSpec(name=protocol['target'],task=spec.full_name,
                    solution='puzzler.pool_policy:PooledVisionGraph',solution_kwargs=tuple(kwargs.items()),
                    episodes=(0,1),seed=protocol['seed'],max_steps=protocol['max_steps'],device='cpu',
                    capture=CaptureSpec(enabled=True,every=1,contacts=True,tool_spans=True))]


def prepare(output,graph,baseline,*,target='yellow',prefix=('orange','purple'),
            sequence=10,seed=501,max_steps=12000,timeout=1200.,reuse_rgb_history=False):
    from puzzler import task
    from puzzler.episode import snapshot_sources
    from tools.make_staged_vision_graph import build

    output,graph=Path(output).resolve(),Path(graph).resolve()
    if output.exists() or graph.exists():
        raise FileExistsError('Prefix probes require fresh output/graph directories')
    output.mkdir(parents=True)
    poses,provenance=extract_prefix(baseline,sequence,list(prefix),target)
    posefile=output/'prefix_poses.json';write_json(posefile,poses)
    task.validate_start_poses(posefile)
    graph=build(graph,[target],'flat',2)
    parameters=json.loads((graph/'params.json').read_text())
    variant=replace(task.sample(seed,'flat'),appearance='monochrome',scattered=False,
                    pyramid_plan=parameters['pyramid_plan'])
    prefix_context=dict(kind='prefix-conditioned-initialization',initialized_prefix=list(prefix),
                        start_poses_path=str(posefile),start_poses_sha256=digest(posefile),
                        provenance=provenance,
                        scope='Fresh native Env.reset from recorded object world poses; not exact continuation or full-task success.')
    seedfile=None;seed_sources=[]
    if reuse_rgb_history:
        baseline=Path(baseline).resolve();records_path=baseline/'rgb/observations.json'
        records=json.loads(records_path.read_text());source=provenance['source_sensor_provenance']
        matches=[(i,row) for i,row in enumerate(records) if row['timestamp']==source['timestamp'] and row['frame_sha256']==source['frame_sha256']]
        if len(matches)!=1:
            raise ValueError('Baseline checkpoint does not identify one accepted RGB source record')
        index,record=matches[0];archive=baseline/f'rgb/{index:04d}.npz'
        seedfile=output/'visual_history_seed.json'
        write_json(seedfile,dict(schema='puzzler.accepted_rgb_history_seed.v1',
            source_observations_path=str(records_path),source_observations_sha256=digest(records_path),
            source_observation_index=index,source_rgb_archive=str(archive),source_rgb_archive_sha256=digest(archive),
            source_timestamp=record['timestamp'],frame_sha256=record['frame_sha256'],poses=record['poses'],
            scope='Accepted image-derived optimizer priors only, never a fresh measurement.'))
        from puzzler.pool_policy import validated_visual_history_seed
        validated_visual_history_seed(seedfile)
        seed_sources=[seedfile,records_path,archive]
    jobfile=output/'native_jobs.py'
    jobfile.write_text('from tools.run_prefix_probe import jobs_from_protocol\n'
                       'JOBS=jobs_from_protocol('+repr(str(output/'protocol.json'))+')\n')
    paths=[*ROOT.joinpath('puzzler').glob('*.py'),ROOT/'RUNTIME.lock',ROOT/'requirements-cpu.lock',
           ROOT/'third_party/robosimstudio/assets/embodiments/yam/yam_bimanual_arms.xml',
           posefile,jobfile,ROOT/'tests/test_prefix_probe.py',*seed_sources]
    paths.extend(ROOT/'tools'/name for name in ('run_prefix_probe.py','rss.sh',
        'make_staged_vision_graph.py','validate_staged_graph.py','explore_staged_pieces.py',
        'native_piece_jobs.py','audit_staged_episode.py','audit_visual_measurements.py'))
    for directory,suffixes in ((graph,{'.py','.json','.md','.toml'}),
                                (ROOT/'assets',{'.obj','.json','.xml','.png'})):
        paths.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in suffixes)
    hashes=snapshot_sources(ROOT,output,paths)
    episode=output/'jobs'/target/'episode';episode.parent.mkdir(parents=True)
    protocol=dict(schema='puzzler.native_prefix_probe.v1',scope=prefix_context['scope'],
        target=target,initialized_prefix=list(prefix),prefix_context=prefix_context,
        visual_history_seed=str(seedfile) if seedfile else None,
        baseline_source=provenance,graph=str(graph),parameters=parameters,
        workflow_sha256=digest(graph/'workflow.json'),variant=asdict(variant),seed=seed,
        task_suite='prefix_'+hashlib.sha256(str(graph).encode()).hexdigest()[:12],
        episode=str(episode),output=str(output),native_root=str(output/'native_pool'),
        jobfile=str(jobfile),max_steps=max_steps,whole_batch_timeout_s=timeout,
        source_sha256=hashes,source_snapshot='source_snapshot',
        native_scheduler='robosimstudio_bench.run.pool.run_pooled',workers=1,
        native_capture=dict(enabled=True,every=1,contacts=True,tool_spans=True),
        acceptance='Target graph+native checkpoints+independent final subset audit+exact phase physics+separate perception gates. Prefix probes always full_success=false.',
        policy_boundary='Fresh RGB+known CAD/calibration/goals+robot encoders. Prefix pose JSON is initialization-only; no pose arrays or prefix provenance enters graph/tool arguments.')
    write_json(output/'protocol.json',protocol)
    (output/'protocol.sha256').write_text(digest(output/'protocol.json')+'\n')
    write_json(output/'state.json',dict(status='queued',initialized_prefix=list(prefix),full_success=False))
    command=[str(ROOT/'tools/rss.sh'),str(ROOT/'tools/validate_staged_graph.py'),str(graph)]
    validation=run_process(command,output/'validation.log',cwd=ROOT,environment=ENVIRONMENT,timeout=120.)
    write_json(output/'validation.json',validation)
    load_protocol(output/'protocol.json')
    if validation['exit_code']!=0:
        raise RuntimeError('Cold native graph validation failed')
    # Compile the native recorded-pose task as well, without constructing Env.
    jobs_from_protocol(output/'protocol.json')
    return protocol


def native(path):
    from robosimstudio_bench.run.jobs import load_jobs
    from robosimstudio_bench.run.pool import run_pooled
    protocol=load_protocol(path)
    results=run_pooled(load_jobs(protocol['jobfile']),root=protocol['native_root'],workers=1,
                        threads_per_worker=2,stagger=1.,preload=protocol['jobfile'])
    write_json(Path(protocol['output'])/'native_results.json',
               {name:dict(rows=value.rows,unrun=list(value.unrun)) for name,value in results.items()})


def run(path):
    protocol=load_protocol(path);output=Path(protocol['output']);episode=Path(protocol['episode'])
    if episode.exists() or json.loads((output/'state.json').read_text())['status']!='queued':
        raise FileExistsError('A prefix episode is never rerun in its original output directory')
    write_json(output/'state.json',dict(status='running',initialized_prefix=protocol['initialized_prefix'],full_success=False))
    process=run_process([str(ROOT/'tools/rss.sh'),'-u','-m','tools.run_prefix_probe','_native','--protocol',str(Path(path).resolve())],
        output/'native_pool.log',cwd=ROOT,environment=ENVIRONMENT,timeout=protocol['whole_batch_timeout_s'])
    state=dict(process,initialized_prefix=protocol['initialized_prefix'],full_success=False,
               prefix_conditioned=True,source_changes=changed_sources(ROOT,protocol['source_sha256']))
    if not (episode/'result.json').exists():
        state.update(status='process_error',error='Native job produced no project result')
    else:
        from tools.audit_staged_episode import audit_episode,stage_paths
        result=json.loads((episode/'result.json').read_text())
        job=dict(job_id=protocol['target'],seed=protocol['seed'],task_suite=protocol['task_suite'],
                 episode_path=str(episode.relative_to(output)))
        pointer=native_capture_pointer(output,job,dict(native_pool_root=protocol['native_root']))
        if pointer is None:
            raise RuntimeError('Native capture did not map to the completed prefix probe')
        audit=audit_episode(episode);write_json(output/'audit.json',audit)
        exact=audit['exact_checkpoint_audit']
        visited=[json.loads(p.read_text()).get('state',{}).get('piece') for p in stage_paths(episode)]
        target_only=bool(visited) and set(visited)=={protocol['target']}
        checkpoints=result['checkpoint_results']
        physical=bool(result.get('independent_subset_audit',{}).get('success'))
        passed=bool(result['graph_success'] and physical and target_only
            and checkpoints and all(row['passed'] for row in checkpoints)
            and not result['success'] and result.get('diagnostic_preset')
            and result.get('initialized_prefix')==protocol['initialized_prefix']
            and not state['source_changes'] and result['source_integrity_verified']
            and all(exact.get(k) is True for k in ('available','pairing_valid','complete',
                  'all_required_physical_predicates_pass','all_required_perception_gates_pass')))
        state.update(status='completed',probe_passed=passed,target_physical_success=physical,
            graph_success=result['graph_success'],reported_full_success=result['success'],
            initialized_prefix=result['initialized_prefix'],target_only_execution=target_only,
            actual_piece_visits=visited,checkpoint_results=checkpoints,
            independent_subset_audit=result.get('independent_subset_audit'),
            exact_checkpoint_audit=exact,native_capture=pointer,
            error=result.get('graph_trace',{}).get('graph_error') or result.get('error'),
            sim_time_s=result['sim_time_s'],audit_path=str(output/'audit.json'))
    write_json(output/'state.json',state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--out',required=True);p.add_argument('--graph',required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--target',default='yellow')
    p.add_argument('--prefix',default='orange,purple');p.add_argument('--sequence',type=int,default=10)
    p.add_argument('--max-steps',type=int,default=12000);p.add_argument('--timeout',type=float,default=1200.)
    p.add_argument('--reuse-rgb-history',action='store_true',help='Historical accepted RGB optimizer priors; first new image must revalidate them')
    for name in ('run','_native'):
        p=sub.add_parser(name);p.add_argument('--protocol',required=True)
    args=vars(parser.parse_args());command=args.pop('command')
    if command=='prepare':
        args['prefix']=tuple(args['prefix'].split(','));value=prepare(args.pop('out'),args.pop('graph'),args.pop('baseline'),**args)
        print(json.dumps(dict(prepared=True,target=value['target'],initialized_prefix=value['initialized_prefix'])))
    elif command=='_native':
        native(args['protocol'])
    else:
        value=run(args['protocol']);print(json.dumps({k:value.get(k) for k in ('status','probe_passed','target_physical_success','graph_success','full_success','error')}))


if __name__=='__main__':
    main()
