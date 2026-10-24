"""Reusable native pooled action-only replay → graph on the SAME live Env.

The source initial TaskSpec may use its ORIGINAL canonical scene initializer;
no physical state is restored at the later fork. Protocol/graph freezing is
owned by the caller. One source and one episode per native process.
"""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import shutil
import sys

from robosimstudio_bench.solutions import SolutionBase
from puzzler.pool_policy import PooledVisionGraph,validated_visual_history_seed
from puzzler.episode import snapshot_sources
from tools.explore_staged_pieces import ROOT,ENVIRONMENT,digest,write_json,run_process,changed_sources,native_capture_pointer
from tools.replay_policy_binding import bind_current_targets
from tools.run_catalog_stages import CatalogVisionGraph


def load(path):
    path=Path(path).resolve();p=json.loads(path.read_text())
    if digest(path)!=path.with_suffix('.sha256').read_text().strip():raise ValueError('Fork protocol mismatch')
    changed=changed_sources(ROOT,p['source_sha256'])
    changed.extend(changed_sources(Path('/'),p.get('external_source_sha256',{})))
    if changed:raise ValueError('Fork source changed: '+str(changed))
    return p


def replay_api(p):
    path=Path(p['replay_api']);name='frozen_replay_api_'+digest(path)[:12]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


class IndependentVerificationMixin:
    def reset(self,obs,info):
        if not getattr(self,'_independent_verifier_ready',False):
            import importlib.util
            from puzzler import modular_verification
            p=self._verification_protocol
            path=Path(p['independent_verifier_path']);key='frozen_independent_verifier_'+digest(path)[:16]
            spec=importlib.util.spec_from_file_location(key,path);provider=importlib.util.module_from_spec(spec);sys.modules[key]=provider;spec.loader.exec_module(provider)
            provider.install(self.env,expected_order=self.parameters['assembly_order'],initialized_prefix=p.get('verification_prefix',[]),output_dir=str(Path(p['output'])/'independent_verification'))
            if p.get('verification_resume_context'):provider.restore_rehearsal_baselines(self.env,p['verification_resume_context'])
            self._independent_provider=provider
            class Adapter:
                def capture(adapter,env,world):return provider.capture(env,self.bus,world)
                def verify(adapter,kind,world,outputs):
                    if kind in ('release_dispatch','finish'):
                        from verification import unified_checkpoints
                        return getattr(unified_checkpoints,kind)(provider,world,outputs)
                    return provider.validate(kind,world,outputs)
            modular_verification.install(Adapter());self._independent_verifier_ready=True
        return super().reset(obs,info)

    def close(self):
        try:return super().close()
        finally:
            if getattr(self,'_independent_verifier_ready',False):
                self._independent_provider.finalize(self.env)
                self._independent_verifier_ready=False

    def _new_connector(self):
        connector=super()._new_connector();base=connector.world_snapshot
        def snapshot():
            world=base()
            if not getattr(self,'_independent_verifier_ready',False):raise RuntimeError('Independent verifier must be installed before graph checkpoint')
            self._independent_provider.capture(self.env,self.bus,world)
            return world
        connector.world_snapshot=snapshot
        return connector


class ControlReplayGraph(IndependentVerificationMixin,CatalogVisionGraph):
    """Explicit RGB-seed context for a control replay, not an object preset."""
    def __init__(self,env,*,protocol,conditional_initial_records=(),**kwargs):
        self._conditional_initial_records=tuple(conditional_initial_records)
        p=load(protocol);self._verification_protocol=p;seed=validated_visual_history_seed(p['visual_history_seed']) if p.get('visual_history_seed') else None
        if int(env.step_count)!=p['fork_step'] or seed and seed['source_timestamp']>env.step_count/env.pipeline.control_hz:
            raise ValueError('Wrong control-replay/image-history boundary')
        original_prefix=p.get('source_prefix_context')
        if original_prefix:
            # This is the source TaskSpec's INITIAL canonical scene metadata,
            # never a fork-point object snapshot or a late state setter.
            super().__init__(env,prefix_context=original_prefix,public_prefix=p.get('initialized_prefix',()),
                visual_history_seed=p.get('visual_history_seed'),**kwargs)
        else:
            super().__init__(env,**kwargs)
            if seed:
                self.visual.previous={name:list(pose) for name,pose in seed['poses'].items()}
                self.visual_seed_provenance=dict(path=p['visual_history_seed'],sha256=digest(p['visual_history_seed']),
                    source_observations_path=seed['source_observations_path'],source_observations_sha256=seed['source_observations_sha256'],
                    source_rgb_archive=seed['source_rgb_archive'],source_rgb_archive_sha256=seed['source_rgb_archive_sha256'],
                    source_timestamp=seed['source_timestamp'],frame_sha256=seed['frame_sha256'],
                    scope='CONTROL-REPLAY RGB optimizer seeds only; fresh images validate every returned pose.')
        self.replay_context={key:p[key] for key in ('fork_step','action_tape_sha256','source_episode','source_context_kind')}
        files=[Path(__file__),ROOT/'tools/replay_policy_binding.py',Path(p['replay_api']),Path(p['action_tape']),
            Path(p['replay_metadata'])]
        if p.get('visual_history_seed'):files.append(Path(p['visual_history_seed']))
        self.source_hashes.update(snapshot_sources(ROOT,self.output,files))
        launch=json.loads((self.output/'launch.json').read_text());launch.update(source_sha256=self.source_hashes,
            control_replay_context=self.replay_context,visual_seed_provenance=self.visual_seed_provenance)
        self._write('launch.json',launch)

    def _register_visual(self,registry,env,bus):
        super()._register_visual(registry,env,bus)
        from puzzler import pyramid_held_belief
        pyramid_held_belief.register(registry,env,bus,source_provider=lambda:tuple(self.visual.records))
        from puzzler import conditional_attachment
        history=[]
        if getattr(self,'_conditional_initial_records',()):
            history=json.loads(Path(self._verification_protocol['conditional_owned_history']).read_text())
        conditional_attachment.register(registry,env,bus,self.visual,initial_records=getattr(self,'_conditional_initial_records',()),historical_owned_records=history)
        from puzzler import robot_proprioception
        robot_proprioception.register(registry,env,bus)

    def _new_connector(self):
        connector=super()._new_connector();base=connector.world_snapshot;registry=connector.tool_registry
        def execution_journal_cursor():
            return dict(call_id=registry.call_count,session_nonce=registry.session_nonce,timestamp=registry.clock()if registry.clock else None)
        registry.source.register_callable('robot.execution_journal_cursor',execution_journal_cursor,summary='Native robot tool journal cursor; no object state')
        registry._tools['robot.execution_journal_cursor']=registry.source._tools['robot.execution_journal_cursor']
        def snapshot():
            world=base()
            world.sim_state['vision_history']=list(self.visual.records)
            world.sim_state['robot_command_session']=registry.session_nonce
            world.sim_state['robot_command_journal']=[json.loads(l) for l in self.tool_record_path.read_text().splitlines()] if self.tool_record_path.exists() else []
            world.sim_state['pyramid_held_belief_history']=list(registry.pyramid_held_belief_history)
            return world
        connector.world_snapshot=snapshot
        return connector

    def row_extras(self):
        extra=super().row_extras();result=json.loads((self.output/'result.json').read_text())
        result.update(control_replay_context=self.replay_context,
            scope='Conditional phase graph after source-scene robot-control replay. No object/velocity restore at the fork, no full-task claim.',success=False)
        self._write('result.json',result);return extra


class AdaptiveModularGraph(ControlReplayGraph):
    def __init__(self,env,*,graph,output_dir,variant,independent_verifier_path,verification_prefix=(),known_unplaced_names=None,**kwargs):
        self._verification_protocol=dict(independent_verifier_path=independent_verifier_path,verification_prefix=list(verification_prefix),output=str(Path(output_dir).parent))
        self.replay_context=dict(initialization='FRESH_NATIVE_RESET',recorded_action_source=None)
        PooledVisionGraph.__init__(self,env,graph=graph,output_dir=output_dir,variant=variant,**kwargs)
        if known_unplaced_names:
            from functools import partial
            from puzzler.shape_vision import estimate_shapes
            from puzzler import vision as plain_vision

            known=tuple(known_unplaced_names)
            def guarded_estimator(images,cameras,previous=None):
                """Shape tracker plus bounded plain RGB/CAD recovery.

                Packed lower-layer pieces can fail the shape tracker's
                all-bead occlusion rule after a neighbouring release.  For a
                missing target, run the independent calibrated RGB/CAD fit
                as a second observer.  A pose is merged only when the fresh
                plain fit itself passes its four-view support and residual
                gates and the same packet contains a board fit; no CAD pose
                is copied into policy state.
                """
                strict=estimate_shapes(images,cameras,previous,
                                        known_unplaced_names=known)
                missing=[name for name in known
                         if name not in strict.get('poses',{})]
                if not missing or 'board' not in strict.get('poses',{}):
                    return strict
                plain=plain_vision.estimate(images,cameras,previous={})
                board_ok='board' in plain.get('poses',{})
                for name in missing:
                    detection=plain.get('detections',{}).get(name,{})
                    if name=='red':
                        from puzzler.shape_vision import foreground,raw_image_evidence,bead_image_support,refuse_shape_aliases,object_masks
                        # A held RGB/FK prediction is an optimizer proposal,
                        # never the fresh pose. Preserve full-palette pixel
                        # competition when refitting this missing shape.
                        masks={key:plain_vision.detect(rgb,only_names=(name,))[name]
                               for key,rgb in images.items()}
                        if previous and name in previous:
                            detection=plain_vision.fit_piece(name,masks,cameras,previous=previous[name])
                        foregrounds={key:foreground(rgb) for key,rgb in images.items()}
                        evidence=raw_image_evidence(foregrounds,cameras)
                        supported,diagnostic=bead_image_support(name,detection['pose'],evidence,cameras) if detection.get('visible') else (False,{})
                        if supported:
                            trial=deepcopy(strict['detections'])
                            trial[name]=dict(detection,independent_rgb_support=diagnostic)
                            cache={}
                            def fit_hypothesis(shape,seed):
                                key=(shape,tuple(seed))
                                if key not in cache:
                                    if shape==name:
                                        fitted=plain_vision.fit_piece(shape,masks,cameras,previous=seed)
                                    else:
                                        local_masks,contours=object_masks(shape,seed,foregrounds,cameras)
                                        fitted=plain_vision.fit_piece(shape,local_masks,cameras,previous=seed,contour_points=contours)
                                    if fitted.get('visible'):
                                        supported_beads,proof=bead_image_support(shape,fitted['pose'],evidence,cameras)
                                        fitted['independent_rgb_support']=proof
                                        if not supported_beads:fitted.update(visible=False,reason=proof['reason'])
                                    cache[key]=fitted
                                return cache[key]
                            # Reuse the shape observer's joint alias and
                            # foreign-evidence withholding gates unchanged.
                            refuse_shape_aliases(trial,evidence,cameras,fit_candidate=fit_hypothesis)
                            supported=trial[name].get('visible') is True
                            if supported:detection=trial[name]
                        if not supported:
                            strict['detections'][name]['appearance_recovery_diagnostic']=dict(
                                accepted=False,independent_rgb_support=diagnostic,
                                reason='current RGB recovery lacks unique independently attributed bead support')
                            continue
                    support=detection.get('center_support',[])
                    residual=float(detection.get('median_residual_m',float('inf')))
                    rms=float(detection.get('rms_residual_m',float('inf')))
                    accepted=bool(board_ok and detection.get('visible')
                                  and len(support)>=4 and all(float(v)>=.6 for v in support)
                                  and residual<.003 and rms<.004
                                  and (name=='red' or name in plain.get('poses',{})))
                    if not accepted:
                        continue
                    merged=dict(detection)
                    merged['identity_source']='plain RGB/CAD seeded fallback'
                    if name=='red':
                        merged['optimizer_proposal_source']="previous['red']; optimizer only, never a measurement" if previous and name in previous else 'current RGB initialization only'
                        merged['shape_hypothesis_consistency_checked']=True
                    merged['bounded_rgb_source']=dict(
                        source='fresh_rgb_calibrated_cad_fit',
                        current_board_measured=True,
                        support_views=int(sum(float(v)>=.6 for v in support)),
                        support_by_view=[float(v) for v in support],
                        median_ray_fit_residual_m=residual,
                        rms_ray_fit_residual_m=rms,
                        pose_fitted_from_current_rgb=True,
                        pose_not_copied_from_public_cad=True,
                        q16_fk_required=True)
                    strict.setdefault('poses',{})[name]=detection['pose'] if name=='red' else plain['poses'][name]
                    strict.setdefault('detections',{})[name]=merged
                    strict.setdefault('bounded_rgb_source',{})[name]=dict(merged['bounded_rgb_source'])
                if strict.get('bounded_rgb_source'):
                    strict['bounded_rgb_source']['board']=dict(
                        source='fresh_rgb_board_fit', current_board_measured=True,
                        frame_bound=True)
                return strict
            self.visual.estimator=guarded_estimator
            self.visual.previous={name:pose for name,pose in self.visual.previous.items()if name=='board'or name in known_unplaced_names}

    def row_extras(self):
        extra=PooledVisionGraph.row_extras(self)
        result=json.loads((self.output/'result.json').read_text())
        result.update(control_replay_context=self.replay_context,
            scope='Fresh native reset with explicitly declared given-prefix TaskSpec fixture; all actions generated by current RGB/publicCAD/robot proprioception modules. No recorded actions or later physical state restore.')
        self._write('result.json',result)
        return extra



class ReplayThenGraph(SolutionBase):
    def __init__(self,env,*,protocol,**kwargs):
        super().__init__(env,**kwargs);self.p=load(protocol);self.api=replay_api(self.p)
        self.actions,self.meta=self.api.load_controls();self.delegate=None;self._info={};self._binding={}
        self._new_conditional_attachment=None
        private=self.api.source_task_module(self.meta['source_episode'])
        variant=private.Variant(**{**self.meta['source_variant'],'preset_pieces':()})
        private.CURRENT.clear();private.CURRENT.update(env=env,variant=variant)

    @property
    def control_mode(self):return 'joint_position'

    def reset(self,obs,info):
        if self.delegate is not None:raise RuntimeError('One native fork episode only')
        self._info=dict(info)
        self._initialization_check=None
        if self.p.get('inline_initialization_equivalence'):
            from tools.initialization_equivalence import InitializationEquivalence
            self._initialization_check=InitializationEquivalence(self.env,self.meta,self.p['output'])

    def auxiliary_capture(self,name):
        # Native evaluator recording only; the policy never reads these arrays.
        from robosimstudio.record import EpisodeRecorder
        from robosimstudio.specs import RecordSpec,CaptureSpec
        recorder=EpisodeRecorder(Path(self.p['output'])/name,RecordSpec(enabled=False),
            capture=CaptureSpec(enabled=True,every=1,contacts=True,tool_spans=False))
        recorder.begin(self.env.episode,self.env)
        recorder.end(self.env,self._info,extra=dict(policy_input=False,scope='Zero-step attachment audit only'))
        if recorder.failures:raise ValueError('Native attachment recorder failed')

    def get_action(self,obs):
        step=int(self.env.step_count)
        if self.p.get('conditional_reattachment_materials')and self._new_conditional_attachment is None:
            material=json.loads(Path(self.p['conditional_reattachment_materials']).read_text())
            if step==material['reissue_at_actual_replay_step']:
                from tools.initialization_attachment import capture_at_replayed_close
                self._new_conditional_attachment=capture_at_replayed_close(self.env,material,
                    json.loads(Path(self.p['conditional_owned_history']).read_text()),self.actions,self.p['fork_step'])
                write_json(Path(self.p['output'])/'replayed_new_close_attachment.json',self._new_conditional_attachment)
        if self.delegate is None and step%500==0:
            write_json(Path(self.p['output'])/'progress.json',dict(phase='TEST_INITIALIZATION_ONLY',native_step=step,fork_step=self.p['fork_step'],online_policy_started=False))
        if step<self.p['fork_step']:return self.actions[step]
        if self.delegate is None:
            if step!=self.p['fork_step']:raise ValueError('Replay boundary was skipped')
            if self._initialization_check is not None:
                self._initialization_check.finish(self._info);self._initialization_check=None
            self.auxiliary_capture('attachment_before')
            if self.p.get('conditional_reattachment_materials')and self._new_conditional_attachment is None:raise ValueError('Actual replay close capture was skipped')
            self.delegate=ControlReplayGraph(self.env,protocol=self.p['protocol_path'],conditional_initial_records=([self._new_conditional_attachment]if self._new_conditional_attachment else()),graph=self.p['graph'],
                output_dir=self.p['episode'],variant=self.p['variant'],observation='rgb-shape',max_steps=self.p.get('max_steps',12000))
            args=dict(expected_facade_sha256=self.p['native_facade_sha256'],expected_slots=self.p['arm_slots'])
            self._binding['before_reset']=bind_current_targets(self.delegate.connector.facade,self.env,**args)
            self.delegate.reset(obs,self._info)
            self._binding['after_reset']=bind_current_targets(self.delegate.connector.facade,self.env,**args)
            write_json(Path(self.p['output'])/'command_cache_binding.json',self._binding)
            self.auxiliary_capture('attachment_after')
            write_json(Path(self.p['output'])/'progress.json',dict(phase='MODULAR_RGB_CAD_POLICY',native_step=step,fork_step=self.p['fork_step'],online_policy_started=True))
        # Recorded controls are unavailable after the test-initialization boundary.
        self.actions=None;self.api=None;self.meta=None
        self.delegate.get_action(obs)  # All following actions are generated by RGB/CAD/proprio graph tools.
        space=self.env.controllers.get('joint_position')
        if not callable(getattr(space,'target_action',None)):raise ValueError('Public commanded-action inverse is required')
        return space.target_action(self.env.control)

    def on_step(self,obs,reward,info):
        self._info=dict(info)
        if self.delegate is not None:self.delegate.on_step(obs,reward,info)

    def row_extras(self):
        if self.delegate is None:return dict(replay_fork_reached=False)
        return {**self.delegate.row_extras(),'control_replay_fork':self.p['fork_step']}

    def raw_spans(self):return self.delegate.raw_spans() if self.delegate else []
    def recording_stage(self):return self.delegate.recording_stage() if self.delegate else 'robot_control_replay_prefix'
    def close(self):
        if self.delegate is not None:self.delegate.close()


def jobs(path):
    from robosimstudio.specs import JobSpec,CaptureSpec
    p=load(path);api=replay_api(p);_,meta=api.load_controls();private=api.source_task_module(meta['source_episode'])
    variant=private.Variant(**{**meta['source_variant'],'preset_pieces':()})
    source_kwargs=dict(meta.get('source_task_kwargs',{}))
    if meta.get('source_start_poses'):source_kwargs['start_poses']=meta['source_start_poses']
    if set(source_kwargs)-{'start_poses','completed'}:raise ValueError('Unreviewed source task initializer arguments')
    original_prefix=p.get('source_prefix_context')
    if original_prefix:
        declared=source_kwargs.get('start_poses')
        if (not declared or Path(declared).resolve()!=Path(original_prefix['start_poses_path']).resolve()
                or digest(declared)!=original_prefix['start_poses_sha256']):
            raise ValueError('Canonical source initializer/context differ; a fork snapshot is not an initializer')
    if callable(getattr(api,'make_source_spec',None)):
        spec,private,variant=api.make_source_spec()
    else:
        spec=private.task_spec(variant,meta['source_max_steps'],suite=meta['source_task_name'].rsplit('/',1)[0],**source_kwargs)
    if spec.full_name!=meta['source_task_name']:raise ValueError('Original source TaskSpec identity changed')
    return [JobSpec(name=p['job_id'],task=spec.full_name,solution='tools.modular_native:ReplayThenGraph',
        solution_kwargs=(('protocol',str(Path(path).resolve())),),episodes=(0,1),seed=0,max_steps=p.get('max_steps',12000),
        control_mode='joint_position',device='cpu',renderer='',capture=CaptureSpec(enabled=True,every=1,contacts=True,tool_spans=True))]



def run_native(path):
    """Native RSS owns warming/reset/recording/stepping and the worker process."""
    from robosimstudio_bench.run.jobs import load_jobs
    from robosimstudio_bench.run.pool import run_pooled
    p=load(path)
    results=run_pooled(load_jobs(p['jobfile']),root=p['native_root'],workers=1,
        threads_per_worker=2,stagger=1.,preload=p['jobfile'])
    write_json(Path(p['output'])/'native_results.json',
        {name:dict(rows=value.rows,unrun=list(value.unrun)) for name,value in results.items()})
    if (Path(p['episode'])/'result.json').exists():
        from tools.explore_staged_pieces import native_capture_pointer
        native_capture_pointer(Path(p['output']),dict(job_id=p['job_id'],seed=0,
            task_suite=json.loads(Path(p['replay_metadata']).read_text())['source_task_name'].rsplit('/',1)[0],episode_path='episode'),dict(native_pool_root=p['native_root']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True)
    args=parser.parse_args();run_native(args.protocol)
