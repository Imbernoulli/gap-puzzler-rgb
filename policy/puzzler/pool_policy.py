"""Puzzle telemetry adapter for RSS's native pooled runner.

RSS owns environment construction, multiprocessing, reset, stepping and rollout.
The inherited visual graph owns GaP execution. This adapter binds the task's
evaluation context and exports rehearsal evidence through native row_extras.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import model, task
from .audit import audit
from .episode import (NativeCheckpointIndexRecorder, capture_development_physics,
                      changed_sources, snapshot_sources,
                      wrap_native_checkpoint_index, write_development_spans)
from .vision_policy import VisionGraph


def validated_prefix_context(context):
    """Explicit evaluator provenance; never forwarded to visual policy tools."""
    if context is None:
        return {}
    if not isinstance(context,dict):
        raise ValueError('Prefix context must be an explicit provenance object')
    names=context.get('initialized_prefix',[])
    if (not names or len(names)!=len(set(names)) or set(names)-set(model.CELLS)
            or context.get('kind')!='prefix-conditioned-initialization'):
        raise ValueError('Prefix context must name unique known previously initialized pieces')
    path=task.validate_start_poses(context['start_poses_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=context.get('start_poses_sha256'):
        raise ValueError('Prefix start-pose provenance hash mismatch')
    return dict(context)


def full_assembly_eligible(subset, prefix_context):
    return not subset and not prefix_context


def validated_visual_history_seed(path):
    """Recover accepted RGB priors, never an observation or simulator pose."""
    path=Path(path).resolve();seed=json.loads(path.read_text())
    records_path=Path(seed['source_observations_path']).resolve()
    archive=Path(seed['source_rgb_archive']).resolve()
    if (hashlib.sha256(records_path.read_bytes()).hexdigest()!=seed['source_observations_sha256'] or
            hashlib.sha256(archive.read_bytes()).hexdigest()!=seed['source_rgb_archive_sha256']):
        raise ValueError('Historical RGB seed source hash mismatch')
    record=json.loads(records_path.read_text())[seed['source_observation_index']]
    if (record.get('observation_source')!='rgb_shape_geometry' or record.get('poses')!=seed['poses']
            or record.get('frame_sha256')!=seed['frame_sha256'] or
            record.get('timestamp')!=seed['source_timestamp']):
        raise ValueError('Seed does not match an accepted historical RGB record')
    with np.load(archive,allow_pickle=False) as images:
        if {key:hashlib.sha256(images[key].tobytes()).hexdigest() for key in images.files}!=record['frame_sha256']:
            raise ValueError('Historical RGB pixels disagree with recorded frame hashes')
    for name,pose in seed['poses'].items():
        value=np.asarray(pose,dtype=float)
        if (name not in {'board',*model.CELLS} or value.shape!=(7,) or not np.isfinite(value).all()
                or abs(float(np.linalg.norm(value[3:]))-1.)>1.e-4):
            raise ValueError('Invalid accepted historical RGB prior')
    return seed


class PooledVisionGraph(VisionGraph):
    """One native RSS policy instance, one immutable diagnostic output directory."""

    def __init__(self, env, *, graph, output_dir, variant,
                 observation='rgb-shape', max_steps=12000, prefix_context=None,
                 visual_history_seed=None, **kwargs):
        if observation != 'rgb-shape':
            raise ValueError('This rehearsal adapter requires shape-based RGB')
        self.output = Path(output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.variant = task.Variant(**{**variant, 'preset_pieces': tuple(variant.get('preset_pieces', ()))})
        self.source_graph_path = Path(graph).resolve()
        self.parameters = json.loads((self.source_graph_path / 'params.json').read_text())
        if self.parameters.get('observation_required') != observation:
            raise ValueError('Frozen graph observation contract mismatch')
        if self.variant.preset_pieces:
            raise ValueError('This sweep starts with all twelve pieces loose')
        self.prefix_context=validated_prefix_context(prefix_context)
        if visual_history_seed is not None and not self.prefix_context:
            raise ValueError('Historical RGB restart priors require explicit prefix context')
        seed=validated_visual_history_seed(visual_history_seed) if visual_history_seed else None
        self.visual_seed_provenance=(dict(path=str(Path(visual_history_seed).resolve()),
            sha256=hashlib.sha256(Path(visual_history_seed).read_bytes()).hexdigest(),
            source_observations_path=seed['source_observations_path'],
            source_observations_sha256=seed['source_observations_sha256'],
            source_rgb_archive=seed['source_rgb_archive'],source_rgb_archive_sha256=seed['source_rgb_archive_sha256'],
            source_timestamp=seed['source_timestamp'],frame_sha256=seed['frame_sha256'],
            scope='Accepted historical RGB optimizer priors only; fresh RGB must revalidate every returned pose.') if seed else {})
        self.samples, self.checks, self.events = [], [], []
        self._episode_started = False
        self._exported = False
        self._last_info = {}
        self._hold = 0
        self._achieved = False
        self._completion_step = None
        self.checkpoint_indices = NativeCheckpointIndexRecorder(self.output)
        task.CURRENT.clear()
        task.CURRENT.update(env=env, variant=self.variant)
        paths = [*model.ROOT.joinpath('puzzler').glob('*.py'),
                 model.ROOT / 'RUNTIME.lock',
                 model.ROOT / 'third_party/robosimstudio/assets/embodiments/yam/yam_bimanual_arms.xml']
        for directory, suffixes in ((self.source_graph_path, {'.py', '.json', '.md', '.toml'}),
                                    (model.ROOT / 'assets', {'.obj', '.xml', '.json', '.png'})):
            paths.extend(p for p in directory.rglob('*') if p.is_file()
                         and '__pycache__' not in p.parts and p.suffix in suffixes)
        paths.extend(model.ROOT / 'tools' / name for name in
                     ('audit_staged_episode.py', 'audit_visual_measurements.py',
                      'unified_native.py', 'launch_unified.py', 'build_unified_graph.py'))
        paths.append(model.ROOT / 'verification/independent.py')
        paths.append(model.ROOT / 'verification/unified_checkpoints.py')
        if self.prefix_context:
            paths.extend([Path(self.prefix_context['start_poses_path']),
                          model.ROOT/'tools/run_prefix_probe.py'])
        if seed:
            paths.extend([Path(visual_history_seed),Path(seed['source_observations_path']),Path(seed['source_rgb_archive'])])
        self.source_hashes = snapshot_sources(model.ROOT, self.output, paths)
        self._write('launch.json', dict(
            schema='puzzler.native_pool_launch.v1', policy_input=False,
            graph=str(self.source_graph_path.relative_to(model.ROOT)),
            seed=self.variant.seed, observation_mode=observation, variant=asdict(self.variant),
            source_sha256=self.source_hashes, source_snapshot='source_snapshot',
            max_steps=max_steps, runner='robosimstudio_bench.run.pool.run_pooled',
            prefix_context=self.prefix_context,
            visual_seed_provenance=self.visual_seed_provenance,
            completion='Launch metadata only; consult result.json when available.'))
        options = dict(roll='held', motion='osc', bus_timeout=600., settle_steps=150)
        options.update(kwargs)
        super().__init__(env, graph=str(self.source_graph_path),
                         visual_mode=self.variant.mode, visual_plan=self.variant.pyramid_plan,
                         visual_identity='shape', observation_dir=self.output / 'rgb',
                         trace_dir=str(self.output / 'trace'), **options)
        if seed:
            self.visual.previous={name:list(pose) for name,pose in seed['poses'].items()}
        self._observer = self._observe_physics
        env.add_step_observer(self._observer)

    def _write(self, filename, value):
        (self.output / filename).write_text(json.dumps(value, indent=2, default=str))

    def _new_connector(self):
        connector = super()._new_connector()
        connector.world_snapshot = wrap_native_checkpoint_index(
            connector.world_snapshot, self.checkpoint_indices,
            boundary_step=lambda: int(self.env.step_count),
            generation=lambda: getattr(self, '_generation', None))
        return connector

    def reset(self, obs, info):
        # Pool._build performs an Env-only warming reset. This method runs at
        # the real rollout reset, before the graph thread is started.
        if self._episode_started:
            raise RuntimeError('A frozen piece job permits one episode per output directory')
        self._episode_started = True
        self.samples.clear()
        self.checks.clear()
        self._start = time.perf_counter()
        self._sample()
        return super().reset(obs, info)

    def _sample(self):
        state = capture_development_physics(self.env)
        self.samples.append(dict(step=state['step'], poses=state['poses'], hands=state['hands'],
                                 body_q=self.env.state.body_q.numpy().tolist(),
                                 joint_q=self.env.state.joint_q.numpy().tolist()))

    def _observe_physics(self, *args):
        if not self._episode_started or self._exported:
            return
        grade = task.grade(self.env)
        self._hold = self._hold + 1 if grade['success'] else 0
        if self._hold >= 50 and not self._achieved:
            self._achieved = True
            self._completion_step = int(self.env.step_count)
        if self.env.step_count % 5 == 0:
            self._sample()
        if self.env.step_count % 50 == 0:
            self.checks.append(dict(step=int(self.env.step_count), **grade))

    def on_step(self, obs, reward, info):
        self._last_info = dict(info)
        return super().on_step(obs, reward, info)

    def row_extras(self):
        extra = super().row_extras()
        if self._exported:
            return {**extra, 'puzzler_result_path': str(self.output / 'result.json')}
        if not self.samples or self.samples[-1]['step'] != self.env.step_count:
            self._sample()
        raw_spans = self.raw_spans()
        graph_success = bool(self._result is not None and self._result.success)
        native_success = bool(self._last_info.get('success', False))
        final = task.grade(self.env)
        changes = changed_sources(model.ROOT, self.source_hashes)
        if self.variant.mode == 'pyramid':
            from .pyramid import get_plan, physical
            targets = {n: physical(p['sites']).tolist() for n, p in
                       get_plan(self.variant.pyramid_plan)['placements'].items()}
        else:
            targets = {n: np.c_[model.xy(cells), np.full(len(cells), model.BEAD_Z)].tolist()
                       for n, cells in model.CELLS.items()}
        subset = self.parameters.get('diagnostic_subset', [])
        result = dict(
            success=False, rss_success=native_success, graph_success=graph_success,
            seed=self.variant.seed, variant=asdict(self.variant),
            graph=str(self.source_graph_path.relative_to(model.ROOT)), observation_mode='rgb-shape',
            steps=int(self.env.step_count), control_hz=float(self.env.pipeline.control_hz),
            sim_time_s=float(self.env.step_count / self.env.pipeline.control_hz),
            wall_s=time.perf_counter() - self._start, completion_step=self._completion_step,
            final=final, events=self.events, graph_trace=self.trace,
            error=str(self._error or getattr(self._result, 'error', '') or ''),
            checkpoint_results=[c.to_dict() for c in getattr(self._result, 'checkpoint_results', [])],
            workflow_sha256=hashlib.sha256((self.source_graph_path / 'workflow.json').read_bytes()).hexdigest(),
            source_sha256=self.source_hashes, source_changed_during_run=changes,
            source_integrity_verified=not changes, target_beads_local=targets,
            diagnostic_preset=bool(self.prefix_context), diagnostic_subset=subset,
            prefix_conditioned=bool(self.prefix_context), prefix_context=self.prefix_context,
            initialized_prefix=self.prefix_context.get('initialized_prefix',[]),
            visual_seed_provenance=self.visual_seed_provenance,
            rgb_observation_count=len(self.visual.records),
            policy_tool_allowlist=sorted(self.connector.tool_registry.runtime_tools()),
            evaluator_is_policy_input=False,
            native_runner='robosimstudio_bench.run.pool.run_pooled',
            native_close_kind=self._last_info.get('close_kind'),
            native_termination_reason=self._last_info.get('termination_reason'))
        # Static measurement conventions omitted by the native capture
        # manifest; offline readers need these to reproduce Gripper.width.
        result['evaluation_gripper_readout'] = {}
        for arm in ('left', 'right'):
            gripper = self.env.gripper(arm)
            result['evaluation_gripper_readout'][arm] = dict(
                coords=list(map(int, gripper.coords)),
                width_coords=list(map(int, gripper.width_coords)),
                mirrored=bool(gripper.mirrored), open_q=np.asarray(gripper.open_q).tolist(),
                closed_q=np.asarray(gripper.closed_q).tolist(), span=float(gripper.span))
        result['independent_audit'] = audit(result, self.samples)
        result['diagnostic_success'] = bool(
            self._achieved and final['success'] and native_success and graph_success
            and result['independent_audit']['success'] and not changes)
        result['success'] = bool(result['diagnostic_success'] and full_assembly_eligible(subset,self.prefix_context))
        if subset:
            subset_result = {**result, 'target_beads_local': {n: targets[n] for n in subset}}
            result['independent_subset_audit'] = audit(subset_result, self.samples)
            result['single_piece_physical_diagnostic'] = dict(
                success=bool(graph_success and not changes and result['independent_subset_audit']['success']
                             and all(final['bead_error_m'][n] < .004 and final['speed_mps'][n] < .015
                                     for n in subset)),
                bead_error_m={n: final['bead_error_m'][n] for n in subset},
                scope=('Subset diagnostic from recorded prefix-conditioned initialization; not exact continuation or full task success.'
                       if self.prefix_context else 'Subset diagnostic, all twelve initially loose; not full task success.'))
        result['development_artifacts'] = write_development_spans(
            self.output, raw_spans, capture_step=self.env.step_count,
            control_hz=self.env.pipeline.control_hz)
        result['development_artifacts'].update(self.checkpoint_indices.finalize())
        result['native_checkpoint_recording'] = self.checkpoint_indices.metadata()
        result['native_capture'] = dict(
            enabled=True, every=1, contacts=True, tool_spans=True,
            path_resolution='Native RSS EpisodeRecorder shard; parent writes development/native_capture_pointer.json.',
            policy_input=False)
        self._write('frames.json', self.samples)
        self._write('checks.json', self.checks)
        self._write('result.json', result)
        self._exported = True
        return {**extra, 'puzzler_result_path': str(self.output / 'result.json'),
                'puzzler_graph_success': graph_success,
                'puzzler_subset_physical_success': result.get('single_piece_physical_diagnostic', {}).get('success'),
                'puzzler_full_success': result['success']}

    def close(self):
        try:
            if self._episode_started and not self._exported:
                self._write('frames.json', self.samples)
                self._write('checks.json', self.checks)
                self._write('interruption.json', dict(
                    status='No completed row_extras export; physical outcome unknown.',
                    step=int(self.env.step_count), policy_input=False))
                self.checkpoint_indices.finalize()
        finally:
            if hasattr(self, '_observer'):
                self.env.remove_step_observer(self._observer)
            super().close()
