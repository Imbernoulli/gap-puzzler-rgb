"""Native RSS-only runner for the unified graph; no replay or state restore API."""
from dataclasses import asdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

from puzzler import model, task
from puzzler.pool_policy import PooledVisionGraph

ROOT = Path(__file__).resolve().parents[1]
CYAN_TRANSFER_PHASES = frozenset({'handoff_approach', 'handoff_dualhold', 'handoff_receiveronly', 'handoff_receiver_hold'})


class UnifiedPyramidGraph(PooledVisionGraph):
    def __init__(self, env, *, independent_verifier_path, verification_prefix=(), **kwargs):
        self._verifier_ready = False
        self._verifier_path = Path(independent_verifier_path)
        self._verification_prefix = tuple(verification_prefix)
        super().__init__(env, **kwargs)
        # Public names restrict identity proposals; only fresh RGB/CAD fits
        # populate poses. Original-material proposal rescue stays geometry
        # gated and does not identify anything from simulator state.
        from puzzler.shape_vision import estimate_shapes
        from puzzler.red_rgb_recovery import recover_missing_red
        from puzzler.azure_rgb_attribution import refuse_unattributed_original_azure
        def estimate(images, cameras, previous=None):
            import numpy as np
            from scipy.spatial.transform import Rotation
            observed = estimate_shapes(images, cameras, previous, known_unplaced_names=tuple(model.CELLS))
            observed = recover_missing_red(observed, images, cameras, previous)
            observed = refuse_unattributed_original_azure(observed, images, cameras)
            if 'cyan' in observed.get('poses', {}):
                pose = np.asarray(observed['poses']['cyan'], dtype=float)
                rotation = Rotation.from_quat(pose[3:]).as_matrix()
                if rotation[2, 2] < -.8:
                    points = model.local_beads('cyan')
                    for turn in range(4):
                        symmetry = model.rotation(turn * np.pi / 2) @ np.diag([1., -1., -1.])
                        error = float(np.linalg.norm((points @ symmetry.T)[:, None] - points[None], axis=2).min(axis=1).max())
                        if error < 1.e-7 and (rotation @ symmetry)[2, 2] > .8:
                            canonical = np.r_[pose[:3], Rotation.from_matrix(rotation @ symmetry).as_quat()].tolist()
                            observed['poses']['cyan'] = canonical
                            observed['detections']['cyan']['pose'] = canonical
                            observed['detections']['cyan']['source_plane_CAD_symmetry_canonicalization'] = dict(
                                turn=turn, flip=True, world_bead_set_error_m=error, no_new_measurement=True)
                            break
            return observed
        self.visual.estimator = estimate
        from puzzler.episode import snapshot_sources
        self.source_hashes.update(snapshot_sources(ROOT, self.output, [
            *ROOT.joinpath('verification').glob('*.py'),
            *ROOT.joinpath('profiles').glob('*.json'),
            ROOT / 'registry_modular/skills/modular-rgb-puzzler/SKILL.md',
            ROOT / 'tools/layout_c_native.py',
            ROOT / 'runs/full12_original_order_layout_c_v10/native_jobs.py',
            ROOT / 'runs/full12_original_order_layout_c_v10/manifest.json']))
        launch = json.loads((self.output / 'launch.json').read_text())
        launch.update(source_sha256=self.source_hashes,
            bimanual_typed_candidate=True, full12_acceptance=False,
            canonical_original_staging=False,
            public_initial_layout=dict(name='C_Lime_world_y_plus070',piece='lime',
                translation_m=[0.,.070,0.],all12_loose=True),
            physical_checkpoint_extension='Cyan four phases and Red two distinct four-phase hops; original release/finish')
        self._write('launch.json', launch)

    def reset(self, obs, info):
        if self._verifier_ready:
            raise RuntimeError('One fresh native episode per unified graph process')
        from puzzler import modular_verification
        from verification import cyan_transfer
        key = '_unified_independent_native_verifier'
        spec = importlib.util.spec_from_file_location(key, self._verifier_path)
        provider = importlib.util.module_from_spec(spec)
        sys.modules[key] = provider
        spec.loader.exec_module(provider)
        provider.install(self.env, expected_order=self.parameters['assembly_order'],
                         initialized_prefix=list(self._verification_prefix), output_dir=str(self.output.parent / 'independent_verification'))
        cyan_transfer.install(provider)
        self._independent_provider = provider
        owner = self
        class Adapter:
            def capture(self, env, world):
                return provider.capture(env, owner.bus, world)
            def verify(self, kind, world, outputs):
                # Cyan and Red both name dual hold; piece identity selects the
                # proper independent physical predicate without cross-aliasing.
                if outputs['state']['piece'] == 'cyan' and kind in CYAN_TRANSFER_PHASES:
                    return cyan_transfer.verify(kind, world, outputs)
                if kind in ('release_dispatch', 'finish'):
                    from verification import unified_checkpoints
                    return getattr(unified_checkpoints, kind)(provider, world, outputs)
                return provider.validate(kind, world, outputs)
        modular_verification.install(Adapter())
        self._verifier_ready = True
        return super().reset(obs, info)

    def _register_visual(self, registry, env, bus):
        super()._register_visual(registry, env, bus)
        from puzzler import conditional_attachment, pyramid_held_belief, robot_proprioception
        pyramid_held_belief.register(registry, env, bus, source_provider=lambda:tuple(self.visual.records))
        conditional_attachment.register(registry, env, bus, self.visual)
        robot_proprioception.register(registry, env, bus)

    def _new_connector(self):
        connector = super()._new_connector()
        base = connector.world_snapshot
        registry = connector.tool_registry
        def execution_journal_cursor():
            return dict(call_id=registry.call_count, session_nonce=registry.session_nonce,
                        timestamp=registry.clock() if registry.clock else None)
        registry.source.register_callable('robot.execution_journal_cursor', execution_journal_cursor,
                                         summary='Native robot tool journal cursor; no object state')
        registry._tools['robot.execution_journal_cursor'] = registry.source._tools['robot.execution_journal_cursor']
        def snapshot():
            world = base()
            world.sim_state['vision_history'] = self.bus.call(lambda: deepcopy(self.visual.records))
            world.sim_state['robot_command_session'] = registry.session_nonce
            world.sim_state['robot_command_journal'] = [json.loads(line) for line in self.tool_record_path.read_text().splitlines()] if self.tool_record_path.exists() else []
            world.sim_state['pyramid_held_belief_history'] = list(registry.pyramid_held_belief_history)
            if not self._verifier_ready:
                raise RuntimeError('Independent verifier must be installed before checkpoint capture')
            self._independent_provider.capture(self.env, self.bus, world)
            return world
        connector.world_snapshot = snapshot
        return connector

    def close(self):
        try:
            return super().close()
        finally:
            if self._verifier_ready:
                self._independent_provider.finalize(self.env)
                self._verifier_ready = False


def jobs(manifest_path):
    from robosimstudio.specs import CaptureSpec, JobSpec
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['initialization'] != 'fresh_canonical_all12_loose':
        raise ValueError('Only fresh canonical native initialization is supported')
    if manifest.get('device') != 'cuda:0':
        raise ValueError('This delivery requires GPU simulation on cuda:0')
    variant = task.Variant(mode='pyramid', pyramid_plan='pyramid_robot6_plan.json',
                           preset_pieces=(), seed=manifest['seed'], appearance='original')
    spec = task.task_spec(variant, manifest['max_steps'], suite=manifest['suite'], terminate_on_success=False, contact_capacity=4096)
    kwargs = dict(graph=manifest['graph'], output_dir=manifest['episode'], variant=asdict(variant),
                  max_steps=manifest['max_steps'], observation='rgb-shape',
                  independent_verifier_path=str(ROOT / 'verification/bimanual_independent.py'))
    return [JobSpec(name='pyramid_unified', task=spec.full_name,
        solution='tools.unified_native:UnifiedPyramidGraph', solution_kwargs=tuple(kwargs.items()),
        episodes=(0, 1), seed=variant.seed, max_steps=manifest['max_steps'], device='cuda:0',
        control_mode='joint_position', renderer='',
        capture=CaptureSpec(enabled=True, every=1, contacts=True, tool_spans=True))]


def run(manifest_path):
    from robosimstudio_bench.run.jobs import load_jobs
    from robosimstudio_bench.run.pool import run_pooled
    manifest = json.loads(Path(manifest_path).read_text())
    result = run_pooled(load_jobs(manifest['jobfile']), root=manifest['native_root'],
                        workers=1, threads_per_worker=2, stagger=1., preload=manifest['jobfile'])
    output = {name: dict(rows=value.rows, unrun=list(value.unrun)) for name, value in result.items()}
    (Path(manifest_path).parent / 'native_results.json').write_text(json.dumps(output, indent=2) + '\n')
    return output
