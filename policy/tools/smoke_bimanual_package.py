"""Cold actual-base API and CUDA public-robot math smoke; never constructs Env."""
import ast
import importlib
import inspect
import json
from pathlib import Path
import tempfile
import types

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


def main(*, imports_only=False):
    module_names = [
        'puzzler.cyan_bimanual_adapter', 'puzzler.cyan_bimanual_common_adapter', 'puzzler.cyan_bimanual_common_fast_adapter', 'puzzler.cyan_receiver_delivery',
        'puzzler.cyan_receiver_motion', 'puzzler.red_bimanual_handoff',
        'puzzler.cyan_geometry', 'puzzler.cyan_vertical_source_v2',
        'puzzler.cyan_vertical_transfer_adapter', 'puzzler.cyan_bimanual_near_stream_adapter',
        'puzzler.cyan_vertical_transfer_v2_adapter', 'puzzler.cyan_bimanual_feedback_adapter',
        'puzzler.red_rgb_recovery',
        'puzzler.azure_rgb_attribution',
        'puzzler.cyan_receiver_release_feedback_delivery',
        'puzzler.cyan_receiver_motion_feedback', 'puzzler.cyan_dated_scene',
        'puzzler.board_vision', 'puzzler.shape_vision',
        'puzzler.red_bimanual_goal', 'puzzler.public_bimanual_geometry',
        'puzzler.public_bimanual_material_geometry',
        'verification.bimanual_independent', 'verification.cyan_transfer',
        'tools.unified_native',
        'tools.conditional_suffix_native',
    ]
    modules = [importlib.import_module(name) for name in module_names]
    missing = []
    checked = 0
    for module in modules:
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
                continue
            source = vars(module).get(node.value.id)
            if isinstance(source, types.ModuleType):
                # Warp resolves kernel-only builtins at compilation time.
                if source.__name__ == "warp" and node.attr == "tid":
                    continue
                checked += 1
                if not hasattr(source, node.attr):
                    missing.append(f'{module.__name__}:{node.lineno}: {node.value.id}.{node.attr}')
    if missing:
        raise AssertionError('Actual-base missing module API: ' + '; '.join(sorted(set(missing))))
    if imports_only:
        report = dict(status='PASS', modules_imported=module_names,
            module_attributes_checked=checked, environment_constructed=False,
            rendered=False, GPU_geometry_executed=False, physical_success_claim=False,
            scope='Cold actual-package imports and referenced module API only')
        (ROOT/'verification/package_import_smoke.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report))
        return
    from puzzler import red_bimanual_handoff as h
    from puzzler.public_bimanual_material_geometry import PublicBimanualMaterialGeometry
    from puzzler.vision_observer import goals
    from puzzler.vision_planning import finite_fk
    source = REPO / 'runs/pink_red_adjacent_gpu_v10/failed_ingress_evidence.json'
    archived = json.loads(source.read_text())
    actual = archived['last_actual_robot']['result']
    belief = archived['latest_authentic_held_belief']['result']
    q = np.asarray(actual['robot_encoder_values'])
    board = np.asarray([row['board_pose'] for row in archived['current_board_look_observations'] if row.get('board_pose')][-1])
    rb = Rotation.from_quat(board[3:])
    scene = dict(occupancy={name: dict(source='explicit_public_goal4mm_conditional_API_fixture',
        inflation_m=.004, pose_hypotheses=[np.r_[board[:3]+rb.apply(p),
        (rb*Rotation.from_matrix(r)).as_quat()].tolist()])
        for name, (p, r) in goals('pyramid', 'pyramid_robot6_plan.json').items()},
        measurement=dict(poses=dict(board=board.tolist())))
    geometry = h.geometry()
    assert geometry.device.is_cuda
    geometry.fk(q)
    tcps = {arm: finite_fk(geometry.robot.iks[geometry._arm(arm)], q) for arm in (0, 1)}
    for arm in (0, 1):
        home = geometry.robot.iks[geometry._arm(arm)].home_pose
        assert np.asarray(home[0]).shape == (3,) and np.asarray(home[1]).shape == (4,)
    class PublicStub:
        def tool(self, name, **kwargs):
            arm = kwargs.get('arm_id', 1)
            p, r = tcps[arm]
            if name == 'robot.read_proprioception':
                return {**actual, 'tcp': [*p.tolist(), *r.tolist()],
                    'width_coordinates': [8*arm+6, 8*arm+7],
                    'width_m': float(q[8*arm+6]-q[8*arm+7])}
            if name == 'puzzle.plan_pose_sequence':
                sample = dict(position=p.tolist(), rotation=dict(zip('xyzw', r.tolist())))
                return dict(feasible=True, min_joint_limit_margin_rad=.035,
                    path=[q[8*arm:8*arm+6].tolist()]*2, tcp_path=[sample]*2)
            raise AssertionError('No Env or motor tool permitted in API smoke: ' + name)
    p, r = tcps[1]
    point = h._waypoint(p, Rotation.from_quat(r).as_matrix())
    result = h._screen(PublicStub(), dict(piece='red', arm=1), scene, belief, 1, 1, [point], {1: [4]})
    assert result['guard']['safe'] and result['guard']['held_envelope_m'] == .006
    larger = h._screen(PublicStub(), dict(piece='red', arm=1), scene,
        {**belief, 'uncertainty_m': .008}, 1, 1, [point], {1: [4]})
    assert larger['guard']['safe'] and larger['guard']['held_envelope_m'] == .008
    material = PublicBimanualMaterialGeometry(device='cuda:0', minimum_clearance_m=.0005)
    assert material.device.is_cuda
    dense = material.densify_q16_path([q.tolist(), q.tolist()])
    assert dense.shape[1] == 16
    from tools.unified_native import jobs
    with tempfile.TemporaryDirectory(prefix='puzzler_bimanual_api_') as tmp:
        path = Path(tmp) / 'manifest.json'
        path.write_text(json.dumps(dict(initialization='fresh_canonical_all12_loose',
            graph=str(ROOT/'graphs/pyramid_original_order_bimanual_v7'), episode=str(Path(tmp)/'episode'),
            seed=0, max_steps=30000, suite='bimanual_full12_api_smoke', device='cuda:0')))
        job, = jobs(path)
        assert job.device == 'cuda:0' and job.renderer == ''
    from unittest.mock import patch
    from puzzler import red_bimanual_goal as goal
    retained = dict(uncertainty_m=.008)
    actual_state = dict(piece='red', arm=1, observation={})
    refreshed_state = {**actual_state, 'planning_scene': {'actual_reacquired': True}}
    scene = {'actual_reacquired': True}
    with patch.object(goal.handoff, '_fresh_sender', return_value=(actual_state, retained, {'status':'actual_RGB_q16_rebound'})), \
         patch.object(goal.handoff, 'scene_for_motion', return_value=(refreshed_state, scene, {'status':'new_actual_board_RGB'})) as scene_helper:
        refreshed = goal._refresh(None, actual_state, 'actual_package_helper_contract', optimizer_belief=retained)
        assert scene_helper.call_count == 1 and refreshed[2] is scene
        assert refreshed[3]['scene_refresh']['status'] == 'new_actual_board_RGB'
    prepared = ROOT/'runs/canonical_full12_bimanual_v7/manifest.json'
    prepared_job = None
    if prepared.exists():
        from tools.launch_unified import _fresh_graph
        manifest = json.loads(prepared.read_text())
        _fresh_graph(Path(manifest['graph']))
        prepared_job, = jobs(prepared)
        assert prepared_job.device == 'cuda:0' and prepared_job.max_steps == 60000
        assert manifest['initial_ledgers_empty'] and not manifest['initialized_prefix']
        assert manifest['piece_phase_count'] == 83 and manifest['contact_capacity'] == 4096
        assert manifest['native_run_started'] is False
    report = dict(status='PASS', actual_base=inspect.getfile(h.base),
        modules_imported=module_names, module_attributes_checked=checked,
        public_robot_device=str(geometry.device), native_job_device=job.device,
        prepared_job_max_steps=prepared_job.max_steps if prepared_job else None,
        prepared_manifest_fresh_all12_empty_ledgers_verified=prepared_job is not None,
        actual_archived_q16_source=str(source), archived_robot_timestamp=actual['timestamp'],
        public_bimanual_iks='left/right keys selected through actual _arm mapping',
        full_scene_static_screen_6mm=True, full_scene_static_screen_8mm=True,
        uncertainty_preserved_m=.008, environment_constructed=False, rendered=False,
        physical_checkpoint_or_handoff_success_claim=False,
        scope='Cold packaged actual-base imports and CUDA robot-only scratch math. Archived RGB/q16 fixture and stationary planner stub; no native rollout, control or physics success claim.')
    (ROOT/'verification/package_api_smoke.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imports-only', action='store_true')
    main(imports_only=parser.parse_args().imports_only)
