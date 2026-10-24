"""Explicit offline boundary fixtures; no Env, render, motors or object truth."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


def main():
    from puzzler import pyramid_held_belief as held, conditional_attachment as conditional
    from tools import unified_native
    from verification import bimanual_independent, independent
    # Reuse the reviewed labelled dependency fixture, execute this workspace's
    # unchanged actual registration and RestrictedRegistry implementations.
    source = REPO/'runs/owned_held_opening_invalidation_v1/API_probe.py'
    tree = ast.parse(source.read_text())
    wanted = {'Registry', 'Bus', 'Array', 'RobotFixture', 'fixture', 'refuses', 'load_restricted'}
    definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                   and node.name in wanted]
    namespace = dict(ast=ast, deepcopy=deepcopy, hashlib=hashlib, json=json,
        Path=Path, np=np, SimpleNamespace=SimpleNamespace)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), namespace)
    restricted = namespace['load_restricted'](ROOT/'puzzler/vision_policy.py')
    fixture, refuses = namespace['fixture'], namespace['refuses']
    checks = {}
    fk = lambda ik, seed: (np.array([0., 0., 1.]), np.array([0., 0., 0., 1.]))
    seed = lambda env: np.zeros(16, np.float32)
    with patch.object(held, 'finite_fk', side_effect=fk), patch.object(held, 'robot_seed', side_effect=seed), \
         patch.object(conditional, 'finite_fk', side_effect=fk), patch.object(conditional, 'robot_seed', side_effect=seed):
        for donor in (0, 1):
            receiver = 1-donor
            f = fixture(held, restricted)
            f.robot.controllers = SimpleNamespace(get=lambda name: SimpleNamespace(
                target_action=lambda control: np.ones(14)))
            f.robot.control = None
            conditional.register(f.registry, f.robot, f.bus, SimpleNamespace(records=f.records))
            f.restricted = restricted(f.registry)
            source_rgb = f.archive(10)
            owned = lambda arm: f.restricted.invoke('puzzle.propagate_pyramid_held_belief', source=source_rgb, arm_id=arm)
            attachments = {arm: f.restricted.invoke('puzzle.capture_conditional_attachment',
                source_rgb=source_rgb, arm_id=arm, assumption=dict(
                    source_stationary_or_bounded_until_close=True, bead_displacement_bound_m=.008))
                for arm in (donor, receiver)}
            propagate = lambda arm: f.restricted.invoke('puzzle.propagate_conditional_attachment',
                attachment_id=attachments[arm]['attachment_id'], arm_id=arm)
            owned(donor); owned(receiver)
            checks[f'actual_close_epoch_receiver_{receiver}'] = (
                attachments[receiver]['arm_id'] == receiver
                and attachments[receiver]['close_proprio_epoch']['step'] == 11
                and attachments[receiver]['current_target_measured'] is False)
            f.restricted.invoke('robot.set_grip', arm_id=donor, width_m=.055)
            checks[f'donor_owned_revoked_{donor}'] = refuses(lambda: owned(donor))
            checks[f'donor_conditional_revoked_{donor}'] = refuses(lambda: propagate(donor))
            checks[f'receiver_owned_survives_donor_open_{receiver}'] = not refuses(lambda: owned(receiver))
            checks[f'receiver_conditional_survives_donor_open_{receiver}'] = not refuses(lambda: propagate(receiver))
            f.restricted.invoke('robot.set_grip', arm_id=receiver, width_m=.055)
            checks[f'receiver_postopen_owned_revoked_{receiver}'] = refuses(lambda: owned(receiver))
            checks[f'receiver_postopen_conditional_revoked_{receiver}'] = refuses(lambda: propagate(receiver))
            f.restricted.invoke('robot.set_grip', arm_id=receiver, width_m=0.)
            checks[f'receiver_reclose_cannot_revive_owned_{receiver}'] = refuses(lambda: owned(receiver))
            checks[f'receiver_reclose_cannot_revive_conditional_{receiver}'] = refuses(lambda: propagate(receiver))
            fresh = f.archive(f.robot._step)
            forged = {**fresh, 'pose': [.020, 0., 1., 0., 0., 0., 1.]}
            checks[f'new_owned_epoch_allowed_{receiver}'] = not refuses(lambda: f.restricted.invoke(
                'puzzle.propagate_pyramid_held_belief', source=fresh, arm_id=receiver))
            checks[f'unarchived_desired_transform_refused_{receiver}'] = refuses(lambda: f.restricted.invoke(
                'puzzle.propagate_pyramid_held_belief', source=forged, arm_id=receiver))
    # Compile the real nested evaluator adapter with recorder-only providers;
    # shared dualhold labels must resolve by target identity, never by chance.
    tree = ast.parse((ROOT/'tools/unified_native.py').read_text())
    adapter = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == 'Adapter')
    calls = []
    provider = SimpleNamespace(validate=lambda kind, world, outputs: calls.append(('red_or_original', kind)) or True)
    cyan = SimpleNamespace(verify=lambda kind, world, outputs: calls.append(('cyan', kind)) or True)
    ns = dict(provider=provider, cyan_transfer=cyan, CYAN_TRANSFER_PHASES=unified_native.CYAN_TRANSFER_PHASES)
    exec(compile(ast.Module(body=[adapter], type_ignores=[]), '<actual_combined_adapter>', 'exec'), ns)
    obj = ns['Adapter']()
    for piece, expected in [('cyan', 'cyan'), ('red', 'red_or_original')]:
        obj.verify('handoff_dualhold', object(), {'state': {'piece': piece}})
        checks[f'dualhold_{piece}_uses_distinct_physical_evaluator'] = calls[-1] == (expected, 'handoff_dualhold')
    for kind in ['handoff_approach', 'handoff_receiveronly', 'handoff_receiver_hold']:
        obj.verify(kind, object(), {'state': {'piece': 'cyan'}})
        checks['cyan_dispatch_'+kind] = calls[-1] == ('cyan', kind)
    for kind in ['handoff_prepared', 'handoff_receiver_only', 'handoff_receiver_bound',
                 'return_handoff_prepared', 'return_handoff_dualhold',
                 'return_handoff_receiver_only', 'return_handoff_receiver_bound']:
        obj.verify(kind, object(), {'state': {'piece': 'red'}})
        checks['red_dispatch_'+kind] = calls[-1] == ('red_or_original', kind)
    checks['exact_owner_cache_shared'] = (bimanual_independent._SESSIONS is independent._SESSIONS
        and bimanual_independent._EVENTS is independent._EVENTS
        and bimanual_independent._LOCK is independent._LOCK)
    assert all(checks.values()), {key: value for key, value in checks.items() if not value}
    report = dict(status='PASS', checks=checks, cases=len(checks),
        actual_package=str(ROOT), fixture_source=str(source),
        scope='Explicit labelled offline robot/archive dependencies. Actual source/conditional registrations and nested evaluator dispatch; no Env, renderer, physics, real motor, object truth, fabricated camera receipt, or physical success claim.')
    (ROOT/'verification/bimanual_boundary_checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', cases=len(checks), physical_success_claim=False)))


if __name__ == '__main__':
    main()
