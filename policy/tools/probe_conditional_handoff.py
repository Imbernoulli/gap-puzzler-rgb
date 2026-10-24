"""No-motor dataflow checks using an actual saved released-Pink RGB UNKNOWN."""
from copy import deepcopy
import json
from pathlib import Path

from puzzler import modular_controller as controller
from puzzler.unified_contract import settle_placement_evidence, verified_handoff, require_execution_hypothesis
from tools.audit_unified_dataflow import NoMotorContext
from tools.build_unified_graph import ROOT


def main():
    source = Path('/srv/home/bohanlyu/gap-puzzler-web/runs/pink_strict_rehearsal_v1/workspace/runs/pink_prefix10_v1/episode/trace/0000/node_data/release_sg.act/iters/000/output.json')
    state = json.loads(source.read_text())['state']
    assert state['piece'] == 'pink' and state['last_outcome'] == 'uncertain'
    assert state['execution_tokens']['pink']['new_visual_release_receipt'] is None
    # Public graph configuration is changed; actual RGB/q16/pair/token data
    # remain byte-equivalent to the saved native module outcome.
    state['parameters']['conditional_execution_handoff_pieces'] = ['navy', 'pink']
    state['parameters']['assembly_order'] = ['pink', 'red']
    state['parameters']['skill_profiles'] = json.loads((ROOT / 'profiles/pyramid.json').read_text())['profiles']
    preserved = deepcopy(state)
    outcome = settle_placement_evidence(NoMotorContext(), state)
    updated = outcome['state']
    assert outcome['route'] in ('executed_visual_unknown', 'executed_visual_unverified')
    assert updated['executed_placements'] == state['executed_placements']
    assert updated['memory'] == state['memory'] and 'pink' not in updated['memory']['placements']
    assert updated['execution_tokens'] == state['execution_tokens']
    assert updated['released_placements'] == ['pink']
    hypothesis = require_execution_hypothesis(updated, 'pink')
    assert not hypothesis['is_current_measurement'] and not hypothesis['visual_receipt_earned']
    advanced = verified_handoff(NoMotorContext(), updated, 'pink')['state']
    assert advanced['piece'] == 'red' and advanced['released_placements'] == ['pink']
    assert advanced['belief_ledger']['pink']['execution_hypothesis'] == hypothesis
    assert advanced['execution_tokens'] == state['execution_tokens'] and advanced['memory'] == state['memory']
    assert state == preserved
    # A geometry-only negative fixture perturbs the estimated Pink pose. It
    # never mints a receipt, drives a motor, or claims to be a new RGB frame.
    fixture = deepcopy(updated['observation'])
    if 'pink' not in fixture['poses']:
        fixture['poses']['pink'] = deepcopy(hypothesis['public_goal_pose'])
    fixture['poses']['pink'][0] += .020
    conflicted = controller.save_observation(updated, fixture)
    assert conflicted['belief_ledger']['pink']['status'] == 'current_rgb_conflict_goal_envelope_invalidated'
    assert conflicted['belief_ledger']['pink']['execution_hypothesis'] == hypothesis
    scene = controller.scene_belief(conflicted, fixture)
    assert scene['occupancy']['pink']['source'] == 'current_rgb_after_goal_envelope_invalidation'
    assert scene['belief_ledger']['pink']['execution_hypothesis'] == hypothesis
    reacquired_fixture = deepcopy(fixture)
    reacquired_fixture['poses']['pink'] = deepcopy(hypothesis['public_goal_pose'])
    reacquired = controller.save_observation(conflicted, reacquired_fixture)
    assert reacquired['belief_ledger']['pink']['status'] == 'reacquired_current_goal_compatible'
    assert reacquired['belief_ledger']['pink']['execution_hypothesis'] == hypothesis
    report = dict(schema='puzzler.actual_conditional_handoff_dataflow_probe.v1', source=str(source),
        positive_input='Actual saved native Pink release UNKNOWN; only public graph config changed',
        motor_calls=0, native_success_claim=False, receipt_created=False,
        actual_execution_token_retained=True, released_ledger_separate_from_visual_earned=True,
        current_raw_poses_unchanged_by_positive_handoff=True, next_piece='red',
        dated_public_goal_hypothesis_only=True, no_GT_policy_input=True,
        invalidation_fixture='Explicit20mm estimated-pose perturbation for software checks only; not a measurement',
        conflict_keeps_provenance_but_disables_goal_occupancy=True,
        reacquisition_preserves_execution_provenance=True)
    (ROOT / 'conditional_handoff_probe.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
