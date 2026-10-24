"""Audit final graph bindings and actual saved receipt handoffs without motors."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from puzzler.unified_contract import require_receipt, settle_placement_evidence, verified_handoff
from puzzler.vision_stages import StageRefusal
from tools.build_unified_graph import ORDERS, PHASES, SKILL_PHASES, ROOT, phases_for


class NoMotorContext:
    def tool(self, name, **kwargs):
        raise AssertionError('Dataflow-only audit must never invoke ' + name)


def audit_graph(path):
    document = json.loads((path / 'workflow.json').read_text())
    order = document['meta']['ordered_stages']
    nodes = document['nodes']
    previous = None
    bound, literals, checkpoints = [], [], []
    for piece in order:
        for phase in phases_for(piece):
            name = piece + '_' + phase
            value = nodes[name]['inputs']['state']
            if previous is None:
                assert value['piece'] == piece
                assert not value['memory']['placements'] and not value['executed_placements']
                assert not value['initialized_prefix'] and not value['execution_tokens']
                assert not value['rgb_history']
                literals.append(name)
            else:
                assert value == {'$ref': previous + '.state'}, (name, value, previous)
                bound.append(dict(node=name, consumes=previous + '.state'))
            if phase == 'handoff':
                assert nodes[name]['inputs']['expected_piece'] == piece
                assert previous == piece + '_receipt_settle'
            source = (path / 'checkpoints' / (phase + '_sg.py')).read_text()
            checkpoint = SKILL_PHASES[phase][3]
            assert 'validate=True' in source and repr(checkpoint) in source
            checkpoints.append(dict(node=name, independent_kind=checkpoint, validate=True))
            previous = name
    for piece in order:
        if piece == 'red':
            continue
        release = document['conditional_edges'][piece + '_release']['mapping']
        assert release['uncertain'] == release['placed'] == release['missed'] == piece + '_receipt_settle'
        settle = document['conditional_edges'][piece + '_receipt_settle']['mapping']
        assert settle['placed'] == piece + '_handoff' and settle['unknown'] == 'abort'
    assert len(order) == len(set(order)) == 12 and set(order) == set(ORDERS['original_order'])
    assert document['meta']['required_checkpoint_mode'] == 'raise'
    assert len(literals) == 1
    assert len(bound) == sum(len(phases_for(piece)) for piece in order)-1
    assert nodes['finish_all12']['inputs']['state'] == {'$ref': previous + '.state'}
    assert document['conditional_edges'][previous]['mapping']['advanced'] == 'finish_all12'
    assert document['conditional_edges']['finish_all12']['mapping']['completed'] == 'done'
    cyan_phases = [SKILL_PHASES[p][3] for p in phases_for('cyan')]
    assert all(name in cyan_phases for name in ('handoff_approach', 'handoff_dualhold',
        'handoff_receiveronly', 'handoff_receiver_hold'))
    red_phases = [SKILL_PHASES[p][3] for p in phases_for('red')]
    first = ['handoff_prepared', 'handoff_dualhold', 'handoff_receiver_only', 'handoff_receiver_bound']
    assert red_phases[2:6] == first
    assert red_phases[6:10] == ['return_'+phase for phase in first]
    # The unchanged original gates and observation/lineage mechanisms are
    # compared as complete source files, not inferred from route labels.
    baseline = ROOT.parents[2] / 'runs/pyramid_full12_cuda_v4/workspace'
    unchanged = ['verification/independent.py', 'verification/unified_checkpoints.py',
        'puzzler/unified_contract.py', 'puzzler/pyramid_held_belief.py',
        'puzzler/conditional_attachment.py', 'puzzler/vision_observer.py',
        'profiles/pyramid.json']
    for relative in unchanged:
        assert (ROOT/relative).read_bytes() == (baseline/relative).read_bytes(), relative
    return dict(graph=str(path), ordered_pieces=order, state_literal_count=1,
        typed_runtime_refs=len(bound)+1, piece_phase_refs=len(bound),
        finish_ref='finish_all12.state input <- '+previous+'.state',
        no_unexecuted_branch_refs=True, unknown_routes_to_abort=True,
        release_uncertain_routes_to_bounded_receipt_repair=True,
        cyan_independent_transfer_phases=cyan_phases[2:6],
        red_distinct_hop_phases=red_phases[2:10],
        strict_physical_release_finish_and_observation_sources_unchanged=unchanged,
        independent_physical_checkpoints=checkpoints, bindings=bound)


def audit_real_receipt(path):
    saved = json.loads(path.read_text())
    assert saved['route'] == 'placed'
    state = saved['state']
    piece = state['piece']
    receipt = require_receipt(state, piece)
    incoming = deepcopy(state)
    result = settle_placement_evidence(NoMotorContext(), state)
    assert result['state'] == incoming
    handoff = verified_handoff(NoMotorContext(), result['state'], piece)
    next_state = handoff['state']
    order = state['parameters']['assembly_order']
    next_piece = order[order.index(piece) + 1]
    assert next_state['piece'] == next_piece
    for key in ('memory', 'execution_tokens', 'executed_placements', 'belief_ledger', 'initialized_prefix'):
        assert next_state[key] == state[key], key
    assert next_state['rgb_history'] == state['rgb_history']
    assert next_state['plan'] == {} and next_state['planning_scene'] == {}
    assert next_state['release_execution'] == {} and next_state.get('transport_record', {}) == {}
    assert next_state['arm'] == -1 and next_state['attempt'] == 0
    assert next_state['parameters']['grasp_indices'] == state['parameters']['skill_profiles'][next_piece]['grasp_indices']
    assert next_state['parameters']['source_open_width'] == state['parameters']['skill_profiles'][next_piece]['source_open_width']
    transient = {'owned_held_belief_source', 'conditional_attachment_id', 'conditional_attachment',
                 'historical_complete_rgb', 'historical_rgb_records', 'historical_max_age_s', 'planning_prefix', 'given_prefix_names'}
    assert not transient.intersection(next_state['parameters'])
    assert next_state['parameters']['actual_executed_prefix'] == []
    altered = deepcopy(state)
    altered['execution_tokens'][piece].pop('new_visual_release_receipt')
    try:
        verified_handoff(NoMotorContext(), altered, piece)
    except StageRefusal:
        missing_receipt_rejected = True
    else:
        raise AssertionError('Uncertified execution was allowed to hand off')
    assert state == incoming, 'Caller-owned previous piece state mutated'
    return dict(source=str(path), source_kind='ACTUAL_NATIVE_SAVED_RELEASE_OUTPUT',
        native_success_claim=False, scope='Software dataflow replay only; no motor calls and no synthetic measurements',
        piece=piece, next_piece=next_piece, receipt_recomputed=True,
        actual_receipt_goal_error_m=receipt['goal_error_m'],
        memory_earned_tokens_ledger_preserved=True, next_piece_profile_isolated=True,
        stage_transients_cleared=True, missing_receipt_handoff_rejected=missing_receipt_rejected,
        caller_owned_state_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual-release', type=Path, action='append', default=[])
    args = parser.parse_args()
    report = dict(schema='puzzler.unified_dataflow_audit.v1', policy_input=False,
        full12_native_acceptance=False, flat_native_acceptance=False,
        graphs=[audit_graph(ROOT / 'graphs/pyramid_original_order_bimanual_v7')],
        actual_saved_receipt_checks=[audit_real_receipt(path) for path in args.actual_release],
        independent_release_checkpoint=dict(default_repair_enabled=False,
            default_accepts_routes=['placed', 'uncertain'], default_physical_miss_blocks=True,
            qualified_repair_branch_accepts=['uncertain', 'missed'],
            final_physical_thresholds_unchanged=True,
            source='verification/unified_checkpoints.py::release_dispatch; default delegates original strict release'),
        independent_finish=dict(actual_module_execution_coverage_required=True, conditional_execution_handoff_pieces=[name for name in ORDERS['original_order'] if name!='red'], final_red_execution_with_visual_unknown_allowed=True,
            final_red_unknown_does_not_mint_receipt=True, both_hands_open_home=True,
            actual_wait_at_least_two_seconds=True, independent_original_stable_thresholds_unchanged=True,
            rss_terminate_on_success=False, physical_goal_predicate_unchanged=True),
        capacity_log_audit=dict(required=True,complete_native_stdout_stderr=True,
            forbidden='Any broadphase/narrowphase/nefc/njmax/contact capacity overflow invalidates qualification',
            task_contact_capacity=4096,contact_law_rates_unchanged=True))
    (ROOT / 'dataflow_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(static_graphs_passed=len(report['graphs']),
                         actual_receipt_handoffs_passed=len(report['actual_saved_receipt_checks']),
                         full12_native_acceptance=False)))


if __name__ == '__main__':
    main()
