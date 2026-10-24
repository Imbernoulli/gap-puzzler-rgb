"""Author the typed full pyramid with independently checked physical regrasp states."""
import argparse
import json
from pathlib import Path

from gap.builder import Ref, Subgraph, Workflow
from puzzler.modular_controller import initial_state
from puzzler.modular_verification import verify

ROOT = Path(__file__).resolve().parents[1]
ORDERS = {
    'original_order': ('azure', 'orange', 'yellow', 'teal', 'purple', 'cyan',
                       'mint', 'lime', 'maroon', 'navy', 'pink', 'red'),
}
# Each entry is a real typed subgraph with an independent raise checkpoint.
PHASES = {
    'perceive_plan': ('puzzler.modular_controller', 'observe_and_plan', ['planned'], 'perceive_plan'),
    'grasp': ('puzzler.modular_controller', 'grasp_and_bind', ['verified'], 'grasp'),
    'transport': ('puzzler.modular_controller', 'transport', ['carried'], 'transport'),
    'release': ('puzzler.modular_controller', 'carry_and_release', ['placed', 'uncertain', 'missed'], 'release_dispatch'),
    'receipt_settle': ('puzzler.unified_contract', 'settle_placement_evidence', ['placed', 'unknown', 'executed_visual_unknown', 'executed_visual_unverified'], 'release_dispatch'),
    'handoff': ('puzzler.unified_contract', 'verified_handoff', ['advanced'], 'handoff'),
}
CYAN_PHASES = {
    'cyan_approach': ('puzzler.cyan_vertical_transfer_v2_adapter', 'approach_receiver', ['receiver_ready'], 'handoff_approach'),
    'cyan_receiver_close': ('puzzler.cyan_vertical_transfer_v2_adapter', 'close_receiver', ['receiver_closed'], 'handoff_dualhold'),
    'cyan_donor_release': ('puzzler.cyan_vertical_transfer_v2_adapter', 'release_donor', ['donor_released'], 'handoff_receiveronly'),
    'cyan_receiver_bind': ('puzzler.cyan_vertical_transfer_v2_adapter', 'verify_receiver', ['receiver_verified'], 'handoff_receiver_hold'),
    'cyan_receiver_transport': ('puzzler.cyan_receiver_release_feedback_delivery', 'execute_transport', ['carried'], 'transport'),
}
RED_PHASES = {
    'red_handoff_prepare': ('puzzler.red_bimanual_handoff', 'handoff_prepare', ['prepared'], 'handoff_prepared'),
    'red_receiver_close': ('puzzler.red_bimanual_handoff', 'receiver_close', ['dual_held'], 'handoff_dualhold'),
    'red_sender_release': ('puzzler.red_bimanual_handoff', 'sender_release', ['transferred'], 'handoff_receiver_only'),
    'red_receiver_bind': ('puzzler.red_bimanual_handoff', 'receiver_bind', ['verified'], 'handoff_receiver_bound'),
    'red_return_prepare': ('puzzler.red_bimanual_handoff', 'return_handoff_prepare', ['prepared'], 'return_handoff_prepared'),
    'red_return_close': ('puzzler.red_bimanual_handoff', 'receiver_close', ['dual_held'], 'return_handoff_dualhold'),
    'red_return_release': ('puzzler.red_bimanual_handoff', 'sender_release', ['transferred'], 'return_handoff_receiver_only'),
    'red_return_bind': ('puzzler.red_bimanual_handoff', 'receiver_bind', ['verified'], 'return_handoff_receiver_bound'),
    'red_goal_transport': ('puzzler.red_bimanual_goal', 'carry', ['carried'], 'transport'),
    'red_goal_release': ('puzzler.red_bimanual_goal', 'release', ['placed', 'uncertain', 'missed'], 'release_dispatch'),
    'red_goal_verify': ('puzzler.red_bimanual_goal', 'verify_release', ['advanced'], 'handoff'),
}
SKILL_PHASES = {**PHASES, **CYAN_PHASES, **RED_PHASES,
    'finish': ('puzzler.unified_contract', 'finish_assembly', ['completed'], 'finish')}


def phases_for(piece):
    if piece == 'cyan':
        return ['perceive_plan', 'grasp', *CYAN_PHASES, 'release', 'receipt_settle', 'handoff']
    if piece == 'red':
        return ['perceive_plan', 'grasp', *RED_PHASES]
    return list(PHASES)


def build(destination, *, order_name='original_order', pieces=None, given_prefix=()):
    dest = Path(destination).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    (dest / 'scripts').mkdir()
    (dest / 'checkpoints').mkdir()
    order = tuple(pieces) if pieces is not None else ORDERS[order_name]
    given_prefix = tuple(given_prefix)
    if (not order or len(order) != len(set(order)) or set(order) & set(given_prefix)
            or set(order + given_prefix) - set(ORDERS['original_order'])):
        raise ValueError('Unique known targets and a disjoint explicit diagnostic prefix are required')
    full_coverage = set(order + given_prefix) == set(ORDERS['original_order'])
    profiles = json.loads((ROOT / 'profiles/pyramid.json').read_text())['profiles']
    # Isolate the reviewed vertical source and low handoff proposals to Cyan.
    # Original strict grasp and the four independent transfer gates remain.
    profiles['cyan'].update(json.loads((ROOT / 'profiles/cyan_vertical_bimanual_v1.json').read_text()))
    profiles['cyan']['release_module'] = 'puzzler.cyan_receiver_release_feedback_delivery'
    # Apply only the qualified public source-recipe fields. Keep the baseline
    # profile, quaternion, Maroon MAP and all other Red guards unchanged.
    red_source = json.loads((ROOT / 'profiles/red_public_source_v7_four_fields.json').read_text())
    profiles['red']['object_relative_grasp_se3'].update(red_source['object_relative_grasp_se3'])
    profiles['red']['source_contact_grasp'].update(red_source['source_contact_grasp'])
    profiles['red']['source_profile_provenance'] = red_source['source_profile_provenance']
    parameters = dict(mode='pyramid', pyramid_plan='pyramid_robot6_plan.json', anchor=False,
        observation_required='rgb-shape', conditional_planning=True,
        skip_preopen_rgb=False, allow_conditional_board_persistence=True,
        min_ik_joint_margin_rad=.035, max_attempts_per_piece=1, max_reobserve_attempts=1,
        enable_released_piece_repair=False, terminal_visual_receipt_optional=True,
        conditional_execution_handoff_pieces=[name for name in order if name != 'red'],
        assembly_order=list(order), diagnostic_subset=list(order) if pieces is not None or given_prefix else [],
        diagnostic_prefix_finish=bool(given_prefix) and full_coverage, skill_profiles=profiles)
    (dest / 'params.json').write_text(json.dumps(parameters, indent=2) + '\n')
    (dest / 'scripts/schema.py').write_text(
        'from gap_core.schema import TYPE_REGISTRY\n'
        'from puzzler.modular_controller import ModularState, ModularResult\n'
        "TYPE_REGISTRY['dict'] = dict\n")
    wf = Workflow(name=dest.name,
        description='RGB/proprio/public CAD pyramid with separate physical Cyan/Red transfer checkpoints')
    used_phases = {phase for piece in order for phase in phases_for(piece)}
    if full_coverage:
        used_phases.add('finish')
    for phase, (module, function, routes, checkpoint) in SKILL_PHASES.items():
        if phase not in used_phases:
            continue
        sg = Subgraph(name=phase + '_sg', skill='modular-rgb-puzzler')
        sg.add_input('state', type_name='ModularState')
        inputs = {'state': Ref('in.state')}
        extra_arg, extra_call = '', ''
        if phase == 'handoff':
            sg.add_input('expected_piece', type_name='string')
            inputs['expected_piece'] = Ref('in.expected_piece')
            extra_arg, extra_call = ', expected_piece: str', ', expected_piece'
        sg.add_node('act', type='script', script='scripts/' + phase + '.py', inputs=inputs)
        sg.add_edge('START', 'act')
        sg.add_edge('act', 'END')
        sg.set_exit_router(router_field='route', success_values=routes)
        sg.set_on_error('error')
        sg.set_outputs(state=Ref('act.state'), evidence=Ref('act.evidence'), decision=Ref('act.route'))
        sg.add_checkpoint(phase + '_independent_postcondition',
            lambda world, outputs, kind=checkpoint: verify(kind, world, outputs),
            rationale='Independent native physical verification outside policy observation boundary', validate=True)
        wf.add_subgraph(sg)
        release_dispatch = (
            "    if state['piece'] == 'cyan':\n"
            "        parameters = {**state['parameters'], 'release_module': 'puzzler.cyan_receiver_release_feedback_delivery'}\n"
            "        state = {**state, 'parameters': parameters}\n" if phase == 'release' else '')
        (dest / 'scripts' / (phase + '.py')).write_text(
            'from puzzler.modular_controller import ModularState, ModularResult\n'
            f'from {module} import {function}\n'
            f'def run(ctx, state: ModularState{extra_arg}) -> ModularResult:\n'
            + release_dispatch +
            f'    return {function}(ctx, state{extra_call})\n')
        (dest / 'checkpoints' / (phase + '_sg.py')).write_text(
            'from gap.runtime.verify import Checkpoint\n'
            'from puzzler.modular_verification import verify\n'
            f'def verified(world, outputs):\n    return verify({checkpoint!r}, world, outputs)\n'
            f'CHECKPOINTS = [Checkpoint(name={phase + "_independent_postcondition"!r}, '
            f'subgraph={phase + "_sg"!r}, predicate=verified, validate=True, '
            'rationale="Independent native physical verification outside policy inputs")]\n')
    wf.add_node('done', type='end', status='success')
    wf.add_node('abort', type='end', status='failure')
    previous, previous_phase = None, None
    for piece in order:
        for phase in phases_for(piece):
            node = piece + '_' + phase
            inputs = {'state': initial_state(piece, parameters, given_prefix=given_prefix)
                      if previous is None else Ref(previous + '.state')}
            if phase == 'handoff':
                inputs['expected_piece'] = piece
            wf.add_node(node, type='subgraph', ref=phase + '_sg', inputs=inputs)
            if previous is None:
                wf.add_edge('START', node)
            else:
                routes = SKILL_PHASES[previous_phase][2]
                mapping = {route: node for route in routes if route != 'unknown'}
                mapping.update({'error': 'abort'})
                if 'unknown' in routes:
                    mapping['unknown'] = 'abort'
                wf.add_conditional_edges(previous, mapping, router_field='exit')
            previous, previous_phase = node, phase
    if full_coverage:
        wf.add_node('finish_all12', type='subgraph', ref='finish_sg', inputs={'state': Ref(previous + '.state')})
        wf.add_conditional_edges(previous, {'advanced': 'finish_all12', 'error': 'abort'}, router_field='exit')
        wf.add_conditional_edges('finish_all12', {'completed': 'done', 'error': 'abort'}, router_field='exit')
    else:
        wf.add_conditional_edges(previous, {'advanced': 'done', 'error': 'abort'}, router_field='exit')
    wf.save(dest / 'workflow.json')
    document = json.loads((dest / 'workflow.json').read_text())
    document['meta'].update(required_checkpoint_mode='raise', schema_module_path='scripts/schema.py',
        skill_registry_paths=[str(ROOT / 'registry_modular'), str(ROOT / 'third_party/open-robot-skills')],
        observation_mode='on_demand', ordered_stages=list(order), order_name=order_name,
        phase_schedule={piece: phases_for(piece) for piece in order},
        source_qualification='Frozen Cyan v6 continuous5–7 physical22/22 PASS; fixed Red V19 plus public V7 four-field source overlay, continuous10–12 placement pending',
        state_flow='Exactly one initial_state; every later node consumes Ref(previous.state)',
        regrasp_contract='Cyan four physical phases; Red two distinct four-phase hops. Receiver actual close q16/FK binds dated authentic RGB, never a desired transform. Opening revokes old sender source.',
        receipt_contract='Authentic RGB alone earns placements. Explicit execution-only conditional hypotheses preserve UNKNOWN/conflicts and require unchanged physical gates.',
        finish_contract='All12 execution/receipt lineage, both hands open/home and >=2s actual wait, unchanged independent physical completion gates',
        task_termination='terminate_on_success=False preserves final finish; original physical goal unchanged',
        contact_capacity=4096, contact_law_and_rates='Original RSS MuJoCo500Hz; capacity storage only',
        initialization='Explicit diagnostic given-prefix fixture; never full assembly acceptance' if given_prefix else 'Fresh RSS canonical reset, all12 loose, empty ledger; no start_poses, presets, tape or GT policy inputs',
        initialized_prefix=list(given_prefix), full_assembly_eligible=not given_prefix and full_coverage,
        full12_acceptance=False, flat_acceptance=False)
    (dest / 'workflow.json').write_text(json.dumps(document, indent=2) + '\n')
    return dest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', required=True)
    parser.add_argument('--order', choices=ORDERS, default='original_order')
    args = parser.parse_args()
    print(build(args.dest, order_name=args.order))
