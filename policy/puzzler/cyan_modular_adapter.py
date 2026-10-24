"""Cyan leaf skills with canonical earned-state ownership and typed graph I/O."""
from copy import deepcopy

from . import cyan_geometry as geometry, cyan_original_transport as transport
from .modular_controller import ModularState, ModularResult, result, save_observation
from .vision_stages import StageRefusal


def _prepare(state):
    if state['piece'] != 'cyan':
        raise StageRefusal('Cyan leaf adapter received another piece')
    prefix = [*state.get('initialized_prefix', []), *state.get('executed_placements', []), *state.get('released_placements', [])]
    return {**state, 'parameters': {**state['parameters'], 'conditional_prefix': prefix,
        'planning_prefix': prefix, 'actual_executed_prefix': list(state.get('executed_placements', [])),
        'given_prefix_names': list(state.get('initialized_prefix', []))}}


def _preserve_lineage(previous, updated):
    for key in ('initialized_prefix', 'executed_placements', 'released_placements', 'execution_tokens', 'belief_ledger'):
        if updated.get(key, {} if key.endswith(('tokens', 'ledger')) else []) != previous.get(key, {} if key.endswith(('tokens', 'ledger')) else []):
            raise StageRefusal('Cyan leaf changed canonical predecessor ' + key)
    for name in previous.get('executed_placements', []):
        before = previous['memory']['placements'][name]['receipt']
        after = updated['memory']['placements'][name]['receipt']
        if before != after:
            raise StageRefusal('Cyan leaf changed predecessor receipt provenance')
    return updated


def observe_and_plan(ctx, state: ModularState) -> ModularResult:
    prepared = _prepare(state)
    if prepared['parameters'].get('cyan_leaf_variant') == 'inverse_tip':
        from . import cyan_tip_adapter
        out = cyan_tip_adapter.observe_and_plan(ctx, prepared)
    else:
        out = geometry.observe_and_plan(ctx, prepared)
    updated = _preserve_lineage(state, out['state'])
    evidence = {k: v for k, v in out['evidence'].items() if k not in ('kind', 'timestamp', 'frame_sha256')}
    return result('planned', updated, updated['observation'], 'perceive_plan', **evidence)


def grasp_and_bind(ctx, state: ModularState) -> ModularResult:
    if state['parameters'].get('cyan_leaf_variant') == 'inverse_tip':
        from . import cyan_tip_adapter
        out = cyan_tip_adapter.grasp_and_bind(ctx, _prepare(state))
    else:
        out = transport.grasp_and_bind(ctx, _prepare(state))
    if out['route'] != 'verified':
        raise StageRefusal('Cyan transport requires actual fresh postlift RGB; conditional inspection exhausted')
    updated = _preserve_lineage(state, out['state'])
    evidence = {k: v for k, v in out['evidence'].items() if k not in ('kind', 'timestamp', 'frame_sha256')}
    return result('verified', updated, updated['observation'], 'grasp', **evidence)


def execute_transport(ctx, state: ModularState) -> ModularResult:
    state = _prepare(state)
    if state['parameters'].get('cyan_leaf_variant') == 'inverse_tip':
        from . import cyan_tip_adapter
        return cyan_tip_adapter.execute_transport(ctx, state)
    choice, candidates, belief = transport.carry_target(ctx, state)
    motion = transport.move_held_screened(ctx, state, choice, belief)
    updated = {**state, 'plan': {**state['plan'], 'target_R': choice['relative_target_R']},
        'geometry_carry': motion, 'goal_frame_candidates': candidates,
        'cyan_selected_goal': choice, 'transport_record': motion}
    return result('carried', updated, updated['observation'], 'transport', transport_record=motion)


def segment_conditional_release(ctx, state: ModularState) -> ModularResult:
    """Use the tip's original real pair; never overwrite a valid receipt."""
    from . import cyan_tip_adapter
    state = _prepare(state)
    out = cyan_tip_adapter.segment_conditional_release(ctx, state)
    _preserve_lineage(state, out['state'])
    return out
