"""Bounded, default-disabled hook for repairing a currently measured released piece."""
from copy import deepcopy
import importlib
from typing import Any, TypedDict

from .modular_controller import ModularResult, ModularState
from .vision_stages import StageRefusal


class RepairAttempt(TypedDict, total=False):
    index: int
    source_rgb_timestamp: float
    source_frame_sha256: dict[str, str]
    robot_proprioception: dict[str, Any]
    module: str
    outcome: str
    evidence: dict[str, Any]


APPROVED_REPAIR_MODULES = frozenset({'puzzler.cyan_released_repair'})


def repair(ctx, state: ModularState) -> ModularResult:
    """Call one validated current-RGB/proprio/CAD repair, never a blind regrasp.

    Independent release-dispatch admission runs BEFORE this node when the
    feature is enabled. Its GT stays in the verifier; no GT values enter here.
    """
    state = deepcopy(state)
    parameters, name = state['parameters'], state['piece']
    if not parameters.get('enable_released_piece_repair', False):
        return dict(route='unknown', state=state, evidence=dict(kind='released_piece_repair', reason='Repair feature disabled'))
    attempts = deepcopy(state.get('repair_attempts', []))
    if len(attempts) >= 1:
        return dict(route='unknown', state=state, evidence=dict(kind='released_piece_repair', reason='One-repair budget exhausted'))
    module_name = parameters.get('release_repair_module')
    if module_name not in APPROVED_REPAIR_MODULES:
        raise StageRefusal('No qualified released-piece repair module configured')
    observed = state['observation']
    if not {'board', name}.issubset(observed.get('poses', {})):
        return dict(route='unknown', state=state, evidence=dict(kind='released_piece_repair', reason='Current board and target RGB are required for a repair action'))
    if observed.get('observation_source') not in ('rgb_shape_geometry', 'calibrated_rgb_cad_seeded'):
        raise StageRefusal('Repair action cannot use a predicted or fabricated current pose')
    if name in state.get('executed_placements', []) or state.get('execution_tokens', {}).get(name, {}).get('new_visual_release_receipt'):
        raise StageRefusal('A certified placement must not be repaired automatically')
    proprio = ctx.tool('robot.read_proprioception', arm_id=state['arm'])
    if float(proprio.get('width_m', 0.)) <= .045:
        raise StageRefusal('Repair requires an actually opened gripper after release')
    row: RepairAttempt = dict(index=0, source_rgb_timestamp=observed['timestamp'],
        source_frame_sha256=deepcopy(observed['frame_sha256']), robot_proprioception=proprio,
        module=module_name, outcome='attempt_started')
    attempts.append(row)
    state['repair_attempts'] = attempts
    module = importlib.import_module(module_name)
    result = module.repair(ctx, state)
    if result.get('route') not in ('placed', 'uncertain', 'missed', 'unknown'):
        raise StageRefusal('Released-piece repair returned an unsupported route')
    repaired = result['state']
    if repaired['piece'] != name or repaired.get('initialized_prefix', []) != state.get('initialized_prefix', []):
        raise StageRefusal('Repair changed task lineage')
    if repaired['parameters']['assembly_order'] != parameters['assembly_order']:
        raise StageRefusal('Repair changed assembly order')
    for earned in state.get('executed_placements', []):
        if earned not in repaired.get('executed_placements', []):
            raise StageRefusal('Repair discarded an earned predecessor')
        before = state['memory']['placements'].get(earned, {}).get('receipt')
        after = repaired.get('memory', {}).get('placements', {}).get(earned, {}).get('receipt')
        if before != after:
            raise StageRefusal('Repair changed predecessor receipt provenance')
    row.update(outcome=result['route'], evidence=deepcopy(result.get('evidence', {})))
    repaired['repair_attempts'] = attempts
    if result['route'] == 'placed':
        from .unified_contract import require_receipt
        require_receipt(repaired, name)
        repaired['last_outcome'] = 'placed'
    elif name in repaired.get('executed_placements', []):
        raise StageRefusal('A failed repair added an uncertified earned piece')
    return result
