"""Canonical graph joins and receipt-only handoff; no simulator state access."""
from copy import deepcopy

from . import modular_controller as controller, visual_memory
from .modular_controller import ModularResult, ModularState
from .vision_stages import StageRefusal


def require_receipt(state, expected_piece):
    """Recompute the actual RGB certificate before accepting an earned piece."""
    if state.get('piece') != expected_piece:
        raise StageRefusal('Handoff piece differs from its graph module')
    token = state.get('execution_tokens', {}).get(expected_piece, {})
    receipt = token.get('new_visual_release_receipt')
    if not isinstance(receipt, dict) or receipt.get('piece') != expected_piece:
        raise StageRefusal('Execution without an authentic RGB receipt is not earned')
    checked = visual_memory.release_receipt(expected_piece, receipt['before'], receipt['after'],
                                           receipt['released_tcp_position'], state['parameters'])
    if checked != receipt:
        raise StageRefusal('Placement receipt fails recomputation')
    track = state.get('memory', {}).get('placements', {}).get(expected_piece, {})
    if track.get('receipt') != checked or track.get('status') == 'unknown_or_conflicted':
        raise StageRefusal('Receipt is absent from memory or has been invalidated')
    if expected_piece not in state.get('executed_placements', []):
        raise StageRefusal('Receipt was not committed to the earned ledger')
    if state.get('belief_ledger', {}).get(expected_piece, {}).get('status') == 'current_rgb_conflict_goal_envelope_invalidated':
        raise StageRefusal('Current RGB invalidated this placement')
    return checked


TERMINAL_CONTEST_MARGIN_M = .004  # the belief ledger's goal envelope (modular_controller.save_observation)


def _confirmed_by_current_pixels(state, name):
    """Completion-check counts for `name` in the latest observation, if they confirm it."""
    from .assembly_completion import MIN_SHARE
    row = state.get('observation', {}).get('assembly_completion', {}).get('per_piece', {}).get(name)
    if not row:
        return None
    decisive = row['match'] + row['mismatch']
    return row if row['match'] >= 1 and row['match'] >= MIN_SHARE * decisive else None


def require_terminal_receipt(state, name):
    """Earned receipt for the final ledger, allowing a bounded later contest.

    Under the finished pyramid, lower pieces are partly hidden and their late
    RGB fits (and the case fit) drift by a few millimetres. A receipt that was
    authentic at release stays usable when later RGB contests it by less than
    the 4 mm goal envelope that execution-only pieces already use. The contest
    is recorded, and the independent physical finish check decides the goal.
    """
    try:
        require_receipt({**state, 'piece': name}, name)
        return dict(piece=name, status='receipt_verified')
    except StageRefusal as refusal:
        confirmed = _confirmed_by_current_pixels(state, name)
        if confirmed is not None and name in state.get('executed_placements', []):
            # A later direct look at the goal sites outranks an earlier noisy
            # contradiction: the current frame shows this piece where it belongs.
            return dict(piece=name, status='receipt_confirmed_by_current_goal_pixels',
                        earlier_refusal=str(refusal), current_goal_pixels=confirmed)
        track = state.get('memory', {}).get('placements', {}).get(name, {})
        if (track.get('status') != 'unknown_or_conflicted'
                or track.get('invalidation_reason') not in (visual_memory.BOARD_CONFLICT, visual_memory.PIECE_CONFLICT)):
            raise
    token = state.get('execution_tokens', {}).get(name, {})
    receipt = token.get('new_visual_release_receipt')
    if not isinstance(receipt, dict) or receipt.get('piece') != name or track.get('receipt') != receipt:
        raise StageRefusal('Contested receipt is not the committed release receipt: ' + name)
    if visual_memory.release_receipt(name, receipt['before'], receipt['after'],
                                     receipt['released_tcp_position'], state['parameters']) != receipt:
        raise StageRefusal('Contested receipt fails recomputation: ' + name)
    if name not in state.get('executed_placements', []):
        raise StageRefusal('Contested receipt was not committed to the earned ledger: ' + name)
    if state.get('belief_ledger', {}).get(name, {}).get('status') == 'current_rgb_conflict_goal_envelope_invalidated':
        raise StageRefusal('Current RGB places this piece outside the goal envelope: ' + name)
    evidence = track.get('conflict_evidence') or {}
    sizes = [evidence.get(key) for key in ('board_displacement_m', 'goal_error_m')]
    if not evidence or any(size is not None and size >= TERMINAL_CONTEST_MARGIN_M for size in sizes):
        raise StageRefusal('Later RGB contests this receipt beyond the goal envelope: ' + name)
    return dict(piece=name, status='receipt_contested_within_goal_envelope',
                invalidation_reason=track['invalidation_reason'], conflict_evidence=evidence,
                margin_m=TERMINAL_CONTEST_MARGIN_M)


def _completion_scope(state):
    from . import model
    order = state['parameters']['assembly_order']
    given = state.get('initialized_prefix', [])
    if state['parameters'].get('diagnostic_prefix_finish', False):
        if (not given or not state['parameters'].get('diagnostic_subset')
                or set(given) & set(order) or len(set(given)) != len(given)
                or set(given + order) != set(model.CELLS)):
            raise StageRefusal('Diagnostic finish needs explicit disjoint given-prefix and target coverage')
    elif given or len(order) != 12 or set(order) != set(model.CELLS):
        raise StageRefusal('Full completion requires a fresh all12 order and no given prefix')
    return order, given


def execution_coverage(state):
    return set(state.get('executed_placements', [])) | set(state.get('released_placements', []))


def require_execution_hypothesis(state, piece):
    token = state.get('execution_tokens', {}).get(piece, {})
    ledger = state.get('belief_ledger', {}).get(piece, {})
    hypothesis = ledger.get('execution_hypothesis', {})
    if (piece not in state.get('released_placements', []) or not token
            or not hypothesis or hypothesis.get('piece') != piece
            or hypothesis.get('source') != 'dated_RGB_board_times_public_goal_conditional_execution'
            or not hypothesis.get('source_board_rgb', {}).get('frame_sha256')
            or not hypothesis.get('conditional_goal_margin_m') == .004):
        raise StageRefusal('Actual execution has no dated public-goal hypothesis: ' + piece)
    return hypothesis


def conditional_handoff_allowed(state):
    name = state['piece']
    return name in state['parameters'].get('conditional_execution_handoff_pieces', [])


def commit_execution_hypothesis(state):
    """Record execution and a conditional goal proposal, never a measured pose."""
    from scipy.spatial.transform import Rotation
    import numpy as np
    from .vision_observer import goals
    state = deepcopy(state)
    name = state['piece']
    token = state.get('execution_tokens', {}).get(name, {})
    pair = token.get('release_rgb_pair', [])
    if not token or len(pair) != 2 or any(not row.get('frame_sha256') for row in pair):
        raise StageRefusal('Conditional handoff needs actual release execution and RGB provenance')
    if pair[1]['timestamp'] <= pair[0]['timestamp']:
        raise StageRefusal('Conditional release observations did not advance')
    if token.get('new_visual_release_receipt') is not None or name in state.get('memory', {}).get('placements', {}):
        raise StageRefusal('A visual receipt must remain in its distinct earned ledger')
    rows = [*state.get('rgb_history', []), *pair]
    now = state['observation']['timestamp']
    candidates = [row for row in rows if 'board' in row.get('poses', {})
                  and 0 <= now - row['timestamp'] <= 30.]
    board_row = max(candidates, key=lambda row: row['timestamp']) if candidates else None
    if board_row is None:
        raise StageRefusal('No dated accepted board RGB for conditional execution occupancy')
    board = np.asarray(board_row['poses']['board'])
    rotation = Rotation.from_quat(board[3:]).as_matrix()
    center, relative = goals(state['parameters']['mode'], state['parameters']['pyramid_plan'])[name]
    pose = np.r_[board[:3] + rotation @ center, Rotation.from_matrix(rotation @ relative).as_quat()].tolist()
    visual = terminal_visual_status(state, piece=name)
    hypothesis = dict(piece=name, source='dated_RGB_board_times_public_goal_conditional_execution',
        source_board_rgb=dict(poses={'board': list(board_row['poses']['board'])},
            timestamp=board_row['timestamp'], frame_sha256=deepcopy(board_row['frame_sha256']),
            observation_source=board_row['observation_source']),
        public_goal_pose=pose, conditional_goal_margin_m=.004,
        board_uncertainty_m=0. if board_row['timestamp'] == now else .004,
        is_current_measurement=False, visual_receipt_earned=False,
        visual_status=visual, independent_release_and_prefix_checkpoint_required=True)
    ledger = deepcopy(state.get('belief_ledger', {}))
    prior = ledger.get(name, {})
    ledger[name] = {**prior, 'status': prior['status'] if prior.get('status') == 'current_rgb_conflict_goal_envelope_invalidated'
                    else 'executed_visual_unverified_conditional_goal', 'execution_hypothesis': hypothesis,
                    'placement_verified': False}
    released = list(state.get('released_placements', []))
    if name not in released:
        released.append(name)
    state.update(released_placements=released, belief_ledger=ledger,
        last_outcome='executed_visual_unknown' if visual['status'] == 'UNKNOWN' else 'executed_visual_unverified')
    return state, visual


def terminal_visual_unknown_allowed(state):
    """Only final Red may finish with unverified vision and an execution token."""
    parameters = state['parameters']
    try:
        order, given = _completion_scope(state)
    except StageRefusal:
        return False
    if (not parameters.get('terminal_visual_receipt_optional', False)
            or state['piece'] != 'red' or order[-1] != 'red'):
        return False
    if execution_coverage(state) - {'red'} != set(order[:-1]):
        return False
    try:
        for name in order[:-1]:
            if name in state.get('executed_placements', []):
                require_terminal_receipt(state, name)
            else:
                require_execution_hypothesis(state, name)
    except (StageRefusal, visual_memory.MemoryRefusal, KeyError):
        return False
    token = state.get('execution_tokens', {}).get('red', {})
    pair = token.get('release_rgb_pair', [])
    return bool(token and 'red' not in state.get('memory', {}).get('placements', {})
        and token.get('new_visual_release_receipt') is None
        and len(pair) == 2 and all(row.get('frame_sha256') and isinstance(row.get('timestamp'), (int, float))
            and row.get('observation_source') in ('rgb_shape_geometry', 'calibrated_rgb_cad_seeded') for row in pair)
        and pair[1]['timestamp'] > pair[0]['timestamp'])


def terminal_visual_status(state, piece='red'):
    pair = state['execution_tokens'][piece]['release_rgb_pair']
    visible = [row for row in pair if {'board', piece}.issubset(row.get('poses', {}))]
    errors = [visual_memory.goal_error(piece, row, state['parameters']) for row in visible]
    current = state.get('observation', {})
    if current.get('timestamp', -1.) >= pair[-1]['timestamp']:
        if not {'board', piece}.issubset(current.get('poses', {})):
            return dict(status='UNKNOWN', current_pair_target_available=False, visual_goal_errors_m=errors,
                        initial_pair_visible_conflict=any(error>=.003 for error in errors))
        current_error = visual_memory.goal_error(piece, current, state['parameters'])
        if current_error >= .003:
            return dict(status='observed_visual_miss', current_pair_target_available=True,
                        visual_goal_errors_m=errors, current_visual_goal_error_m=current_error)
        return dict(status='observed_pair_unverified', current_pair_target_available=True,
                    visual_goal_errors_m=errors, current_visual_goal_error_m=current_error,
                    initial_pair_visible_conflict=any(error>=.003 for error in errors))
    if any(error >= .003 for error in errors):
        return dict(status='observed_visual_miss', current_pair_target_available=len(visible) == 2,
                    visual_goal_errors_m=errors)
    if len(visible) < 2:
        return dict(status='UNKNOWN', current_pair_target_available=False, visual_goal_errors_m=errors)
    return dict(status='observed_pair_unverified', current_pair_target_available=True, visual_goal_errors_m=errors)


def require_final_ledger(state):
    from . import model
    names = set(model.CELLS)
    order, given = _completion_scope(state)
    terminal_unknown = terminal_visual_unknown_allowed(state)
    required = set(order)
    earned = state.get('executed_placements', [])
    if execution_coverage(state) != required or len(earned) != len(set(earned)):
        raise StageRefusal('Final execution coverage is incomplete or visual ledger duplicates a piece')
    receipts = {}
    for name in required:
        if name in earned:
            receipts[name] = require_terminal_receipt(state, name)
        else:
            require_execution_hypothesis(state, name)
    return dict(receipt_names=sorted(earned), execution_only_names=sorted(required - set(earned)),
                contested_receipts={name: row for name, row in receipts.items()
                                    if row['status'] != 'receipt_verified'},
                initialized_prefix=list(given), full_assembly_eligible=not given,
                terminal_visual_status=terminal_visual_status(state)['status'] if terminal_unknown else 'receipt_verified',
                terminal_execution_token_present=bool(state.get('execution_tokens', {}).get('red')),
                red_visual_receipt_earned=not terminal_unknown)


def settle_placement_evidence(ctx, state: ModularState) -> ModularResult:
    """One canonical result for release and its bounded perception repair.

    A verified release passes unchanged. An uncertain release invokes the
    bounded active RGB/CAD observer once, returning its NEW state. Unknown
    occupancy does not authorize regrasping a potentially seated object.
    """
    state = deepcopy(state)
    piece = state['piece']
    initial_visual_status = state.get('release_visual_status', state.get('last_outcome'))
    if state.get('last_outcome') == 'placed':
        receipt = require_receipt(state, piece)
        observed = state['observation']
        return dict(route='placed', state=state, evidence=dict(
            kind='receipt_settle', timestamp=observed['timestamp'],
            frame_sha256=observed['frame_sha256'], receipt=receipt,
            source='Actual release RGB pair; no additional motion requested'))
    if state.get('last_outcome') not in ('uncertain', 'missed'):
        raise StageRefusal('Evidence join requires placed, uncertain or observed missed release')
    state['release_visual_status'] = initial_visual_status
    if conditional_handoff_allowed(state):
        observed = state['observation']
        conflict = ({'board', piece}.issubset(observed.get('poses', {}))
                    and visual_memory.goal_error(piece, observed, state['parameters']) >= .004)
        if conflict:
            repaired = controller.reobserve_and_replan(ctx, state)
            repaired['evidence']['initial_release_visual_status'] = initial_visual_status
            if repaired['route'] == 'placed':
                require_receipt(repaired['state'], piece)
                return repaired
            state = repaired['state']
            current = state['observation']
            if ({'board', piece}.issubset(current.get('poses', {}))
                    and visual_memory.goal_error(piece, current, state['parameters']) >= .004):
                repaired['evidence']['unresolved_current_visual_conflict'] = True
                return repaired
    if conditional_handoff_allowed(state) or terminal_visual_unknown_allowed(state):
        state, visual = commit_execution_hypothesis(state)
        route = state['last_outcome']
        state['pending_placement_receipt'] = dict(piece=piece, target_3d_status=visual['status'],
            visual_diagnostics=visual,
            terminal_only=piece == 'red', visual_receipt_earned=False, independent_final_verification_required=True)
        observed = state['observation']
        return dict(route=route, state=state, evidence=dict(
            kind='conditional_release_execution', timestamp=observed['timestamp'], frame_sha256=observed['frame_sha256'],
            visual_status=visual['status'], visual_diagnostics=visual, visual_receipt_earned=False,
            independent_final_verification_required=True))
    if int(state.get('attempt', 0)) >= 1:
        return dict(route='unknown', state=state, evidence=dict(
            kind='receipt_settle', reason='Bounded active observation repair exhausted'))
    out = controller.reobserve_and_replan(ctx, state)
    out['evidence']['initial_release_visual_status'] = initial_visual_status
    if out['route'] == 'placed':
        require_receipt(out['state'], piece)
    elif out['route'] != 'unknown':
        raise StageRefusal('No recovery was validated for route ' + out['route'])
    elif state['parameters'].get('enable_released_piece_repair', False):
        from .released_piece_repair import repair
        out = repair(ctx, out['state'])
        if out['route'] != 'placed':
            out['route'] = 'unknown'
    return out


def verified_handoff(ctx, state: ModularState, expected_piece: str) -> ModularResult:
    if state.get('last_outcome') in ('executed_visual_unknown', 'executed_visual_unverified'):
        if not (conditional_handoff_allowed(state) or (expected_piece == 'red' and terminal_visual_unknown_allowed(state))):
            raise StageRefusal('Execution-only handoff is not enabled for this module')
        require_execution_hypothesis(state, expected_piece)
        return controller.handoff_piece(ctx, state, expected_piece)
    require_receipt(state, expected_piece)
    if state.get('last_outcome') != 'placed':
        raise StageRefusal('Handoff requires the latest canonical outcome to be placed')
    return controller.handoff_piece(ctx, state, expected_piece)


def finish_assembly(ctx, state: ModularState) -> ModularResult:
    """Withdraw both hands, observe the final scene and require all real receipts."""
    from . import model, vision_controller as control
    from .vision_stages import _pause
    names = set(model.CELLS)
    final_ledger = require_final_ledger(state)
    for arm in (0, 1):
        ctx.tool('robot.set_grip', arm_id=arm, width_m=.055)
        ctx.tool('robot.go_home', arm_id=arm)
    first_epoch = ctx.tool('robot.read_proprioception', arm_id=0)
    _pause(ctx, 110)
    last_epoch = ctx.tool('robot.read_proprioception', arm_id=0)
    if last_epoch['timestamp'] - first_epoch['timestamp'] < 2.:
        raise StageRefusal('Final native settling interval was shorter than two seconds')
    # The policy confirms the finished pyramid itself: every bead site the
    # four cameras should see must show its part. A hand or glare can blur
    # one frame, so it looks up to three times before refusing.
    looks = []
    for _ in range(3):
        observed = ctx.tool('vision.observe', hints={'__completion_check__': True})
        looks.append(observed.get('assembly_completion', {}))
        if looks[-1].get('complete'):
            break
        _pause(ctx, 25)
    if not looks[-1].get('complete'):
        raise StageRefusal('Final RGB does not show a complete pyramid: '
                           + ', '.join(looks[-1].get('unconfirmed_pieces', []) or [looks[-1].get('reason', 'unclear')]))
    state = controller.save_observation(state, observed)
    final_ledger = require_final_ledger(state)
    state = {**state, 'last_stage': 'finish', 'last_outcome': 'completed'}
    return dict(route='completed', state=state, evidence=dict(kind='finish',
        timestamp=observed['timestamp'], frame_sha256=observed['frame_sha256'],
        actual_tail_interval_s=last_epoch['timestamp'] - first_epoch['timestamp'],
        earned_pieces=final_ledger['receipt_names'], terminal_visual_status=final_ledger['terminal_visual_status'],
        contested_receipts=final_ledger['contested_receipts'],
        visual_completion={key: value for key, value in looks[-1].items() if key != 'views'},
        visual_completion_looks=len(looks),
        red_visual_receipt_earned=final_ledger['red_visual_receipt_earned'],
        independent_all12_physical_verification_required=True))
