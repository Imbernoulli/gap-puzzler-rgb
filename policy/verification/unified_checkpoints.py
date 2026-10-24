"""Evaluator-only branch checks; never imported by policy node scripts."""
import json
from functools import wraps

import numpy as np


def _with_provider_lock(function):
    @wraps(function)
    def locked(provider, world, outputs):
        # _Session._observe appends under this exact RLock. Read every frame
        # window and event binding under it as the original validate() does.
        with provider._LOCK:
            return function(provider, world, outputs)
    return locked


def _event(provider, world):
    event = provider._EVENTS.get(id(world))
    if event is None or event['world'] is not world:
        raise RuntimeError('No owner-thread independent snapshot is bound to this World')
    return event


def released_admission(frame, *, piece, arm, prefix_passed, no_earned_piece,
                       authentic_journal_open, honest_rgb_pair, budget_available):
    """Pure offline-testable admission predicate; no placement threshold waived."""
    hands = np.asarray(frame['hands'], dtype=float)
    cloud = np.asarray(frame['clouds'][piece], dtype=float)
    separated = bool(hands.shape == (2, 8) and cloud.ndim == 2 and cloud.shape[1] == 3
        and np.isfinite(hands).all() and np.isfinite(cloud).all()
        and min(float(np.linalg.norm(cloud - hand[:3], axis=1).min()) for hand in hands) > .055)
    conditions = dict(valid_arm=arm in (0, 1), actual_jaw_open=bool(arm in (0, 1) and hands[arm, 7] > .045),
        actual_target_released=not bool(frame['robot_loaded'][piece]),
        target_separated_from_both_hands=separated, finite_scene=bool(frame['finite']),
        original_board_gate=bool(frame['board_ok']), protected_prefix_safe=bool(prefix_passed),
        target_not_earned=bool(no_earned_piece), journal_open_completed=bool(authentic_journal_open),
        honest_actual_rgb_pair=bool(honest_rgb_pair), finite_budget=bool(budget_available))
    return all(conditions.values()), conditions


def _write(session, detail):
    session.results.append(detail)
    with (session.output / 'checkpoint_results.jsonl').open('a') as stream:
        stream.write(json.dumps(detail, allow_nan=False) + '\n')


@_with_provider_lock
def release_dispatch(provider, world, outputs):
    state = outputs['state']
    if outputs.get('decision') == 'missed':
        from puzzler.unified_contract import conditional_handoff_allowed, terminal_visual_unknown_allowed
        if (not state['parameters'].get('enable_released_piece_repair', False)
                or conditional_handoff_allowed(state) or terminal_visual_unknown_allowed(state)):
            event = _event(provider, world)
            session = event['session']
            detail = dict(phase='release_observed_visual_miss', piece=event['piece'],
                          native_step=event['step'], policy_input=False,
                          observed_visual_decision='missed', visual_receipt_earned=False)
            passed = False
            try:
                if event['piece'] != state['piece']:
                    raise ValueError('Observed visual miss belongs to another module')
                prefix_ok, pd = session._prefix_check(event)
                stable, sd = session._stable(event, state['piece'], int(state['arm']))
                passed = bool(prefix_ok and stable)
                detail.update({**pd, **sd})
                if passed:
                    committed_release = dict(piece=state['piece'], step=event['step'],
                        rgb_timestamp=outputs['evidence'].get('timestamp', -1.), active_arm=int(state['arm']))
                    session.begin, session.last_release = event['step'], committed_release
            except Exception as error:
                passed = False
                detail['eval_error'] = f'{type(error).__name__}: {error}'
            detail['passed'] = passed
            _write(session, detail)
            return passed
    if outputs.get('decision') in ('executed_visual_unknown', 'executed_visual_unverified'):
        from puzzler.unified_contract import terminal_visual_unknown_allowed, terminal_visual_status, conditional_handoff_allowed
        event = _event(provider, world)
        session = event['session']
        detail = dict(phase='conditional_release', piece=event['piece'], native_step=event['step'], policy_input=False,
                      visual_status=terminal_visual_status(state, piece=state['piece'])['status'], visual_receipt_earned=False)
        passed = False
        try:
            if event['piece'] != state['piece'] or not (conditional_handoff_allowed(state) or terminal_visual_unknown_allowed(state)):
                raise ValueError('Conditional execution handoff contract was not satisfied')
            prefix_ok, prefix_detail = session._prefix_check(event)
            stable, stable_detail = session._stable(event, state['piece'], int(state['arm']))
            passed = bool(prefix_ok and stable)
            detail.update({**prefix_detail, **stable_detail})
            if passed:
                committed_release = dict(piece=state['piece'], step=event['step'],
                    rgb_timestamp=outputs['evidence'].get('timestamp', -1.), active_arm=int(state['arm']))
                session.begin, session.last_release = event['step'], committed_release
        except Exception as error:
            passed = False
            detail['eval_error'] = f'{type(error).__name__}: {error}'
        detail['passed'] = passed
        _write(session, detail)
        return passed
    # Native repair qualification has not yet passed, so the shipped default
    # still uses the original strict release postcondition.
    if outputs.get('decision') == 'placed' or not state['parameters'].get('enable_released_piece_repair', False):
        return provider.validate('release', world, outputs)
    event = _event(provider, world)
    session, piece = event['session'], event['piece']
    detail = dict(phase='release_dispatch', piece=piece, native_step=event['step'], policy_input=False)
    passed = False
    try:
        if outputs.get('decision') not in ('uncertain', 'missed') or state['piece'] != piece:
            raise ValueError('Invalid released-repair branch')
        prefix_ok, prefix_detail = session._prefix_check(event)
        record = state.get('release_execution', {})
        pair = record.get('release_rgb_pair', [])
        now = event['step'] / session.hz
        honest_pair = len(pair) == 2 and all(
            isinstance(row.get('timestamp'), (float, int)) and row['timestamp'] <= now
            and row.get('frame_sha256') and row.get('observation_source') in ('rgb_shape_geometry', 'calibrated_rgb_cad_seeded')
            for row in pair) and pair[0]['timestamp'] < pair[1]['timestamp']
        mark = record.get('execution_mark', {})
        nonce = world.sim_state.get('robot_command_session')
        journal = world.sim_state.get('robot_command_journal', [])
        returns = {row['call_id'] for row in journal if row.get('event') == 'return' and row.get('session_nonce') == nonce}
        opened = bool(mark.get('session_nonce') == nonce and any(
            row.get('event') == 'call' and row.get('session_nonce') == nonce
            and row.get('tool') == 'robot.set_grip' and row.get('call_id', -1) > mark.get('call_id', float('inf'))
            and row['call_id'] in returns and row.get('arguments', {}).get('arm_id') == state['arm']
            and row.get('arguments', {}).get('width_m', 0.) > .00475 for row in journal))
        passed, conditions = released_admission(session.frames[event['step']], piece=piece, arm=state['arm'],
            prefix_passed=prefix_ok, no_earned_piece=piece not in state.get('executed_placements', [])
                and not state.get('execution_tokens', {}).get(piece, {}).get('new_visual_release_receipt'),
            authentic_journal_open=opened, honest_rgb_pair=honest_pair,
            budget_available=len(state.get('repair_attempts', [])) < 1)
        detail.update(prefix_detail, conditions=conditions, intermediate_target_goal_error_m=session.frames[event['step']]['errors'][piece],
                      placement_success_claim=False)
        if passed:
            session.begin = event['step']
    except Exception as error:
        passed = False
        detail['eval_error'] = f'{type(error).__name__}: {error}'
    detail['passed'] = passed
    _write(session, detail)
    return passed


@_with_provider_lock
def finish(provider, world, outputs):
    from puzzler import model
    from puzzler.unified_contract import require_final_ledger
    event = _event(provider, world)
    session = event['session']
    detail = dict(phase='finish', piece=None, native_step=event['step'], policy_input=False)
    passed = False
    try:
        state = outputs['state']
        all_names = set(model.CELLS)
        ledger = require_final_ledger(state)
        ledger_ok = (outputs.get('decision') == 'completed'
            and session.index == len(session.order) and set(session.prefix) == all_names)
        prefix_ok, prefix_detail = session._prefix_check(event)
        last = session.last_release or {}
        stable, stable_detail = session._stable(event, last['piece'], last['active_arm'])
        passed = bool(ledger_ok and prefix_ok and stable)
        detail.update({**prefix_detail, **stable_detail}, receipt_ledger=ledger,
                      requested_modules_lineage=ledger_ok,
                      all12_independent_module_lineage=ledger_ok and not ledger['initialized_prefix'] and len(session.order)==12)
    except Exception as error:
        passed = False
        detail['eval_error'] = f'{type(error).__name__}: {error}'
    detail['passed'] = passed
    _write(session, detail)
    return passed
