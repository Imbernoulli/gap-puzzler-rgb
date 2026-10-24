"""Independent owner-cache physical checkpoints for an actual Cyan handoff.

This module is installed only by the native evaluator. Its private contact/GT
cache is never attached to the policy World, controller state or tool returns.
"""
from copy import deepcopy
import json

import numpy as np
from scipy.spatial.transform import Rotation

_PROVIDER = None
_ROUTES = {'handoff_approach': 'receiver_ready',
           'handoff_dualhold': 'receiver_closed',
           'handoff_receiveronly': 'donor_released',
           'handoff_receiver_hold': 'receiver_verified'}


def install(provider):
    global _PROVIDER
    if _PROVIDER is not None and _PROVIDER is not provider:
        raise RuntimeError('A transfer evaluator provider is already installed')
    _PROVIDER = provider
    for session in provider._SESSIONS.values():
        if hasattr(session, '_cyan_transfer_frames'):
            continue
        session._cyan_transfer_frames = {}
        session._cyan_transfer_progress = {}

        def observe(env, *unused, _session=session):
            step = int(env.step_count)
            if step in _session._cyan_transfer_frames:
                return
            contacts = env.contacts
            target = np.asarray(env.handles.entity('cyan').collision_shapes)
            fingers = [np.asarray([shape for group in env.handles.arms[arm].finger_groups
                                  for _, shapes in group for shape in shapes])
                       for arm in ('left', 'right')]
            external = True
            sender_contact = [False, False]
            if contacts is not None:
                contacts.update(step)
                a, b = contacts._shape_pairs()
                loaded = np.linalg.norm(contacts.impulses, axis=1) >= 1.e-6
                ta, tb = np.isin(a, target), np.isin(b, target)
                allowed = np.concatenate(fingers)
                external = bool(np.any(loaded & ((ta & ~np.isin(b, allowed))
                                                | (tb & ~np.isin(a, allowed)))))
                sender_contact = [bool(np.any(loaded & ((ta & np.isin(b, fs))
                                                       | (tb & np.isin(a, fs)))))
                                  for fs in fingers]
            with provider._LOCK:
                _session._cyan_transfer_frames[step] = dict(
                    step=step, no_external_support=not external,
                    named_finger_target_loaded=sender_contact)

        session._cyan_transfer_observer = observe
        observe(session.env)
        session.env.add_step_observer(observe)


def _dualhold(session, event, sender, receiver):
    provider = _PROVIDER
    count = int(round(.1 * session.hz)) + 1
    steps = range(event['step'] - count + 1, event['step'] + 1)
    recent = [session.frames[k] for k in steps if k in session.frames]
    contact_rows = [session._cyan_transfer_frames[k] for k in steps
                    if k in session._cyan_transfer_frames]
    continuous = (len(recent) == count and len(contact_rows) == count
                  and all(row['finite'] and row['held'][sender]
                          and row['held'][receiver] for row in recent)
                  and all(row['no_external_support'] for row in contact_rows))
    drifts = []
    for arm in (sender, receiver):
        relative = [(row['clouds']['cyan'] - row['hands'][arm, :3])
                    @ Rotation.from_quat(row['hands'][arm, 3:7]).as_matrix()
                    for row in recent]
        drifts.append(max((provider._error(cloud, relative[0]) for cloud in relative),
                          default=1.))
    frame = session.frames[event['step']]
    start = session.source_centroid_z.get('cyan', float(
        session.frames[min(session.frames)]['clouds']['cyan'][:, 2].mean()))
    rise = float(frame['clouds']['cyan'][:, 2].mean() - start)
    gaps = [float(np.linalg.norm(frame['clouds']['cyan'] - frame['hands'][arm, :3],
                                 axis=1).min()) for arm in (sender, receiver)]
    passed = bool(continuous and max(drifts) < .003 and rise >= .025 and max(gaps) <= .045)
    return passed, dict(continuous_both_hands_bilateral_axis_grasp=continuous,
                        required_window_s=.1, required_points=count,
                        actual_physical_points=len(recent), actual_contact_points=len(contact_rows),
                        both_hand_relative_bead_drifts_m=drifts,
                        actual_centroid_rise_m=rise, actual_target_hand_gaps_m=gaps,
                        permitted_support='Only the two named hands distal finger groups',
                        both_actual_grasp_sides=deepcopy(frame['grasp_contacts']))


def verify(phase, world, outputs):
    provider = _PROVIDER
    if provider is None or phase not in _ROUTES:
        raise RuntimeError('Transfer evaluator/phase is not installed')
    with provider._LOCK:
        event = provider._EVENTS.get(id(world))
        if event is None or event['world'] is not world:
            raise RuntimeError('No exact independent owner snapshot for this World')
        if phase in event['results']:
            return event['results'][phase]['passed']
        session = event['session']
        detail = dict(phase=phase, piece=event['piece'], native_step=event['step'],
                      policy_input=False, visual_diagnostics_affect_physical_result=False)
        passed = False
        try:
            state = outputs['state']
            transfer = state['transfer']
            sender, receiver = int(transfer['sender']), int(transfer['receiver'])
            if {sender, receiver} != {0, 1} or event['piece'] != 'cyan' or state['piece'] != 'cyan':
                raise ValueError('Transfer arms/target do not match the bound physical task')
            if outputs.get('decision', outputs.get('route')) != _ROUTES[phase]:
                raise ValueError('Transfer output route differs from its subgraph promise')
            prefix_ok, prefix_detail = session._prefix_check(event)
            detail.update(prefix_detail)
            frame = session.frames[event['step']]
            progress = session._cyan_transfer_progress
            if phase == 'handoff_approach':
                functional, held_detail = session._held_check(event, {**state, 'arm': sender})
                functional = functional and frame['hands'][receiver, 7] > .045
                detail.update(held_detail, receiver_actual_open_width_m=float(frame['hands'][receiver, 7]))
            elif phase == 'handoff_dualhold':
                if 'handoff_approach' not in progress:
                    raise ValueError('Receiver close lacks a passed prior approach checkpoint')
                functional, dual_detail = _dualhold(session, event, sender, receiver)
                detail.update(dual_detail)
            else:
                if 'handoff_dualhold' not in progress:
                    raise ValueError('Sender release lacks independently certified prior dual hold')
                if phase == 'handoff_receiver_hold' and 'handoff_receiveronly' not in progress:
                    raise ValueError('Receiver hold lacks a prior receiver-only checkpoint')
                functional, held_detail = session._held_check(event, {**state, 'arm': receiver})
                sender_gap = float(np.linalg.norm(frame['clouds']['cyan']
                                   - frame['hands'][sender, :3], axis=1).min())
                sender_clear = bool(frame['hands'][sender, 7] > .045 and sender_gap > .055)
                count = int(round(.1 * session.hz)) + 1
                recent = [session._cyan_transfer_frames[k]
                          for k in range(event['step'] - count + 1, event['step'] + 1)
                          if k in session._cyan_transfer_frames]
                only_receiver = bool(len(recent) == count and all(
                    row['no_external_support'] and not row['named_finger_target_loaded'][sender]
                    for row in recent))
                functional = bool(functional and sender_clear and only_receiver)
                detail.update(held_detail, donor_open_and_clear=sender_clear,
                              continuous_receiver_only_no_other_body_support=only_receiver,
                              receiver_only_contact_window_points=len(recent),
                              donor_actual_jaw_width_m=float(frame['hands'][sender, 7]),
                              donor_actual_target_gap_m=sender_gap)
                if phase == 'handoff_receiver_hold':
                    attachment = state.get('conditional_attachment') or {}
                    ownership = bool(state['arm'] == receiver and
                        (state.get('held_status') == 'CURRENT_RGB_VERIFIED' or
                         (attachment.get('arm_id') == receiver and
                          attachment.get('current_target_measured') is False)))
                    functional = functional and ownership
                    detail['new_receiver_binding_declared_without_relabelling_conditional_as_measurement'] = ownership
            passed = bool(prefix_ok and functional)
            detail.update(functional_postcondition_passed=bool(functional),
                          prefix_postcondition_passed=bool(prefix_ok))
            if passed:
                progress[phase] = event['step']
                session.begin = event['step']
        except Exception as error:
            detail['eval_error'] = f'{type(error).__name__}: {error}'
        detail['passed'] = passed
        event['results'][phase] = detail
        session.results.append(detail)
        with (session.output / 'checkpoint_results.jsonl').open('a') as stream:
            stream.write(json.dumps(provider._json_safe(detail), allow_nan=False) + '\n')
        return passed
