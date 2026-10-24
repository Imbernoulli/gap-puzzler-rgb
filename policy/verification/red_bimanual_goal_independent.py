"""Independent physical checkpoints for two real Red handoffs and original goal delivery.

This owner-thread-only extension never adds privileged fields to policy state,
World, tools, RGB packets or controller inputs. Original source/localization,
held, protected-prefix, release and finish predicates remain in independent.py.
"""
from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation

from verification import independent as base

FIRST_ROUTES = dict(handoff_prepared='prepared', handoff_dualhold='dual_held',
                    handoff_receiver_only='transferred', handoff_receiver_bound='verified')
RETURN_ROUTES = {'return_'+name: route for name, route in FIRST_ROUTES.items()}
ROUTES = {**FIRST_ROUTES, **RETURN_ROUTES}


class _Session(base._Session):
    def _observe(self, env, *unused):
        super()._observe(env, *unused)
        step = int(env.step_count)
        frame = self.frames[step]
        if 'bimanual_forbidden_support' in frame or self.index >= len(self.order):
            return
        piece = self.order[self.index]
        contacts = env.contacts
        forbidden = False
        partners = []
        if contacts is not None:
            contacts.update(step)
            a, b = contacts._shape_pairs()
            positive = np.linalg.norm(contacts.impulses, axis=1) >= 1.e-6
            target = np.asarray(env.handles.entity(piece).collision_shapes)
            # Exactly the two directly driven distal-finger groups per hand.
            # All palms, links, camera/wrist, table, board and other pieces are
            # forbidden support, even though the other hand may hold the piece.
            allowed = np.asarray([s for arm in ('left', 'right')
                for group in env.handles.arms[arm].finger_groups for _, ss in group for s in ss])
            aa, bb = np.isin(a, target), np.isin(b, target)
            external = positive & ((aa & ~np.isin(b, target)) | (bb & ~np.isin(a, target)))
            partners = [int(bb if on_a else aa) for aa, bb, on_a in
                zip(a[external], b[external], np.isin(a[external], target))]
            forbidden = bool(any(p not in allowed for p in partners))
        frame['bimanual_forbidden_support'] = forbidden
        frame['bimanual_target_contact_partner_shapes'] = partners

    def _roles(self, state):
        handoff = state.get('handoff', {})
        sender = int(handoff['sender'])
        receiver = int(handoff['receiver'])
        if {sender, receiver} != {0, 1}:
            raise ValueError('Handoff requires distinct actual sender and receiver arms')
        return sender, receiver

    def _dual_hold(self, event, state):
        sender, receiver = self._roles(state)
        count = int(round(.1*self.hz))+1
        recent = [self.frames[k] for k in range(event['step']-count+1, event['step']+1) if k in self.frames]
        piece = event['piece']
        continuous = len(recent) == count and all(all(row['held'])
            and not row['table_loaded'][piece] and not row.get('bimanual_forbidden_support', True)
            for row in recent)
        drifts, gaps = [], []
        for arm in (sender, receiver):
            attached = [(row['clouds'][piece]-row['hands'][arm, :3]) @
                Rotation.from_quat(row['hands'][arm, 3:7]).as_matrix() for row in recent]
            drifts.append(max((base._error(v, attached[0]) for v in attached), default=1.))
            f = self.frames[event['step']]
            gaps.append(float(np.linalg.norm(f['clouds'][piece]-f['hands'][arm, :3], axis=1).min()))
        f = self.frames[event['step']]
        start = self.source_centroid_z.get(piece, float(self.frames[min(self.frames)]['clouds'][piece][:, 2].mean()))
        rise = float(f['clouds'][piece][:, 2].mean()-start)
        passed = bool(continuous and max(drifts)<.003 and max(gaps)<=.045 and rise>=.025)
        return passed, dict(sender_arm=sender, receiver_arm=receiver, continuous_dualhold_window_s=.1,
            required_points=count, actual_points=len(recent), no_missing_history_filled=True,
            both_opposed_distal_fingers_continuous=continuous, attachment_drifts_m=drifts,
            target_tcp_gaps_m=gaps, actual_centroid_rise_m=rise,
            forbidden_support=f.get('bimanual_forbidden_support', True),
            actual_contact_partner_shapes=f.get('bimanual_target_contact_partner_shapes', []),
            actual_grasp_contact_sides=f['grasp_contacts'])

    def validate(self, phase, event, outputs):
        if phase not in ROUTES:
            return super().validate(phase, event, outputs)
        if phase in event['results']:
            return event['results'][phase]['passed']
        canonical_phase = phase.removeprefix('return_')
        expected_hop = 2 if phase.startswith('return_') else 1
        detail = dict(phase=phase, piece=event['piece'], native_step=event['step'], policy_input=False, handoff_hop=expected_hop)
        passed = False
        try:
            state = outputs['state']
            if outputs.get('decision') != ROUTES[phase] or state['piece'] != event['piece']:
                raise ValueError('Wrong independently bound piece or phase route')
            sender, receiver = self._roles(state)
            if state.get('handoff', {}).get('hop') != expected_hop:
                raise ValueError('Physical handoff checkpoint requires its distinct typed hop')
            prefix_ok, pd = self._prefix_check(event)
            detail.update(pd)
            f = self.frames[event['step']]
            if canonical_phase == 'handoff_dualhold':
                functional, held = self._dual_hold(event, state)
                detail.update(held)
            else:
                selected = sender if canonical_phase == 'handoff_prepared' else receiver
                selected_state = dict(state, arm=selected)
                functional, held = self._held_check(event, selected_state)
                detail.update(held, sender_arm=sender, receiver_arm=receiver)
                count = int(round(.1*self.hz))+1
                recent = [self.frames[k] for k in range(event['step']-count+1, event['step']+1) if k in self.frames]
                no_foreign = len(recent)==count and all(not row.get('bimanual_forbidden_support', True) for row in recent)
                functional = functional and no_foreign
                detail['continuous_no_foreign_support'] = no_foreign
                if canonical_phase == 'handoff_prepared':
                    opened = float(f['hands'][receiver, 7])>.045
                    functional = functional and opened
                    detail['receiver_open_before_close'] = opened
                else:
                    gap = float(np.linalg.norm(f['clouds'][event['piece']]-f['hands'][sender, :3], axis=1).min())
                    withdrawn = float(f['hands'][sender, 7])>.045 and gap>.055
                    functional = functional and withdrawn and int(state['arm'])==receiver
                    detail.update(sender_open_and_withdrawn=withdrawn, sender_target_tcp_gap_m=gap,
                        selected_arm_matches_receiver=int(state['arm'])==receiver)
                    if canonical_phase == 'handoff_receiver_bound':
                        # This route promises receiver-only physical holding.
                        # UNKNOWN RGB or a correctly dated conditional binding
                        # cannot veto that actual contact/attachment promise.
                        # No measured pose or carry-ready binding is minted.
                        try:
                            localized, ld = self._localization_check(event, state)
                            detail.update(receiver_RGB_localization=ld,
                                receiver_RGB_localization_diagnostic_pass=bool(localized))
                        except Exception as error:
                            detail['receiver_RGB_localization'] = dict(diagnostic_error=f'{type(error).__name__}: {error}')
                        detail.update(visual_diagnostics_affect_physical_result=False,
                            receiver_control_binding_status=state.get('handoff', {}).get('receiver_control_binding_status', 'not_reported'),
                            receiver_physical_hold_does_not_certify_followup_carry_binding=True)
            passed = bool(functional and prefix_ok)
            detail.update(functional_postcondition_passed=bool(functional), prefix_postcondition_passed=bool(prefix_ok))
            if passed:
                self.begin = event['step']
        except Exception as error:
            detail['eval_error'] = f'{type(error).__name__}: {error}'
        detail['passed'] = passed
        event['results'][phase] = detail
        self.results.append(detail)
        with (self.output/'checkpoint_results.jsonl').open('a') as stream:
            stream.write(json.dumps(base._json_safe(detail), allow_nan=False)+'\n')
        return passed


def install(env, expected_order, initialized_prefix, output_dir):
    with base._LOCK:
        if id(env) in base._SESSIONS:
            raise RuntimeError('Independent verifier already installed for this Env')
        session = _Session(env, expected_order, initialized_prefix, output_dir)
        base._SESSIONS[id(env)] = session
    return session


capture = base.capture
validate = base.validate
verify = base.verify
diagnostics = base.diagnostics


def finalize(env):
    session = base._SESSIONS[id(env)]
    first_required = {'perceive_plan', 'grasp', *FIRST_ROUTES}
    return_required = set(RETURN_ROUTES)
    goal_required = {'transport', 'release', 'handoff'}
    passed = {row['phase'] for row in session.results if row.get('passed')}
    all_pass = all(row.get('passed') for row in session.results)
    first_endpoint = first_required.issubset(passed) and all(row.get('passed') for row in session.results if row['phase'] in first_required)
    two_handoffs = first_endpoint and return_required.issubset(passed) and all(row.get('passed') for row in session.results if row['phase'] in return_required)
    goal_delivery = two_handoffs and goal_required.issubset(passed) and all_pass
    summary = base.finalize(env)
    summary.update(first_handoff_receiver_endpoint_verified=first_endpoint,
        two_handoffs_verified=two_handoffs,
        handoff_receiver_endpoint_verified=two_handoffs,
        given_prefix_goal_delivery_verified=goal_delivery,
        handoff_required_phases=sorted(first_required | return_required),
        strict_goal_required_phases=sorted(goal_required),
        full_assembly_accepted=False, earned_full12_accepted=False,
        handoff_scope='Two physical Red handoffs and original strict nominal-hover, stable release and release-lineage checkpoints. Prefix11 is given, not earned.')
    (session.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary
