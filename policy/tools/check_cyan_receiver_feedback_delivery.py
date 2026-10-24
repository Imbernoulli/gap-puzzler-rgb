"""Actual candidate APIs with explicit offline fixtures; no Env or motors."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

ROOT = Path('/srv/home/bohanlyu/gap-puzzler-web')
HERE = Path(__file__).resolve().parents[1] / 'verification'
WORK = Path(__file__).resolve().parents[1]
PATCH = WORK / 'puzzler'
RUN = ROOT / 'runs/cyan_vertical_handoff_native_v2/workspace/runs/cyan_vertical_handoff_gpu6_v2'
sys.dont_write_bytecode = True
sys.path[:0] = [str(WORK), str(ROOT / 'third_party/graph-as-policy-runtime/gap-core/src')]
import puzzler
puzzler.__path__.insert(0, str(PATCH))
from puzzler import cyan_receiver_feedback_delivery as m

checks = {}
actual_state = json.loads((RUN / 'episode/trace/0000/node_data/handoff_receiver_hold_sg.act/output.json').read_text())['state']
observations = json.loads((RUN / 'episode/rgb/observations.json').read_text())


class Context:
    def __init__(self, now=63.92): self.now = now; self.calls = []
    def tool(self, name, **kwargs):
        self.calls.append((name, deepcopy(kwargs)))
        if name == 'robot.execution_journal_cursor': return dict(timestamp=self.now)
        return dict(status='ok')


def accepts(fn):
    try: fn(); return True
    except (m.c.PlanningRefusal, ValueError, KeyError): return False


epoch = actual_state['planning_scene']['measurement']['timestamp']
for motor in ('robot.execute_trajectory', 'robot.set_grip'):
    for age, expected in ((0., True), (30., True), (30.0001, False), (-.0001, False), (float('nan'), False)):
        raw = Context(epoch + age); wrapped = m._ReceiverSceneContext(raw, actual_state)
        #42.68+30 subtracts as30.000000000000007 in binary float. The
        # implementation conservatively refuses it; compare the actual
        # numeric age and separately test the exactly representable boundary.
        numeric_age = raw.now - epoch
        expected = bool(np.isfinite(numeric_age) and 0 <= numeric_age <= 30.)
        ok = accepts(lambda: wrapped.tool(motor, arm_id=1))
        checks[f'{motor}_age_{age}'] = ok == expected and (ok or all(name != motor for name, _ in raw.calls))
    boundary_state = deepcopy(actual_state)
    boundary_state['planning_scene']['measurement']['timestamp'] = 10.
    boundary_state['parameters']['historical_complete_rgb']['timestamp'] = 10.
    checks[motor + '_exactly_representable30s_accepted'] = accepts(lambda: m._ReceiverSceneContext(Context(40.), boundary_state).tool(motor))
for limit, age, expected in ((60., 30.1, False), (10., 10.1, False), (10., 10., True), (0., 0., False)):
    state = deepcopy(actual_state); state['parameters']['historical_max_age_s'] = limit
    checks[f'cannot_extend_original_TTL_{limit}_{age}'] = accepts(lambda: m._ReceiverSceneContext(Context(epoch + age), state).tool('robot.set_grip')) == expected
for key in ('timestamp', 'frame_sha256'):
    state = deepcopy(actual_state)
    state['planning_scene']['measurement'][key] = -1 if key == 'timestamp' else {'top': 'OTHER_EXPLICIT_FIXTURE'}
    checks['initial_scene_mismatch_' + key] = not accepts(lambda: m._ReceiverSceneContext(Context(), state))
state = deepcopy(actual_state); raw = Context(epoch + 30.1); wrapped = m._ReceiverSceneContext(raw, state)
state['parameters']['historical_complete_rgb']['timestamp'] = epoch + 30.
checks['caller_deepcopy_mutation_cannot_renew_bound_reference'] = not accepts(lambda: wrapped.tool('robot.execute_trajectory'))
raw = Context(epoch + 29.9); wrapped = m._ReceiverSceneContext(raw, actual_state)
wrapped.tool('robot.execute_trajectory')
raw.now = epoch + 30.1
checks['each_next_motor_reads_actual_journal_again'] = not accepts(lambda: wrapped.tool('robot.execute_trajectory')) and sum(name == 'robot.execute_trajectory' for name, _ in raw.calls) == 1

# The real release-scene builder sees original archived packets. Unrelated
# attachment/observation bookkeeping is mocked; no old datum is made current.
belief = dict(predicted_body_pose=actual_state['parameters']['owned_held_belief_source']['pose'],
              source=actual_state['parameters']['owned_held_belief_source'])
for timestamp, prior_timestamp, expected_reference, label in (
        (63.92, 42.68, 42.68, 'actual_partial63_92'),
        (27.62, .08, 27.62, 'actual_complete27_62')):
    observed = next(item for item in observations if item['timestamp'] == timestamp)
    prior = next(item for item in observations if item['timestamp'] == prior_timestamp)
    state = deepcopy(actual_state); state['parameters']['historical_complete_rgb'] = m.g.rgb_record(prior)
    state['planning_scene'] = dict(measurement=deepcopy(state['parameters']['historical_complete_rgb']))
    raw = Context(timestamp); ctx = m._ReceiverSceneContext(raw, state)
    with patch.object(m.tr, '_belief', return_value=belief), patch.object(m.c, 'state', return_value=deepcopy(observed)), \
         patch.object(m, 'check_bounded_release', return_value={}), patch.object(m, 'save_observation', side_effect=lambda state, observed: {**state, 'observation': observed}), \
         patch.object(m, '_selected_bead', return_value=2):
        updated, new_observed, scene, _, _ = m._refresh(ctx, state)
    checks[label + '_correct_actual_reference_epoch'] = ctx.reference['timestamp'] == expected_reference and scene['reference']['reference_timestamp'] == expected_reference
    checks[label + '_raw_packet_unchanged'] = new_observed == observed and scene['measurement']['poses'] == observed['poses']
    checks[label + '_actual_new_motor_uses_bound_reference'] = accepts(lambda: ctx.tool('robot.execute_trajectory'))
    if label.startswith('actual_partial'):
        raw.now = 79.32
        checks['actual79_32_motors_refused_on_partial63_92_and_old42_68_scene'] = not accepts(lambda: ctx.tool('robot.set_grip'))
        checks['actual_missing_board_lime_never_filled'] = 'board' not in new_observed['poses'] and 'lime' not in new_observed['poses']

# Run the actual transport state machine with ordered geometry/refresh test
# doubles. Verify receiver-only Cartesian view proposals, bounded retries,
# early exits, genuine environment renewal and donor-home ordering.
def transport_case(renew_after):
    state = deepcopy(actual_state); before = deepcopy(state); events = []; refresh_count = 0
    def refresh(ctx, current):
        nonlocal refresh_count
        refresh_count += 1
        current = deepcopy(current)
        current['receiver_delivery'] = dict(selected_receiver_bead=2)
        measured_time = 100. + refresh_count
        renewed = renew_after is not None and refresh_count > renew_after
        scene = dict(reference=dict(reference_timestamp=measured_time if renewed else 42.68))
        q = [0.] * 16; q[14:16] = [.012, -.012]
        belief = dict(current_robot_encoder_values=q, current_tcp=[.6, .2, 1., 0, 0, 0, 1])
        events.append(('refresh', refresh_count, renewed))
        return current, dict(timestamp=measured_time), scene, belief, {}
    def move(ctx, current, pos, rotation, scene, width, **kwargs):
        events.append(('move', np.asarray(pos).tolist(), kwargs)); return {}
    def home(*args): events.append(('home',)); return {}
    target = dict(relative_target_R=np.eye(3).tolist(), hover=[.5, 0, 1.], rotation=dict(x=0, y=0, z=0, w=1), authored_bound_estimate_public_hover_error_m=0.)
    with patch.object(m, '_refresh', side_effect=refresh), patch.object(m, 'move_held', side_effect=move), \
         patch.object(m, '_home_donor_while_held', side_effect=home), patch.object(m.c, 'hand', return_value=(np.array([.6, .2, 1.]), target['rotation'])), \
         patch.object(m, '_select_target', return_value=target), patch.object(m, '_hover', return_value=target), \
         patch.object(m, 'result', side_effect=lambda *args, **kwargs: dict(args=args, kwargs=kwargs)):
        ok = accepts(lambda: m._execute_transport(Context(), state))
    return ok, events, state == before

for renew_after in (0, 1, 2, 3, None):
    ok, events, immutable = transport_case(renew_after)
    homes = [i for i, row in enumerate(events) if row[0] == 'home']
    views = [row for row in events[:homes[0]] if row[0] == 'move'] if homes else [row for row in events if row[0] == 'move']
    expected = 3 if renew_after is None else renew_after
    checks[f'actual_transport_{renew_after}_bounded_view_count_and_home_order'] = len(views) == expected and bool(homes) == (renew_after is not None) and ok == (renew_after is not None)
    checks[f'actual_transport_{renew_after}_input_immutable'] = immutable
    if homes: checks[f'actual_transport_{renew_after}_home_requires_preceding_actual_new_reference'] = any(row[0] == 'refresh' and row[2] for row in events[:homes[0]])
    if views: checks[f'actual_transport_{renew_after}_same_orientation_Cartesian_original_guard_calls'] = all(row[2]['joint'] is False and 'belief' in row[2] for row in views)
    if len(views) == 3: checks[f'actual_transport_{renew_after}_only_three_relative_proposals'] = all(np.allclose(row[1], want) for row, want in zip(views, ([.6, .17, 1.02], [.6, .17, 1.], [.6, .17, 1.])))

old_tree = ast.parse((WORK / 'puzzler/cyan_receiver_delivery.py').read_text())
new_tree = ast.parse((PATCH / 'cyan_receiver_feedback_delivery.py').read_text())
old_functions = {node.name: node for node in old_tree.body if isinstance(node, ast.FunctionDef)}
new_functions = {node.name: node for node in new_tree.body if isinstance(node, ast.FunctionDef)}
for name in ('_selected_bead', '_board_goal', '_hover', '_select_target', '_stream', '_home_donor_while_held', '_material_empty', 'segment_conditional_release'):
    old = deepcopy(old_functions[name]); new = deepcopy(new_functions['_' + name if name == 'segment_conditional_release' else name])
    old.name = new.name
    checks[name + '_entire_executable_AST_unchanged'] = ast.dump(old) == ast.dump(new)
for name in ('execute_transport', 'segment_conditional_release'):
    checks[name + '_public_entry_wraps_scene_context'] = '_ReceiverSceneContext(ctx, state)' in ast.unparse(new_functions[name])
source_path = PATCH / 'cyan_receiver_feedback_delivery.py'
report = dict(schema='puzzler.cyan_receiver_feedback_delivery.packaged_checks.v1', all_passed=all(checks.values()), checks=checks,
    source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    scope='Actual candidate import and motor API fixtures, archived partial/complete RGB through real scene-reference builder, actual transport retry/order tests with geometry mocked, release/sweep/terminal helper AST identity. No Env, native execution, GPU geometry or physical success claim.',
    native_success_claim=False, visual_renewal_from_proposed_moves_not_yet_demonstrated=True)
(HERE / 'cyan_receiver_feedback_delivery_packaged_checks.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
