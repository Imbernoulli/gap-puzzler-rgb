"""Offline staged-rollout evidence audit. Never import this tool into a policy.

Legacy five-step samples support bounded, stationary-bracket comparisons only.
They cannot retrospectively supply exact checkpoint-time physics or contacts.
New checkpoint tapes, when present, remain separate evaluator-only evidence.
"""
import argparse
import ast
import bisect
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation, Slerp

from puzzler import model
from puzzler.audit import audit as released_audit
from tools.audit_visual_measurements import beads


BOARD_CORNERS = np.array([[.0275 + sx*.110, sy*.1575, .013]
                          for sx in (-1, 1) for sy in (-1, 1)])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_frame(point, gripper_readout):
    """Interpret only native manifest indices and native gripper readout rules."""
    manifest, arrays = point.manifest, point.arrays
    body_q, body_qd, joint_q = [np.asarray(arrays[name]) for name in ('body_q', 'body_qd', 'joint_q')]
    poses, velocities = {}, {}
    for name in ['board', *model.CELLS]:
        entity = manifest['entities'][name]
        body = (entity.get('bodies') or [entity['body']])[0]
        poses[name], velocities[name] = body_q[body].tolist(), body_qd[body].tolist()
    hands = []
    for name in ('left', 'right'):
        arm = manifest['arms'][name];readout = gripper_readout[name]
        body = body_q[arm['ee_body']]
        rotation = Rotation.from_quat(body[3:])
        position = body[:3] + rotation.apply(arm['ee_offset'])
        quaternion = (rotation * Rotation.from_quat(arm['ee_offset_rot_xyzw'])).as_quat()
        # Mirror the native Gripper.width() implementation, including YAM's
        # separately measured mirrored pair. Never guess this omitted metadata.
        coords, pair = readout['coords'], readout['width_coords']
        if pair:
            width = float(joint_q[pair[0]] - joint_q[pair[1]])
        elif readout['mirrored']:
            width = float(joint_q[coords[0]] - joint_q[coords[1]])
        else:
            lo, hi = float(readout['open_q'][0]), float(readout['closed_q'][0])
            width = float(readout['span']) if abs(hi - lo) < 1.e-9 else float(readout['span']) * (1. - float(np.clip((joint_q[coords[0]] - lo)/(hi - lo), 0., 1.)))
        hands.append([*position.tolist(), *quaternion.tolist(), width])
    return dict(step=point.step, time_s=point.sim_time, wall_time=point.wall_time,
                poses=poses, hands=hands, body_velocity=velocities)


def load_native_capture(root, result):
    """Use the installed RSS EpisodeCapture reader, with verified job provenance."""
    path = root/'development/native_capture_pointer.json'
    if not path.exists():
        return None
    pointer = json.loads(path.read_text())
    if pointer.get('policy_input') is not False:
        raise ValueError('Native capture pointer is not evaluator-only')
    for name, item in pointer['files'].items():
        if digest(item['path']) != item['sha256']:
            raise ValueError('Native capture artifact hash mismatch: ' + name)
    from robosimstudio.record import EpisodeCapture
    capture = EpisodeCapture(pointer['native_episode_dir'])
    if int(capture.manifest['task']['seed']) != int(result['seed']):
        raise ValueError('Native capture belongs to another seed')
    if abs(float(capture.manifest['task']['control_hz']) - result['control_hz']) > 1.e-8:
        raise ValueError('Native capture/control rate mismatch')
    readout = pointer.get('gripper_readout') or result.get('evaluation_gripper_readout')
    if not readout:
        raise ValueError('Native manifest omits calibrated gripper width metadata; no inferred readout permitted')
    frames = [native_frame(point, readout) for point in capture.points()]
    if any(b['step'] - a['step'] != 1 for a, b in zip(frames, frames[1:])):
        raise ValueError('Native every=1 capture has missing canonical control steps')
    return dict(capture=capture, frames=frames, pointer=pointer, gripper_readout=readout,
                provenance=dict(reader='robosimstudio.record.EpisodeCapture',
                                native_episode_dir=pointer['native_episode_dir'],
                                points=len(frames), exact_control_steps=True,
                                initial_step=frames[0]['step'] if frames else None,
                                native_contacts_available=bool((Path(pointer['native_episode_dir'])/'contacts.jsonl').exists()),
                                contacts_interpreted=False))


def points(name, pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError('Missing/nonfinite rigid pose for ' + name)
    if name == 'board':
        return BOARD_CORNERS @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3]
    return beads(name, pose)


def matched_error(a, b):
    distances = np.linalg.norm(a[:, None] - b[None, :], axis=2)
    ii, jj = linear_sum_assignment(distances)
    return float(distances[ii, jj].max())


class Samples:
    """Read recorded samples without claiming continuous motion between them."""
    def __init__(self, frames, hz, *, max_gap_s=.11, max_motion_m=.001):
        if not np.isfinite(hz) or hz <= 0:
            raise ValueError('control_hz must be finite and positive')
        self.frames, self.hz = frames, float(hz)
        self.steps = [f['step'] for f in frames]
        if not self.steps or any(b <= a for a, b in zip(self.steps, self.steps[1:])):
            raise ValueError('Recorded frame steps must be nonempty and strictly increasing')
        self.max_gap_s, self.max_motion_m = max_gap_s, max_motion_m

    def align(self, timestamp, names, arms=(), *, allow_exact=True):
        report = dict(scorable=False, timestamp=timestamp, exact_sample=False)
        if not isinstance(timestamp, (float, int)) or not np.isfinite(timestamp):
            return None, {**report, 'reason': 'missing/nonfinite measurement timestamp'}
        step = timestamp*self.hz
        hi = bisect.bisect_left(self.steps, step - 1.e-8)
        if allow_exact and hi < len(self.steps) and abs(self.steps[hi] - step) < 1.e-7:
            lo = hi
        elif hi == 0 or hi >= len(self.steps):
            return None, {**report, 'reason': 'measurement outside enclosing recorded frames'}
        else:
            lo = hi - 1
        a, b = self.frames[lo], self.frames[hi]
        selected = lo if abs(self.steps[lo] - step) <= abs(self.steps[hi] - step) else hi
        report.update(frame_bracket_steps=[self.steps[lo], self.steps[hi]],
                      bracket_duration_s=(self.steps[hi] - self.steps[lo])/self.hz,
                      selected_nearest_step=self.steps[selected], exact_sample=lo == hi,
                      nearest_offset_s=abs(self.steps[selected] - step)/self.hz)
        if report['bracket_duration_s'] > self.max_gap_s + 1.e-8:
            return None, {**report, 'reason': 'recorded frame gap exceeds declared alignment limit'}
        try:
            object_motion = {n: float(np.linalg.norm(points(n, a['poses'][n]) -
                                                   points(n, b['poses'][n]), axis=1).max())
                             for n in names}
            hand_motion = {}
            for arm in arms:
                aa, bb = np.asarray(a['hands'][arm]), np.asarray(b['hands'][arm])
                if aa.shape != (8,) or bb.shape != (8,) or not np.isfinite([aa, bb]).all():
                    raise ValueError('Invalid recorded hand')
                hand_motion[str(arm)] = float(np.linalg.norm(aa[:3] - bb[:3]))
        except (KeyError, IndexError, TypeError, ValueError) as error:
            return None, {**report, 'reason': 'missing/invalid sample geometry: ' + str(error)}
        report.update(object_bracket_motion_m=object_motion, hand_bracket_motion_m=hand_motion)
        if max([*object_motion.values(), *hand_motion.values()], default=0.) > self.max_motion_m:
            return None, {**report, 'reason': 'moving bracket; no checkpoint-time interpolation claim'}
        report.update(scorable=True,
                      reason='exact recorded sample' if lo == hi else
                      'nearest sample in a bounded nearly stationary recorded bracket')
        return self.frames[selected], report


def verify_rgb(root, observations):
    rows = []
    for index, observation in enumerate(observations):
        path = root/'rgb'/f'{index:04d}.npz'
        row = dict(observation=index, path=str(path.relative_to(root)), verified=False)
        try:
            with np.load(path, allow_pickle=False) as images:
                actual = {name: hashlib.sha256(images[name].tobytes()).hexdigest()
                          for name in images.files}
            expected = observation.get('frame_sha256')
            row.update(frame_sha256=actual, verified=bool(expected and actual == expected))
            if not row['verified']:
                row['reason'] = 'RGB archive does not match recorded sensor hashes'
        except (OSError, ValueError, KeyError) as error:
            row['reason'] = 'RGB archive unavailable/invalid: ' + str(error)
        rows.append(row)
    return rows


def observation_match(observations, rgb_rows, timestamp, hashes, poses=None):
    matching = [i for i, row in enumerate(observations)
                if row.get('timestamp') == timestamp and hashes and row.get('frame_sha256') == hashes]
    if len(matching) != 1:
        return None, dict(verified=False, reason='sensor timestamp/hash has no unique archived record')
    index = matching[0]
    if not rgb_rows[index]['verified']:
        return None, dict(verified=False, observation=index, reason='raw RGB is missing or changed')
    if poses is not None and poses != observations[index].get('poses'):
        return None, dict(verified=False, observation=index, reason='stage poses differ from sensor record')
    return observations[index], dict(verified=True, observation=index)


def visual_accuracy(samples, observations, rgb_rows):
    """Reuse archived bead geometry and the existing <=1mm stationary-bracket rule.

    As in audit_visual_measurements, linearly interpolate position and SLERP
    orientation only inside accepted stationary brackets. Board motion uses
    public case corners as an additional yaw-sensitive rejection screen.
    """
    measurements, unscored = [], []
    for index, obs in enumerate(observations):
        for name, estimate in obs.get('poses', {}).items():
            label = dict(observation=index, timestamp=obs.get('timestamp'), piece=name)
            if not rgb_rows[index]['verified']:
                unscored.append({**label, 'reason': 'RGB archive unavailable or changed'})
                continue
            _, alignment = samples.align(obs.get('timestamp'), [name], allow_exact=False)
            if not alignment['scorable']:
                unscored.append({**label, **alignment})
                continue
            lo, hi = [samples.steps.index(step) for step in alignment['frame_bracket_steps']]
            fraction = (obs['timestamp']*samples.hz - samples.steps[lo])/(samples.steps[hi] - samples.steps[lo])
            if not -1.e-7 <= fraction <= 1. + 1.e-7:
                unscored.append({**label, 'reason': 'timestamp lies outside selected physical bracket'})
                continue
            fraction = float(np.clip(fraction, 0., 1.))  # Floating timestamp conversion only.
            aa, bb = [np.asarray(samples.frames[i]['poses'][name]) for i in (lo, hi)]
            position = (1-fraction)*aa[:3] + fraction*bb[:3]
            rotation = Slerp([0, 1], Rotation.from_quat([aa[3:], bb[3:]]))([fraction]).as_quat()[0]
            truth = np.r_[position, rotation]
            error = (float(np.linalg.norm(np.asarray(estimate)[:3] - position)) if name == 'board'
                     else matched_error(beads(name, estimate), beads(name, truth)))
            measurements.append({**label, 'error_m': error,
                                 'bracket_motion_m': alignment['object_bracket_motion_m'][name],
                                 'frame_bracket_steps': alignment['frame_bracket_steps']})
    measurements.sort(key=lambda row: -row['error_m'])
    return dict(scope='Offline archived-RGB estimate scoring; not estimator rerun or policy feedback.',
                method='tools.audit_visual_measurements stationary brackets and Hungarian bead matching; board yaw additionally screened.',
                scored=len(measurements), unscored_count=len(unscored),
                accepted_over_3mm=sum(row['error_m'] > .003 for row in measurements),
                max_error_m=max((row['error_m'] for row in measurements), default=None),
                measurements=measurements, unscored=unscored)


def physical_tail(result, frames, names, through_step=None):
    selected = [f for f in frames if through_step is None or f['step'] <= through_step + 1.e-8]
    try:
        scoped = {**result, 'target_beads_local': {n: result['target_beads_local'][n] for n in names}}
        return {'available': True, **released_audit(scoped, selected)}
    except (KeyError, ValueError, IndexError, TypeError) as error:
        return dict(available=False, success=False, reason='Independent recorded-tail audit unavailable: ' + str(error))


def native_spans(root):
    path = root/'development/rss_tool_spans.jsonl'
    if not path.exists():
        return [], dict(available=False, reason='No native tool span archive')
    spans = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return spans, dict(available=True, rows=len(spans),
                       statuses=dict(Counter(row.get('status') for row in spans)),
                       contacts_recorded=False, exact_stage_boundaries=False)


def stage_paths(root):
    paths = []
    for node in sorted((root/'trace').glob('*/node_data/*')):
        iterations = sorted(node.glob('iters/*/output.json'))
        paths.extend(iterations or ([node/'output.json'] if (node/'output.json').exists() else []))
    return paths


def audit_stage(root, path, samples, result, observations, rgb_rows, spans):
    output = json.loads(path.read_text())
    state, evidence = output.get('state', {}), output.get('evidence', {})
    name, kind, arm = state.get('piece'), evidence.get('kind'), state.get('arm')
    row = dict(path=str(path.relative_to(root)), node=path.parents[2].name if path.parent.name.isdigit() else path.parent.name,
               stage=kind, route=output.get('route'), piece=name, attempt=state.get('attempt'),
               evidence_timestamp=evidence.get('timestamp'), physical_contact_verified=False)
    sensor, match = observation_match(observations, rgb_rows, evidence.get('timestamp'),
                                      evidence.get('frame_sha256'), state.get('observation', {}).get('poses'))
    row['sensor_evidence'] = match
    stage_spans = [s for s in spans if s.get('node') == row['node']]
    row['native_tools'] = dict(count=len(stage_spans), statuses=dict(Counter(s.get('status') for s in stage_spans)),
                               interval_steps=[min(s['start_step'] for s in stage_spans), max(s['end_step'] for s in stage_spans)] if stage_spans else None,
                               scope='sim_step tool span interval only; not exact stage entry/exit')
    if not sensor or name not in model.CELLS:
        row['physical_comparison'] = dict(available=False, reason='Missing verified stage sensor record or known piece')
        return row
    active_arms = [arm] if arm in (0, 1) else [0, 1]
    frame, alignment = samples.align(evidence.get('timestamp'), [name, 'board'], active_arms)
    row['alignment'] = alignment
    if frame is None:
        row['physical_comparison'] = dict(available=False, reason=alignment['reason'])
        return row
    cloud = points(name, frame['poses'][name])
    gaps = [float(np.linalg.norm(cloud - np.asarray(hand[:3]), axis=1).min()) for hand in frame['hands']]
    fact = dict(available=True, nearest_bead_tcp_m=gaps, hand_width_m=[h[7] for h in frame['hands']],
                sample_step=frame['step'], exact_checkpoint_physics=False,
                scope='Recorded geometry consistent with a stationary RGB-time boundary; no contact/attachment proof.')
    if kind in ('grasp', 'transport') or row['route'] == 'lost':
        source = state.get('plan', {}).get('s', {})
        source_obs, source_match = observation_match(observations, rgb_rows, source.get('timestamp'), source.get('frame_sha256'), source.get('poses'))
        fact['source_sensor_evidence'] = source_match
        baseline, base_alignment = samples.align(source.get('timestamp'), [name]) if source_obs else (None, dict(scorable=False, reason='No verified source RGB record'))
        fact['source_alignment'] = base_alignment
        if arm not in (0, 1) or baseline is None:
            fact.update(held_geometry_consistent=None, reason='Source truth or active arm unavailable')
        else:
            rise = float(frame['poses'][name][2] - baseline['poses'][name][2])
            fact.update(rise_from_source_m=rise, claimed_rise_m=evidence.get('rise_m'),
                        claimed_hand_gap_m=evidence.get('hand_bead_gap_m'),
                        held_geometry_consistent=bool(rise >= .025 and gaps[arm] <= .045),
                        separated_loss_geometry_consistent=bool(gaps[arm] > .045))
    elif kind in ('release', 'placement', 'finish'):
        names = (state.get('parameters', {}).get('diagnostic_subset') or list(model.CELLS)) if kind == 'finish' else [name]
        fact['recorded_released_tail_through_stage'] = physical_tail(result, samples.frames, names, evidence['timestamp']*samples.hz)
        if 'prior_timestamp' in evidence:
            prior, prior_match = observation_match(observations, rgb_rows, evidence.get('prior_timestamp'), evidence.get('prior_frame_sha256'))
            before, prior_alignment = samples.align(evidence.get('prior_timestamp'), [*names, 'board'], [0, 1]) if prior else (None, dict(scorable=False))
            fact.update(prior_sensor_evidence=prior_match, prior_alignment=prior_alignment)
            if before is not None:
                displacement = {n: matched_error(points(n, before['poses'][n]), points(n, frame['poses'][n])) for n in names}
                displacement['board_corners'] = float(np.linalg.norm(points('board', before['poses']['board']) - points('board', frame['poses']['board']), axis=1).max())
                fact.update(recorded_endpoint_displacement_m=displacement,
                            repeated_measurement_separation_s=evidence['timestamp'] - evidence['prior_timestamp'],
                            claimed_max_displacement_m=evidence.get('max_displacement_m'))
            else:
                fact['repeated_physical_comparison_unavailable'] = 'Prior image boundary lacks stationary physical alignment'
    row['physical_comparison'] = fact
    return row


def scene_motion(frames, manipulated):
    baseline, last = frames[0], frames[-1]
    rows = {}
    for name in baseline['poses']:
        if name not in model.CELLS and name != 'board':
            continue
        initial = points(name, baseline['poses'][name])
        differences = [(f['step'], float(np.linalg.norm(points(name, f['poses'][name]) - initial, axis=1).max()))
                       for f in frames if name in f['poses']]
        step, maximum = max(differences, key=lambda pair: pair[1])
        rows[name] = dict(manipulated=name in manipulated, maximum_sampled_displacement_from_first_m=maximum,
                          maximum_at_step=step, final_displacement_from_first_m=differences[-1][1])
    return dict(first_recorded_step=baseline['step'], last_recorded_step=last['step'], bodies=rows,
                scope='Recorded bead correspondence and board-corner motion at saved samples; unrecorded transients/contact forces unknown.',
                unsupported=['Table/hinge contact and force changes', 'Motion between saved frames'])


def declared_checkpoints(path):
    """Read literal checkpoint identities without executing a frozen sidecar."""
    tree = ast.parse(path.read_text(), filename=str(path))
    assignments = [node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'CHECKPOINTS' for target in node.targets)]
    if len(assignments) != 1 or not isinstance(assignments[0], (ast.List, ast.Tuple)):
        raise ValueError('Checkpoint declaration is not a single literal list')
    names = []
    for node in assignments[0].elts:
        if not isinstance(node, ast.Call):
            raise ValueError('Checkpoint identity is not statically readable')
        values = {entry.arg: ast.literal_eval(entry.value) for entry in node.keywords
                  if entry.arg in ('name', 'subgraph', 'validate')}
        if set(values) != {'name', 'subgraph', 'validate'}:
            raise ValueError('Checkpoint identity/validation mode is not explicit')
        if values['validate']:
            names.append((values['subgraph'], values['name']))
    if not names:
        raise ValueError('No explicit validate checkpoint to pair with physics')
    return names


def exact_stage_predicate(stage, physics, baseline, result):
    """Physical postconditions use truth and public goals, never RGB flags."""
    name, kind, route = stage['piece'], stage['stage'], stage['route']
    output = stage['_output'];state = output['state'];arm = state.get('arm')
    poses, hands = physics['poses'], physics['hands']
    cloud = points(name, poses[name])
    gaps = [float(np.linalg.norm(cloud - np.asarray(hand[:3]), axis=1).min()) for hand in hands]
    facts = dict(nearest_bead_tcp_m=gaps, hand_width_m=[h[7] for h in hands],
                 physics_step=physics['step'], physics_time_s=physics['time_s'],
                 contacts_verified=False)
    if kind == 'plan':
        facts.update(physical_postcondition_pass=True,
                     definition='Finite recorded scene poses; independently scored perception gate is separate')
    elif kind in ('grasp', 'transport') or route == 'lost':
        if arm not in (0, 1):
            return {**facts, 'physical_postcondition_pass': False, 'reason': 'No active arm'}
        if route == 'lost':
            facts['physical_postcondition_pass'] = bool(gaps[arm] > .045)
        elif baseline is None:
            facts.update(physical_postcondition_pass=False, reason='No prior exact plan physics for this piece/attempt')
        else:
            rise = float(poses[name][2] - baseline['poses'][name][2])
            facts.update(rise_from_exact_plan_m=rise,
                         physical_postcondition_pass=bool(rise >= .025 and gaps[arm] <= .045 and hands[arm][7] < .045))
    elif kind in ('release', 'placement', 'handoff', 'finish', 'recovery'):
        names = ((state.get('parameters', {}).get('diagnostic_subset') or list(model.CELLS))
                 if kind in ('finish', 'recovery') else [name])
        board = np.asarray(poses['board']);rotation = Rotation.from_quat(board[3:]).as_matrix()
        errors, speeds = {}, {}
        for piece in names:
            world = points(piece, poses[piece]);local = (world - board[:3]) @ rotation
            errors[piece] = matched_error(local, np.asarray(result['target_beads_local'][piece]))
            velocity = np.asarray(physics['body_velocity'][piece])
            bead_velocity = velocity[:3] + np.cross(velocity[3:], world - np.asarray(poses[piece][:3]))
            speeds[piece] = float(np.linalg.norm(bead_velocity, axis=1).max())
        combined = np.concatenate([points(piece, poses[piece]) for piece in names])
        separated = all(float(np.linalg.norm(combined - np.asarray(hand[:3]), axis=1).min()) > .055
                        and hand[7] > .045 for hand in hands)
        board_ok = abs(board[2] - model.ORIGIN[2] - .003) < .006 and rotation[2, 2] > .995
        settled = max(speeds.values()) < .015
        goal_ok = max(errors.values()) < .004
        if kind == 'recovery':
            passed = separated and board_ok and settled
        elif route == 'missed':
            passed = separated and board_ok and settled and max(errors.values()) >= .003
        else:
            passed = separated and board_ok and settled and goal_ok
        facts.update(physical_goal_error_m=errors, maximum_bead_speed_mps=speeds,
                     released_separated=separated, board_resting_and_level_geometry=bool(board_ok),
                     physical_postcondition_pass=bool(passed))
    else:
        facts.update(physical_postcondition_pass=False, reason='Unsupported stage kind')
    return facts


def exact_checkpoint_audit(root, result, stages, observations, rgb_rows, native=None):
    """Strictly pair one native snapshot invocation to its checkpoint results.

    The pinned public execute API has no stage hook. This limited mapping is
    accepted only with a literal frozen sidecar, exact result group identities,
    one distinct stage output and identical sensor provenance. Any ambiguity
    fails coverage rather than borrowing another stage's physical record.
    """
    failed = dict(available=False, pairing_valid=False, complete=False,
                  all_required_physical_predicates_pass=False,
                  all_required_perception_gates_pass=False, stages=[])
    path = root/'development/checkpoint_physics.jsonl'
    meta_path = root/'development/checkpoint_physics.meta.json'
    native_indices = root/'development/native_checkpoint_indices.jsonl'
    native_meta = root/'development/native_checkpoint_indices.meta.json'
    use_native = native is not None
    if use_native:
        path, meta_path = native_indices, native_meta
    if not path.exists() or not meta_path.exists():
        return {**failed, 'reason': 'No evaluator-only exact checkpoint physics tape; legacy samples are not substituted'}
    failed['available'] = True
    try:
        raw_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        metadata = json.loads(meta_path.read_text())
        if (not metadata.get('rollout_returned') or metadata.get('capture_errors') or
                metadata.get('persistence_errors') or metadata.get('rows') != len(raw_rows)):
            raise ValueError('Recorder incomplete or reports capture/persistence errors')
        if [row.get('sequence') for row in raw_rows] != list(range(len(raw_rows))):
            raise ValueError('Snapshot sequence missing, duplicated or reordered')
        if use_native:
            initial = native_frame(native['capture'].point(step=0), native['gripper_readout'])
            tape = [dict(sequence=0,kind='initial',status='ok',policy_input=False,physics=initial)]
            for index in raw_rows:
                if index.get('status') != 'ok' or index.get('policy_input') is not False:
                    raise ValueError('Native checkpoint index capture failed')
                physics = native_frame(native['capture'].point(step=index['boundary_step']), native['gripper_readout'])
                tape.append({**index,'sequence':index['sequence']+1,'kind':'checkpoint_snapshot','physics':physics})
        else:
            tape = raw_rows
        if (not tape or tape[0].get('kind') != 'initial' or
                any(row.get('kind') != 'checkpoint_snapshot' for row in tape[1:]) or
                any(row.get('status') != 'ok' or row.get('policy_input') is not False for row in tape)):
            raise ValueError('Missing initial state or invalid callback capture')
        workflows = [p for p in (root/'source_snapshot').rglob('workflow.json')
                     if digest(p) == result.get('workflow_sha256')]
        if len(workflows) != 1:
            raise ValueError('No unique workflow matching the recorded graph hash')
        graph = workflows[0].parent
        checkpoints = result.get('checkpoint_results', [])
        cursor, rows, visits, plans = 0, [], Counter(), {}
        for captured in tape[1:]:
            if cursor >= len(checkpoints):
                raise ValueError('Physical snapshot has no native checkpoint result')
            subgraph = checkpoints[cursor]['subgraph']
            declared = declared_checkpoints(graph/'checkpoints'/f'{subgraph}.py')
            chunk = checkpoints[cursor:cursor + len(declared)]
            if [(item.get('subgraph'), item.get('name')) for item in chunk] != declared:
                raise ValueError('Frozen checkpoint/result grouping mismatch')
            provenance = captured.get('sensor_provenance', {})
            candidates = [stage for stage in stages if stage['node'] == subgraph + '.act'
                          and stage['_output'].get('evidence', {}).get('timestamp') == provenance.get('timestamp')
                          and stage['_output'].get('evidence', {}).get('frame_sha256') == provenance.get('frame_sha256')]
            if len(candidates) != 1:
                raise ValueError('No unique subgraph output matching captured sensor provenance')
            stage = candidates[0]
            sensor, match = observation_match(observations, rgb_rows, provenance.get('timestamp'),
                                              provenance.get('frame_sha256'), stage['_output']['state'].get('observation', {}).get('poses'))
            if not sensor:
                raise ValueError('Snapshot sensor provenance is not backed by archived RGB: ' + str(match))
            physics = captured['physics']
            if physics['time_s'] + 1.e-8 < provenance['timestamp']:
                raise ValueError('Physics capture precedes its purported sensor observation')
            for piece in ['board', *model.CELLS]:
                points(piece, physics['poses'][piece])
                velocity = np.asarray(physics['body_velocity'][piece])
                if velocity.shape != (6,) or not np.isfinite(velocity).all():
                    raise ValueError('Invalid captured velocity')
            if len(physics['hands']) != 2 or any(np.asarray(h).shape != (8,) or not np.isfinite(h).all() for h in physics['hands']):
                raise ValueError('Invalid captured robot hands')
            key = (stage['piece'], stage['attempt'])
            facts = exact_stage_predicate(stage, physics, plans.get(key), result)
            if stage['stage'] == 'plan':
                plans[key] = physics
            discrepancies = {piece: (float(np.linalg.norm(np.asarray(pose)[:3] - np.asarray(physics['poses'][piece])[:3]))
                                     if piece == 'board' else matched_error(points(piece, pose), points(piece, physics['poses'][piece])))
                             for piece, pose in sensor['poses'].items()}
            threshold = .002 if stage['stage'] == 'plan' and len(rows) == 0 else .003
            complete_scene = set(['board', *model.CELLS]).issubset(sensor['poses'])
            required_scene = stage['stage'] == 'plan' or stage['stage'] == 'release' and stage['route'] != 'lost'
            perception_ok = bool(discrepancies and max(discrepancies.values()) <= threshold
                                 and (complete_scene or not required_scene))
            rows.append(dict(snapshot_sequence=captured['sequence'], subgraph=subgraph, visit_index=visits[subgraph],
                             node=stage['node'], piece=stage['piece'], attempt=stage['attempt'], route=stage['route'],
                             sensor_timestamp=provenance['timestamp'], capture_step=physics['step'],
                             capture_time_s=physics['time_s'], sensor_to_boundary_delay_s=physics['time_s'] - provenance['timestamp'],
                             exact_sensor_time_alignment=False, native_checkpoint_results=chunk,
                             facts=facts, accepted_pose_discrepancy_at_boundary_m=discrepancies,
                             perception_threshold_m=threshold, perception_gate_pass=perception_ok,
                             perception_comparison_scope='RGB estimates vs checkpoint-time physics; includes sensor-to-boundary lag, not exact image-time error'))
            visits[subgraph] += 1;cursor += len(declared)
        if cursor != len(checkpoints) or not rows:
            raise ValueError('Native checkpoint results lack a one-to-one callback snapshot group')
        return dict(available=True, pairing_valid=True, complete=True,
                    all_required_physical_predicates_pass=all(row['facts']['physical_postcondition_pass'] for row in rows),
                    all_required_perception_gates_pass=all(row['perception_gate_pass'] for row in rows),
                    initial_capture=tape[0]['physics'], stages=rows,
                    scope=('Native EpisodeCapture physics at callback-referenced canonical control steps; no exact continuous wallinstant, stage-entry or errored-exit coverage'
                           if use_native else 'Exact owner-thread callback physics at realized native checkpoint exits; no stage-entry or errored-exit coverage'),
                    contacts_verified=False)
    except (KeyError, ValueError, TypeError, OSError, IndexError, SyntaxError) as error:
        return {**failed, 'reason': 'Exact checkpoint pairing/audit refused: ' + str(error)}


def audit_episode(directory):
    root = Path(directory).resolve()
    result = json.loads((root/'result.json').read_text())
    native = load_native_capture(root, result)
    frames = native['frames'] if native is not None else json.loads((root/'frames.json').read_text())
    observation_path = root/'rgb/observations.json'
    observations = json.loads(observation_path.read_text()) if observation_path.exists() else []
    expected_model = result.get('source_sha256', {}).get('puzzler/model.py')
    if expected_model and digest(Path(model.__file__)) != expected_model:
        raise ValueError('Current public CAD differs from recorded model.py; refusing geometry audit')
    samples = Samples(frames, result['control_hz'])
    rgb_rows = verify_rgb(root, observations)
    spans, span_report = native_spans(root)
    if native is not None:
        spans = native['capture'].spans
        span_report = dict(available=True,rows=len(spans),statuses=dict(Counter(row.get('status') for row in spans)),
                           source='native EpisodeCapture.spans',contacts_interpreted=False)
    stages = [dict(audit_stage(root, path, samples, result, observations, rgb_rows, spans),
                   _output=json.loads(path.read_text())) for path in stage_paths(root)]
    stages.sort(key=lambda row: (row.get('evidence_timestamp') or 0., row['path']))
    exact_audit = exact_checkpoint_audit(root, result, stages, observations, rgb_rows, native)
    for stage in stages:
        stage.pop('_output')
    names = result.get('diagnostic_subset') or list(model.CELLS)
    tools_path = root/'rgb/tools.jsonl'
    reserved_calls = []
    if tools_path.exists():
        reserved_calls = [row for row in map(json.loads, tools_path.read_text().splitlines())
                          if row.get('event') == 'call' and row.get('tool') == 'puzzle.plan_path'
                          and row.get('arguments', {}).get('min_joint_limit_margin_rad', 0.) > 0.]
    input_paths = [root/'result.json', *stage_paths(root)]
    if observation_path.exists():
        input_paths.append(observation_path)
    if (root/'frames.json').exists():
        input_paths.append(root/'frames.json')
    input_paths += [p for p in (root/'development').glob('rss_tool_spans*') if p.is_file()]
    input_paths += [p for p in (root/'development').glob('*checkpoint*') if p.is_file()]
    if native is not None:
        input_paths.append(root/'development/native_capture_pointer.json')
    report = dict(schema='puzzler.offline_staged_audit.v1', policy_input=False, run=str(root),
                  scope='Independent offline recorded-geometry and archived-RGB audit; no new rollout or full-task acceptance.',
                  alignment_limits=dict(max_enclosing_gap_s=samples.max_gap_s, max_stationary_bracket_motion_m=samples.max_motion_m),
                  declared_result={key: result.get(key) for key in ('success', 'rss_success', 'graph_success', 'source_integrity_verified', 'diagnostic_subset', 'single_piece_physical_diagnostic')},
                  native_checkpoint_results=result.get('checkpoint_results', []), native_tool_spans=span_report,
                  native_capture=native['provenance'] if native is not None else dict(available=False),
                  rgb_recording_available=observation_path.exists(),
                  rgb_archives=rgb_rows, stages=stages,
                  independent_released_subset=physical_tail(result, frames, names),
                  independent_released_full_task=physical_tail(result, frames, list(model.CELLS)),
                  visual_measurements=visual_accuracy(samples, observations, rgb_rows),
                  scene_motion=scene_motion(frames, names),
                  active_scene_inspection=dict(explicit_reserved_path_proposals=len(reserved_calls),
                                               physical_helper_motion_established=False,
                                               reason='No reserved helper path proposal recorded' if not reserved_calls else 'A plan proposal alone does not establish helper execution'),
                  exact_checkpoint_audit=exact_audit,
                  input_sha256={str(path.relative_to(root)): digest(path) for path in input_paths},
                  auditor_source_sha256={str(Path(__file__).resolve()): digest(__file__),
                                        str(Path(model.__file__).resolve()): digest(model.__file__)},
                  limitations=['Geometric heldness is not contact or force-closure verification.',
                               'Legacy five-step samples do not supply exact checkpoint-time physics; native indexed control steps require verified native capture.',
                               'Stationary bracket motion is an empirical screen, not a continuous trajectory bound.',
                               'Sensor estimate scoring reuses archived estimator outputs and does not rerun perception.'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode')
    parser.add_argument('--out', required=True, help='New report path outside the preserved episode')
    args = parser.parse_args()
    root, destination = Path(args.episode).resolve(), Path(args.out).resolve()
    if destination.is_relative_to(root):
        parser.error('Write the audit outside the preserved episode directory')
    report = audit_episode(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(report=str(destination), stages=len(report['stages']),
                          subset=report['independent_released_subset'],
                          visual_scored=report['visual_measurements']['scored'],
                          visual_max_error_m=report['visual_measurements']['max_error_m']), indent=2))


if __name__ == '__main__':
    main()
