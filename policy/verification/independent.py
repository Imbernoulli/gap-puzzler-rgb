"""Independent REHEARSAL-ONLY strict-GaP physical postconditions, v1.

GT stays in this evaluator cache. No GT is added to policy World, graph state,
tool returns, planning inputs, or measured RGB. The owner-thread observer reads
completed native control states; it never steps or changes the environment.
"""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import json
import hashlib
import threading

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from puzzler import model, task
from puzzler.pyramid import get_plan, physical

_SESSIONS = {}
_EVENTS = OrderedDict()
_LOCK = threading.RLock()
_PHASE_ROUTES = dict(perceive_plan=('planned',), grasp=('verified',), transport=('carried',),
                     release=('placed', 'uncertain'), reobserve=('reobserved', 'placed'), handoff=('advanced',))


def _error(a, b):
    d = np.linalg.norm(a[:, None]-b[None, :], axis=2)
    i, j = linear_sum_assignment(d)
    return float(d[i, j].max())


def _cloud(name, pose):
    pose = np.asarray(pose, dtype=float)
    return model.local_beads(name) @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3]


def _finite(value, shape):
    value = np.asarray(value, dtype=float)
    return value.shape == shape and bool(np.isfinite(value).all())


def _source_rgb_supported(observed, piece):
    """Validate the declared RGB interface without consulting policy claims.

    Preserve the original shape-source contract. The additional calibrated
    interface requires its existing measured multiview support and same-image
    complete q16 epoch. Raw camera packet completeness is separate from target
    visibility; the target need only have the original two supported views.
    """
    source = observed.get('observation_source')
    if source == 'rgb_shape_geometry':
        return True
    if source != 'calibrated_rgb_cad_seeded':
        return False
    poses = observed.get('poses', {})
    if not {'board', piece}.issubset(poses) or not _finite(poses['board'], (7,)):
        return False
    detection = observed.get('detections', {}).get(piece, {})
    center = np.asarray(detection.get('center_support', []), dtype=float)
    views = detection.get('views', [])
    if (detection.get('visible') is not True or len(views) < 2
            or center.size < 4 or int(np.sum(center >= .6)) < 4):
        return False
    frames = observed.get('frame_sha256', {})
    required = {'top', 'front', 'rear', 'side'}
    if (not isinstance(frames, dict) or not required.issubset(frames)
            or not all(isinstance(frames[v], str) and len(frames[v]) == 64
                and all(ch in '0123456789abcdef' for ch in frames[v]) for v in required)):
        return False
    provenance = observed.get('robot_geometry_provenance', {})
    q16 = np.asarray(provenance.get('robot_encoder_values', []), dtype=np.float32)
    timestamp = observed.get('timestamp')
    return bool(isinstance(timestamp, (int, float)) and np.isfinite(timestamp)
        and provenance.get('coordinate_indices') == list(range(16))
        and provenance.get('timestamp') == timestamp
        and q16.shape == (16,) and np.isfinite(q16).all()
        and hashlib.sha256(q16.tobytes()).hexdigest() == provenance.get('robot_encoder_sha256'))


def _json_safe(value):
    """Malformed diagnostics must remain printable without changing a verdict."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _rgb_interface_diagnostics(observed, piece):
    """Interface/pixel support cannot veto a physical geometry comparison."""
    try:
        return dict(source_interface_supported=_source_rgb_supported(observed, piece),
            observation_source=observed.get('observation_source'),
            frame_hashes_present=bool(observed.get('frame_sha256')))
    except Exception as error:
        return dict(diagnostic_error=f'{type(error).__name__}: {error}')


def _reobserve_diagnostics(state, evidence, piece):
    """Never mutate routes, receipt records, UNKNOWN state or earned ledger."""
    try:
        observation = state.get('observation', {})
        receipt = evidence.get('receipt')
        pending = state.get('pending_placement_receipt', {})
        return dict(RGB_interface=_rgb_interface_diagnostics(observation, piece),
            receipt_present=isinstance(receipt, dict),
            target_currently_measured=piece in observation.get('poses', {}),
            target_3d_status=pending.get('target_3d_status'),
            replan_required=pending.get('replan_required'),
            no_ledger_update=piece not in state.get('executed_placements', []),
            receipt_maximum_displacement_m=receipt.get('maximum_displacement_m') if isinstance(receipt, dict) else None,
            receipt_visual_goal_error_m=receipt.get('goal_error_m') if isinstance(receipt, dict) else None,
            note='1.5 mm RGB pair displacement is temporal visual stability, not absolute localization accuracy.')
    except Exception as error:
        return dict(diagnostic_error=f'{type(error).__name__}: {error}')


def _opening_axes(env, handle):
    """Declared prismatic kinematics, never fit contact forces to an axis."""
    m = env.model
    child, kinds = m.joint_child.numpy(), m.joint_type.numpy()
    starts, axes = m.joint_qd_start.numpy(), m.joint_axis.numpy()
    transforms = m.joint_X_c.numpy()
    result = []
    for side, group in enumerate(handle.finger_groups):
        entries = []
        for body, shapes in group:
            joints = np.flatnonzero(child == body)
            if len(joints) != 1 or int(kinds[joints[0]]) != 0:
                raise ValueError('Functional grasp adapter requires a directly driven prismatic finger')
            joint = int(joints[0])
            if int(starts[joint+1]-starts[joint]) != 1:
                raise ValueError('Finger does not have exactly one public prismatic DOF')
            local = Rotation.from_quat(transforms[joint, 3:]).apply(axes[int(starts[joint])])
            local = local / np.linalg.norm(local) * float(handle.open_axis_signs[side])
            entries.append((int(body), list(shapes), local))
        result.append(entries)
    return result


def _axis_grasp(contacts, groups, shapes, body_q):
    sides = []
    for group in groups:
        evidence = []
        for body, fingers, local in group:
            impulse = np.asarray(contacts.pairwise_impulse(fingers, shapes), dtype=float)
            norm = float(np.linalg.norm(impulse))
            opens = Rotation.from_quat(body_q[body, 3:]).apply(local)
            angle = float(np.degrees(np.arccos(np.clip(np.dot(opens, impulse)/(norm+1.e-30), -1., 1.))))
            evidence.append(dict(body=body, public_local_open_axis=local.tolist(), impulse_Ns=norm,
                                 outward_angle_degrees=angle, passed=norm>=1.e-6 and angle<=85.))
        sides.append(dict(passed=any(e['passed'] for e in evidence), members=evidence))
    return bool(sides) and all(s['passed'] for s in sides), sides


class _Session:
    def __init__(self, env, expected_order, initialized_prefix, output_dir):
        self.env = env
        self.order = list(expected_order)
        self.prefix = list(initialized_prefix)
        if len(set(self.order)) != len(self.order) or set(self.order) & set(self.prefix):
            raise ValueError('Independent verifier requires disjoint ordered targets and initialized prefix')
        if set(self.order+self.prefix)-set(model.CELLS):
            raise ValueError('Unknown public target name')
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.hz = float(env.pipeline.control_hz)
        self.index = 0
        self.begin = int(env.step_count)
        self.frames = OrderedDict()
        self.source_centroid_z = {}
        self.last_release = None
        self.results = []
        self.opening_axes = [_opening_axes(env, env.handles.arms[a]) for a in ('left', 'right')]
        self.targets = {n: physical(v['sites']) for n, v in get_plan('pyramid_robot6_plan.json')['placements'].items()}
        # This is the same public robot reference used by task.register_goal.
        # A late-installed continuation may currently be holding a piece far
        # from home; its current pose must never redefine the final goal.
        self.home = np.asarray([env.arm(a).home_pose[0] for a in ('left', 'right')], dtype=float)
        if not _finite(self.home, (2, 3)):
            raise ValueError('Missing finite public robot home pose')
        self.home_reference_source = 'public_env_arm_home_pose'
        self.observer = self._observe
        self._observe(env)
        env.add_step_observer(self.observer)

    def _observe(self, env, *unused):
        step = int(env.step_count)
        if step in self.frames:
            return
        poses = {n: np.asarray(p, dtype=float).copy() for n, p in task.poses(env).items()}
        board = poses['board']
        rb = Rotation.from_quat(board[3:]).as_matrix()
        clouds = {n: _cloud(n, poses[n]) for n in model.CELLS}
        errors = {n: _error((v-board[:3]) @ rb, self.targets[n]) for n, v in clouds.items()}
        speeds = {}
        for n, beads in clouds.items():
            velocity = np.asarray(env[n].velocity, dtype=float)
            speeds[n] = float(np.linalg.norm(velocity[:3]+np.cross(velocity[3:], beads-poses[n][:3]), axis=1).max())
        hands = []
        for arm in ('left', 'right'):
            position, quaternion = env.arm(arm).ee_pose(env.state)
            hands.append([*map(float, position), *map(float, quaternion), float(env.gripper(arm).width(env.state))])
        hands = np.asarray(hands)
        all_beads = np.concatenate(list(clouds.values()))
        gaps = [float(np.linalg.norm(all_beads-h[:3], axis=1).min()) for h in hands]
        contacts = env.contacts
        robot_loaded = {n: False for n in model.CELLS}
        table_loaded = {n: False for n in model.CELLS}
        held = [False, False]
        native_held = [False, False]
        grasp_contacts = [[], []]
        nonfinger_loaded = [False, False]
        if contacts is not None:
            contacts.update(step)
            a, b = contacts._shape_pairs()
            positive = np.linalg.norm(contacts.impulses, axis=1) >= 1.e-6
            robot = np.asarray(env.handles.robot_shapes)
            table = np.asarray(env.handles.entity('table').collision_shapes)
            for n in model.CELLS:
                shapes = np.asarray(env.handles.entity(n).collision_shapes)
                aa, bb = np.isin(a, shapes), np.isin(b, shapes)
                robot_loaded[n] = bool(np.any(positive & ((aa & np.isin(b, robot)) | (bb & np.isin(a, robot)))))
                table_loaded[n] = bool(np.any(positive & ((aa & np.isin(b, table)) | (bb & np.isin(a, table)))))
            if self.index < len(self.order):
                shapes = env.handles.entity(self.order[self.index]).collision_shapes
                body_q = np.asarray(env.state.body_q.numpy(), dtype=float)
                for i, name in enumerate(('left', 'right')):
                    handle = env.handles.arms[name]
                    native_held[i] = bool(contacts.check_grasp(handle.finger_groups, shapes, body_q,
                        open_axis_signs=handle.open_axis_signs, max_angle=85.))
                    held[i], grasp_contacts[i] = _axis_grasp(contacts, self.opening_axes[i], shapes, body_q)
                    selected_fingers = np.asarray([s for group in handle.finger_groups for _, ss in group for s in ss])
                    others = robot[~np.isin(robot, selected_fingers)]
                    aa, bb = np.isin(a, shapes), np.isin(b, shapes)
                    nonfinger_loaded[i] = bool(np.any(positive & ((aa & np.isin(b, others)) | (bb & np.isin(a, others)))))
        board_velocity = np.asarray(env['board'].velocity, dtype=float)
        frame = dict(step=step, poses=poses, clouds=clouds, board_R=rb, errors=errors,
            speeds=speeds, hands=hands, hand_gaps=gaps, held=held, native_held=native_held,
            grasp_contacts=grasp_contacts, nonfinger_loaded=nonfinger_loaded,
            table_loaded=table_loaded, robot_loaded=robot_loaded,
            board_ok=bool(rb[2, 2]>.995 and abs(board[2]-model.ORIGIN[2]-.003)<.006
                          and np.linalg.norm(board_velocity[:3])<.015),
            finite=bool(all(np.isfinite(v).all() for v in [*poses.values(), hands])))
        with _LOCK:
            self.frames[step] = frame

    def _window(self, event):
        return [v for k, v in self.frames.items() if event['begin'] <= k <= event['step']]

    def _prefix_check(self, event):
        window = self._window(event)
        worst = max((v['errors'][n] for v in window for n in event['prefix']), default=0.)
        contiguous = len(window) == event['step']-event['begin']+1
        return contiguous and worst<.004, dict(prefix=list(event['prefix']),
            protected_window_start=event['begin'], protected_window_end=event['step'],
            protected_points=len(window), protected_samples_contiguous=contiguous,
            maximum_prefix_bead_error_m=worst)

    def _held_check(self, event, state):
        f = self.frames[event['step']]
        piece = event['piece']
        arm = int(state['arm'])
        if arm not in (0, 1):
            return False, dict(error='Invalid selected robot arm')
        gap = float(np.linalg.norm(f['clouds'][piece]-f['hands'][arm, :3], axis=1).min())
        start_z = self.source_centroid_z.get(piece, float(self.frames[min(self.frames)]['clouds'][piece][:, 2].mean()))
        rise = float(f['clouds'][piece][:, 2].mean()-start_z)
        count = int(round(.1*self.hz))+1
        recent = [self.frames[k] for k in range(event['step']-count+1, event['step']+1) if k in self.frames]
        attached = []
        for row in recent:
            tcp = row['hands'][arm]
            attached.append((row['clouds'][piece]-tcp[:3]) @ Rotation.from_quat(tcp[3:7]).as_matrix())
        drift = max((_error(v, attached[0]) for v in attached), default=1.)
        continuous = len(recent)==count and all(row['held'][arm] and not row['table_loaded'][piece]
            and not row['nonfinger_loaded'][arm] for row in recent)
        passed = bool(continuous and drift<.003 and rise>=.025 and gap<=.045)
        return passed, dict(selected_arm=arm, native_RSS_selected_hand_check_grasp=f['native_held'][arm],
            public_joint_axis_grasp=f['held'][arm], actual_grasp_contact_sides=f['grasp_contacts'][arm],
            continuous_grasp_window_s=.1, actual_grasp_window_points=len(recent),
            required_grasp_window_points=count, no_missing_history_filled=True,
            continuous_dual_finger_no_external_support=continuous, actual_attachment_bead_drift_m=drift,
            actual_centroid_rise_m=rise, actual_target_hand_gap_m=gap,
            actual_target_loaded_table=f['table_loaded'][piece], actual_jaw_width_m=float(f['hands'][arm, 7]))

    def _stable(self, event, piece, active_arm):
        names = [*event['prefix'], piece]
        if active_arm not in (0, 1):
            raise ValueError('No actual selected release hand in bound state')
        full12 = len(set(names)) == len(model.CELLS)
        required_tail = 2. if full12 else 1.
        rows = []
        for k, f in reversed(self.frames.items()):
            if k > event['step']:
                continue
            okay = (f['finite'] and f['board_ok'] and max(f['errors'][n] for n in names)<.004
                and max(f['speeds'][n] for n in names)<.015
                and float(f['hands'][active_arm, 7])>.045 and min(f['hand_gaps'])>.055
                and (not full12 or (float(f['hands'][:, 7].min())>.045
                     and float(np.linalg.norm(f['hands'][:, :3]-self.home, axis=1).max())<.015))
                and not any(f['robot_loaded'][n] for n in names))
            if not okay:
                break
            rows.append(f)
        duration = 0. if not rows else (rows[0]['step']-rows[-1]['step'])/self.hz
        f = self.frames[event['step']]
        return duration>=required_tail-1.e-8, dict(released_stable_tail_s=duration,
            selected_release_arm=active_arm, full12_final_goal=full12, required_stable_tail_s=required_tail,
            home_reference_source=self.home_reference_source,
            actual_hand_public_home_distances_m=np.linalg.norm(f['hands'][:, :3]-self.home, axis=1).tolist(),
            target_goal_bead_error_m=f['errors'][piece],
            actual_jaw_widths_m=f['hands'][:, 7].tolist(),
            actual_all_piece_hand_clearances_m=f['hand_gaps'],
            actual_target_robot_loaded=f['robot_loaded'][piece],
            actual_target_max_bead_speed_mps=f['speeds'][piece])

    def _localization_check(self, event, state):
        """Compare authored estimates with the GT frame of that observation.

        Timestamp selects an already captured control state; no nearest-frame,
        checkpoint-frame or GT-pose fallback is permitted. A missing record is
        missing evaluator evidence, never an invented physical observation.
        """
        observed = state['observation']
        timestamp = observed.get('timestamp')
        detail = dict(target_max_bead_error_bound_m=.006, board_max_corner_error_bound_m=.004,
            observation_time_s=timestamp, checkpoint_step=event['step'],
            visual_diagnostics_affect_physical_result=False)
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not np.isfinite(timestamp)):
            return False, dict(detail, localization_evidence_status='missing_exact_observation_time')
        index = float(timestamp)*self.hz
        step = int(round(index))
        detail['bound_observation_step'] = step
        if abs(index-step)>1.e-5 or step>event['step'] or step not in self.frames:
            return False, dict(detail, localization_evidence_status='missing_exact_owner_GT_frame')
        frame = self.frames[step]
        piece = event['piece']
        pose = observed.get('poses', {}).get(piece)
        raw_board = observed.get('poses', {}).get('board')
        planning_board = state.get('planning_scene', {}).get('board_for_planning', {})
        board = planning_board.get('pose', raw_board)
        detail.update(raw_source_present=pose is not None,
            raw_visible_names=sorted(observed.get('poses', {})),
            board_estimate_semantics=('current_bound_observation' if raw_board is not None
                                      else 'conditional_planning_scene_at_bound_observation'),
            planning_board_declared_source=planning_board.get('source'),
            planning_board_source_timestamp=planning_board.get('source_timestamp'),
            historical_board_is_current_measurement=False if raw_board is None else None)
        def valid_pose(value):
            return (_finite(value, (7,)) and abs(np.linalg.norm(np.asarray(value)[3:])-1.)<1.e-4)
        if (not valid_pose(pose) or not valid_pose(board)
                or (raw_board is not None and not valid_pose(raw_board))):
            return False, dict(detail, localization_evidence_status='missing_or_invalid_authored_pose')
        target_error = _error(_cloud(piece, pose), frame['clouds'][piece])
        # Public case outline: rim x=-.0825, lid border x=.1375,
        # rim/lid y=+/- .1575. Compare corresponding corners, not a
        # symmetry-permuted rectangle that could hide a wrong board frame.
        corners = np.array([[x, y, 0.] for x in (-.0825, .1375) for y in (-.1575, .1575)])
        estimated = Rotation.from_quat(np.asarray(board)[3:]).apply(corners)+np.asarray(board)[:3]
        actual_pose = frame['poses']['board']
        actual = Rotation.from_quat(actual_pose[3:]).apply(corners)+actual_pose[:3]
        board_error = float(np.linalg.norm(estimated-actual, axis=1).max())
        current_board_error = None
        if raw_board is not None:
            current = Rotation.from_quat(np.asarray(raw_board)[3:]).apply(corners)+np.asarray(raw_board)[:3]
            current_board_error = float(np.linalg.norm(current-actual, axis=1).max())
            board_error = max(board_error, current_board_error)
        passed = bool(target_error<=.006 and board_error<=.004)
        return passed, dict(detail, localization_evidence_status='exact_owner_GT_frame',
            target_max_matched_bead_error_m=target_error, board_max_corresponding_corner_error_m=board_error,
            current_RGB_board_max_corner_error_m=current_board_error,
            planning_board_checked=planning_board.get('pose') is not None,
            board_public_outline_corners_local_m=corners.tolist(),
            target_localization_passed=target_error<=.006, board_localization_passed=board_error<=.004)

    def validate(self, phase, event, outputs):
        if phase in event['results']:
            return event['results'][phase]['passed']
        detail = dict(phase=phase, piece=event['piece'], native_step=event['step'], policy_input=False)
        passed = False
        try:
            state, evidence = outputs['state'], outputs['evidence']
            piece = event['piece']
            expected_route = _PHASE_ROUTES[phase]
            if outputs.get('decision') not in expected_route:
                raise ValueError('Bound decision does not match the subgraph promise')
            reported_piece = evidence.get('expected_piece') if phase=='handoff' else state['piece']
            if reported_piece != piece:
                raise ValueError('Output target differs from independently ordered task target')
            prefix_ok, pd = self._prefix_check(event)
            detail.update(pd)
            f = self.frames[event['step']]
            if phase=='perceive_plan':
                plan, observed = state['plan'], state['observation']
                pose = observed.get('poses', {}).get(piece)
                grasp = plan.get('grasp')
                rotation = plan.get('rotation', {})
                q = [rotation[k] for k in ('x', 'y', 'z', 'w')]
                extent = np.ptp(model.local_beads(piece), axis=0)+2*model.RADIUS
                plan_sanity = bool(_finite(grasp, (3,)) and _finite(q, (4,))
                    and abs(np.linalg.norm(q)-1.)<1.e-4 and np.all((extent>.01) & (extent<.5)))
                localized, localization = self._localization_check(event, state)
                functional = bool(plan_sanity and localized)
                detail.update(localization, frame_safe_plan_sanity=plan_sanity,
                    RGB_interface_diagnostics=_rgb_interface_diagnostics(observed, piece),
                    source_localization_validated_against_bound_GT=bool(localized))
                if functional:
                    self.source_centroid_z[piece] = float(f['clouds'][piece][:, 2].mean())
            elif phase in ('grasp', 'transport'):
                functional, hd = self._held_check(event, state)
                detail.update(hd)
                if phase=='transport':
                    height = float(state['parameters'].get('transport', .1))
                    actual = (f['clouds'][piece]-f['poses']['board'][:3]) @ f['board_R']
                    target = self.targets[piece]+[0., 0., height]
                    error = _error(actual, target)
                    functional = functional and .025<=height<=.25 and error<=.006
                    detail.update(public_transport_hover_m=height, actual_target_public_hover_error_m=error)
            elif phase=='reobserve':
                # The authored visual route and ledger remain untouched. This
                # checkpoint judges the promised physical released state only.
                functional, sd = self._stable(event, piece, int(state['arm']))
                detail.update(sd, reobserve_visual_diagnostics=_reobserve_diagnostics(state, evidence, piece),
                    visual_diagnostics_affect_physical_result=False)
            elif phase in ('release', 'handoff'):
                active_arm = int(state['arm']) if phase=='release' else (self.last_release or {}).get('active_arm')
                functional, sd = self._stable(event, piece, active_arm)
                detail.update(sd)
                if phase=='handoff':
                    expected_next = self.order[self.index+1] if self.index+1<len(self.order) else piece
                    prior = self.last_release
                    lineage_ok = bool(prior and prior['piece']==piece
                        and state['piece']==expected_next and evidence.get('timestamp', -1)>=prior['rgb_timestamp'])
                    functional = functional and lineage_ok
                    detail.update(expected_next_piece=expected_next, output_next_piece=state['piece'],
                        prior_physical_release_checked=prior is not None,
                        latest_RGB_not_rolled_back=lineage_ok)
            else:
                raise ValueError('Unknown independently verified module phase')
            passed = bool(functional and prefix_ok)
            detail.update(functional_postcondition_passed=bool(functional), prefix_postcondition_passed=bool(prefix_ok))
            if passed:
                self.begin = event['step']
                if phase=='release':
                    self.last_release = dict(piece=piece, step=event['step'], rgb_timestamp=evidence.get('timestamp', -1), active_arm=int(state['arm']))
                if phase=='handoff':
                    if piece not in self.prefix:
                        self.prefix.append(piece)
                    self.index += 1
        except Exception as error:
            detail['eval_error'] = f'{type(error).__name__}: {error}'
        detail['passed'] = passed
        event['results'][phase] = detail
        self.results.append(detail)
        with (self.output/'checkpoint_results.jsonl').open('a') as stream:
            stream.write(json.dumps(_json_safe(detail), allow_nan=False)+'\n')
        return passed


def install(env, expected_order, initialized_prefix, output_dir):
    """Called only by a native rehearsal harness, on the environment owner thread."""
    with _LOCK:
        if id(env) in _SESSIONS:
            raise RuntimeError('Independent verifier already installed for this environment')
        session = _Session(env, expected_order, initialized_prefix, output_dir)
        _SESSIONS[id(env)] = session
    return session


def capture(env, bus, world=None):
    """Bind a read-only evaluator snapshot; return the SAME restricted World.

    Three args marshal via owner bus. Two args are supported only when the
    caller has already marshalled to the owner thread (adapter.capture).
    """
    if world is None:
        world, bus = bus, None
    session = _SESSIONS[id(env)]
    def read():
        session._observe(env)
        with _LOCK:
            piece = session.order[session.index] if session.index<len(session.order) else None
            _EVENTS[id(world)] = dict(world=world, session=session, step=int(env.step_count),
                begin=session.begin, prefix=tuple(session.prefix), piece=piece, results={})
            while len(_EVENTS)>128:
                _EVENTS.popitem(last=False)
    if bus is None:
        read()
    else:
        bus.call(read)
    return world


def validate(phase, world, outputs):
    with _LOCK:
        event = _EVENTS.get(id(world))
        if event is None or event['world'] is not world:
            raise RuntimeError('No independent rehearsal snapshot bound to this World')
        return event['session'].validate(phase, event, outputs)


def verify(phase, world, outputs):
    return validate(phase, world, outputs)


def restore_rehearsal_baselines(env, context_path):
    """Harness-only continuation context, never returned to the policy.

    The referenced original capture and source point authenticate the lift
    baseline. This API cannot restore or change any physical state.
    """
    import hashlib
    context = json.loads(Path(context_path).read_text())
    capture_path = Path(context['native_capture'])
    if hashlib.sha256(capture_path.read_bytes()).hexdigest() != context['native_capture_sha256']:
        raise ValueError('Rehearsal baseline capture hash mismatch')
    manifest = json.loads(Path(context['capture_manifest']).read_text())
    session = _SESSIONS[id(env)]
    with np.load(capture_path, allow_pickle=False) as data:
        matches = np.flatnonzero(data['step'] == context['source_step'])
        if len(matches) != 1:
            raise ValueError('Missing unique original source point')
        for name in context['pieces']:
            if name not in session.order:
                raise ValueError('Baseline target not in independently ordered continuation')
            pose = data['body_q'][int(matches[0]), manifest['entities'][name]['body']]
            session.source_centroid_z[name] = float(_cloud(name, pose)[:, 2].mean())
    (session.output/'restored_rehearsal_baselines.json').write_text(json.dumps(dict(
        policy_input=False, context=context, source_centroid_z=session.source_centroid_z), indent=2)+'\n')


def diagnostics(phase, world, outputs):
    event = _EVENTS.get(id(world))
    if event is None:
        return dict(independent_snapshot_available=False)
    return deepcopy(event['results'].get(phase, dict(independent_snapshot_available=True)))


def finalize(env):
    session = _SESSIONS.pop(id(env))
    env.remove_step_observer(session.observer)
    summary = dict(schema='puzzler.independent_modular_rehearsal_verification.v1', policy_input=False,
        home_reference_source=session.home_reference_source, public_robot_home_positions=session.home.tolist(),
        expected_order=session.order, independently_completed_prefix=session.prefix,
        all_requested_modules_completed=session.index==len(session.order),
        first_step=min(session.frames), final_step=max(session.frames), native_control_points=len(session.frames),
        checkpoint_results=session.results,
        scope='Functional native50Hz verification only; no GT entered policy state or tools; no500Hz contact-absence guarantee.')
    (session.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary
