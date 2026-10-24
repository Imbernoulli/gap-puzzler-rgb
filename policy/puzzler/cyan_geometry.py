"""CYAN public-geometry module; no teacher poses, trajectories or state reads."""
from copy import deepcopy
import json
import numpy as np
from scipy.spatial.transform import Rotation
from . import model, vision_controller as c, conditional_scene, visual_memory
from .vision_observer import goals


def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False,
        default=lambda item: item.tolist() if isinstance(item, np.ndarray)
        else item.item() if isinstance(item, np.generic) else str(item)))


def source_contact_clearance(position, rotation, width, scene, parameters):
    """Keep all original non-target clearance; bound intended finger contact."""
    from .gripper_geometry import clearance, hand_primitives
    board = np.asarray(scene['measurement']['poses']['board'])
    from .finite_release_scene import finite_clearance
    other = finite_clearance(position, rotation, width, scene, exclude=('cyan',))
    target = scene['occupancy']['cyan']
    local, radii = conditional_scene.local_sphere_cover('cyan')
    centers = []
    for value in target['pose_hypotheses']:
        pose = np.asarray(value)
        centers.extend(local@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3])
    sizes = np.tile(radii+target['inflation_m'], len(target['pose_hypotheses']))
    contract = parameters['source_contact_grasp']
    allowed_pairs = contract['permitted_pad_bead_pairs']
    allowed = tuple(shape for shape in hand_primitives(width) if any(
        shape.name == pair['hand_primitive_name'] for pair in allowed_pairs))
    allowed_names = {shape.name for shape in allowed}
    forbidden = tuple(shape for shape in hand_primitives(width) if shape.name not in allowed_names)
    fingers = clearance(position, rotation, width, centers, radius=sizes, primitives=allowed)
    housing = clearance(position, rotation, width, centers, radius=sizes, primitives=forbidden)
    pair_checks = []
    body = np.asarray(target['pose_hypotheses'][0])
    br = Rotation.from_quat(body[3:]).as_matrix()
    approach_reference = body[:3]+br@model.local_beads('cyan')[parameters['grasp_indices'][0]]+[0, 0, parameters['grasp_z']]
    in_contact_zone = (np.linalg.norm((np.asarray(position)-approach_reference)[:2]) < 1.e-5
        and np.asarray(position)[2]-approach_reference[2] <= contract['maximum_vertical_approach_distance_m']+1.e-9)
    for pair in allowed_pairs:
        bead = model.local_beads('cyan')[pair['target_bead_index']]
        # Connector-cover balls entirely inside this public bead are the same
        # permitted contact region, not a licence to touch the exposed neck.
        permitted = np.zeros(len(local),dtype=bool)
        for authorized in allowed_pairs:
            if authorized['hand_primitive_name']==pair['hand_primitive_name']:
                contact_bead=model.local_beads('cyan')[authorized['target_bead_index']]
                permitted |= np.linalg.norm(local-contact_bead,axis=1)+radii <= model.RADIUS+1.e-8
        primitive = tuple(shape for shape in allowed if shape.name == pair['hand_primitive_name'])
        denied = clearance(position, rotation, width, np.asarray(centers)[~permitted],
            radius=sizes[~permitted], primitives=primitive)
        pair_checks.append(dict(pair=pair, all_other_target_regions=denied,
            source_final_descent_zone=bool(in_contact_zone)))
    margin = parameters.get('hand_clearance_margin', .0005)
    passed = (other['clearance_m'] >= margin and housing['clearance_m'] >= margin
        and fingers['clearance_m'] >= (-contract['maximum_public_cad_penetration_m'] if in_contact_zone else margin)
        and all(row['all_other_target_regions']['clearance_m'] >= margin for row in pair_checks))
    return dict(passed=bool(passed), other_objects_and_table=other,
        nonfinger_target_clearance=housing, intended_finger_target_contact=fingers,
        named_pair_checks=pair_checks, declared_target_contact=contract)


def rgb_record(observed):
    return {key: deepcopy(observed[key]) for key in ('poses', 'timestamp', 'frame_sha256',
        'observation_source', 'assessment', 'robot_geometry_provenance') if key in observed}


def planning_scene(observed, state):
    parameters = state['parameters']
    prefix = set(parameters.get('conditional_prefix', ())) | set(state.get('executed_placements', ()))
    raw = observed['poses']
    required = {'board', *(set(model.CELLS)-prefix)}
    missing = required-set(raw)
    if missing:
        raise c.PlanningRefusal('Fresh board and remaining source RGB missing: '+str(sorted(missing)))
    board = np.asarray(raw['board'])
    br = Rotation.from_quat(board[3:]).as_matrix()
    targets = goals(parameters['mode'], parameters['pyramid_plan'])
    occupancy = {}
    for name in model.CELLS:
        if name in prefix:
            center, rotation = targets[name]
            hypotheses = [np.r_[board[:3]+br@center, Rotation.from_matrix(br@rotation).as_quat()].tolist()]
            if name in raw:
                hypotheses.append(raw[name])
            source = 'conditional_prefix_public_goal4mm_union_all_current_rgb_hypotheses'
            margin = .004
        else:
            hypotheses = [raw[name]]
            source = 'current_rgb_measurement'
            margin = 0.
        occupancy[name] = dict(source=source, pose_hypotheses=hypotheses,
            inflation_m=margin, shape='Public bead and connector sphere CAD cover')
    # Matches c.plan's raw measurement contract; uncertain predecessors remain
    # labelled occupancy unions and are never inserted into current RGB poses.
    return dict(contract='modular_cyan_conditional_scene_v1', target='cyan',
        measurement={key: deepcopy(observed[key]) for key in
            ('poses', 'timestamp', 'frame_sha256', 'observation_source', 'prefix_compatibility') if key in observed},
        occupancy=occupancy, conditional_names=sorted(prefix), measured_names=sorted(raw),
        margin_scope='Initialized/executed prefix4mm is an explicit tested precondition, not a measured hidden pose. Contradictory current points remain in the union.')


def prepare_support_anchor(ctx, state, observed):
    """Compute a board-relative opposite-arm brace, current IK and feedback."""
    parameters = state['parameters']
    profile = parameters['support_anchor']
    arm = 1-state['arm']
    board = np.asarray(observed['poses']['board'])
    br = Rotation.from_quat(board[3:]).as_matrix()
    side = 'left' if arm == 0 else 'right'
    offset = np.asarray(profile['board_offset_'+side])
    point = board[:3]+br@offset
    current, current_q = c.hand(ctx, arm)
    if current[2] < .865:
        c.move(ctx, [current[0], current[1], .865], current_q, arm)
    rotation = None
    heading = profile['frame_heading_'+side]
    evidence = []
    for angle in (heading, heading+180.):
        local_frame = c.frame(ctx, angle, arm)
        proposed = c.qdict(Rotation.from_matrix(
            br@Rotation.from_quat(c.qarr(local_frame)).as_matrix()).as_quat())
        checks = [ctx.tool('puzzle.plan_pose', position=target.tolist(), rotation=proposed, arm_id=arm)
            for target in (point+[0, 0, profile['hover_m']], point)]
        evidence.append(dict(heading=angle, checks=checks))
        if all(row['position_error_m'] < .002 and row['rotation_error_rad'] < .03 for row in checks):
            rotation = proposed
            break
    if rotation is None:
        raise c.PlanningRefusal('Public board-relative brace has no native IK solution')
    c.move(ctx, point+[0, 0, profile['hover_m']], rotation, arm)
    opened = ctx.tool('robot.set_grip', arm_id=arm, width_m=profile['width_m'], ramp_steps=18, settle_steps=15)
    c.move(ctx, point, rotation, arm, contact='brace')
    actual, actual_q = c.hand(ctx, arm)
    return dict(contract='current_rgb_board_relative_support_anchor_v1', arm=arm,
        source_board_rgb=rgb_record(observed), board_offset=offset.tolist(), requested_tcp=point.tolist(),
        requested_rotation=rotation, actual_tcp=actual.tolist(), actual_rotation=actual_q,
        native_ik_candidates=evidence, grip_result=opened, width_m=profile['width_m'],
        intended_contact='Board edge brace only; unchanged native force caps.',
        policy_inputs='Current RGB board/public brace profile/robot proprioception and native IK only')


def observe_and_plan(ctx, state):
    observed = c.state(ctx)
    scene = planning_scene(observed, state)
    parameters = state['parameters']
    arm = 0 if observed['poses']['cyan'][1] >= 0 else 1
    plan = c.plan(ctx, 'cyan', parameters, arm, observation=observed, planning_scene=scene)
    plan = {**plan, 's': rgb_record(observed)}
    memory = visual_memory.observe(state['memory'], observed, parameters)
    return _plain(dict(route='planned', state={**state, 'arm': arm, 'plan': plan, 'planning_scene': scene,
        'observation': rgb_record(observed), 'memory': memory,
        'parameters': {**parameters, 'historical_complete_rgb': rgb_record(observed)}},
        evidence=dict(kind='modular_geometry_plan', timestamp=observed['timestamp'],
            frame_sha256=observed['frame_sha256'], source_body=observed['poses']['cyan'],
            runtime_world_target_source='Current RGB board + public goal CAD; no teacher coordinates')))


def _source(observed, name):
    q = observed['robot_geometry_provenance']
    return dict(mode='pyramid', piece=name, pose=deepcopy(observed['poses'][name]),
        timestamp=observed['timestamp'], frame_sha256=deepcopy(observed['frame_sha256']),
        observation_source='rgb_shape_geometry', robot_encoder_values=q['robot_encoder_values'],
        robot_encoder_sha256=q['robot_encoder_sha256'], source_step=q['step'],
        source_role='Actual accepted current postlift RGB and synchronous robot q16')


def accepted_postgrasp_source(fresh, fallback, name='cyan'):
    """Select an exact accepted RGB epoch; never combine fields across epochs.

    A post-grasp observation may hide the source piece. In that case the
    fallback is the already accepted pregrasp record and the caller must keep
    the resulting held state conditional/UNKNOWN until an independent visual
    receipt exists.
    """
    poses=fresh.get('poses',{}) if isinstance(fresh,dict) else {}
    # ``VisualObserver`` keeps the estimator's key even when visibility is
    # rejected; such an entry has ``None`` as its pose and is a diagnostic
    # proposal, not an owned accepted RGB record.  Only a finite pose from an
    # actually visible detection may be bound as the fresh epoch.
    candidate = poses.get(name)
    detection = fresh.get('detections', {}).get(name, {}) if isinstance(fresh, dict) else {}
    if candidate is not None:
        candidate = np.asarray(candidate, dtype=float)
    if (detection.get('visible') is True and candidate is not None
            and candidate.shape == (7,) and np.isfinite(candidate).all()):
        return dict(piece=name, pose=deepcopy(poses[name]),
            timestamp=fresh['timestamp'],
            frame_sha256=deepcopy(fresh['frame_sha256']),
            observation_source=fresh.get('observation_source','rgb_shape_geometry'),
            robot_geometry_provenance=deepcopy(fresh.get('robot_geometry_provenance',{})),
            source_role='Actual accepted post-grasp RGB; synchronous q16 is retained as provenance'), \
            'fresh_postgrasp_accepted_rgb'
    # An overview proposal rejected by the owned observer is not an accepted
    # source. Keep it UNKNOWN and bind only the original actual pregrasp RGB.
    if isinstance(fallback,dict) and name in fallback.get('poses',{}):
        prior=fallback['poses'][name]
        if prior is None or np.asarray(prior,dtype=float).shape!=(7,) or not np.isfinite(prior).all():
            raise c.PlanningRefusal('No actual accepted pregrasp source is available')
        return dict(piece=name,pose=deepcopy(prior),timestamp=fallback['timestamp'],
            frame_sha256=deepcopy(fallback['frame_sha256']),
            observation_source=fallback.get('observation_source','rgb_shape_geometry'),
            robot_geometry_provenance=deepcopy(fallback.get('robot_geometry_provenance',{})),
            source_role='Dated owned pregrasp RGB only; current postgrasp pose remains UNKNOWN'), \
            'pregrasp_accepted_rgb_conditional_unknown'
    if isinstance(fallback,dict) and fallback.get('piece')==name and fallback.get('pose') is not None:
        return deepcopy(fallback),'pregrasp_accepted_rgb_conditional_unknown'
    raise c.PlanningRefusal('Unowned overview proposal cannot replace actual accepted pregrasp RGB')


def _rigid_hint(body, old_p, old_q, new_p, new_q):
    delta = Rotation.from_quat(c.qarr(new_q))*Rotation.from_quat(c.qarr(old_q)).inv()
    body = np.asarray(body)
    return np.r_[new_p+delta.apply(body[:3]-old_p),
        (delta*Rotation.from_quat(body[3:])).as_quat()].tolist()


def grasp_and_bind(ctx, state):
    """Current RGB source grasp; only actual postlift RGB binds the held pose."""
    parameters = state['parameters']
    arm = state['arm']
    plan = state['plan']
    anchor = prepare_support_anchor(ctx, state, state['observation'])
    point = np.asarray(plan['grasp'])
    rotation = plan['rotation']
    c.grip(ctx, arm, parameters['source_open_width'])
    c.move(ctx, point+[0, 0, parameters['source_hover']], rotation, arm)
    c.move(ctx, point, rotation, arm, cartesian=True, contact=False, accuracy=.0025)
    if parameters.get('source_endpoint_feedback_module') == 'puzzler.cyan_vertical_source_v2':
        from .cyan_vertical_source_v2 import refine_open_endpoint
        state = refine_open_endpoint(ctx, state)
        plan = state['plan']
        point = np.asarray(plan['grasp'])
        rotation = plan['rotation']
    close_p, close_q = c.hand(ctx, arm)
    from .cyan_inverse_tip import source_contact_clearance as inverse_source_guard
    actual_guard=inverse_source_guard(close_p,Rotation.from_quat(c.qarr(close_q)).as_matrix(),parameters['source_open_width'],state['planning_scene'],parameters)
    if not actual_guard['passed']:raise c.PlanningRefusal('Actual encoder source endpoint violates inverse-tip contact CAD contract: '+str(actual_guard))
    plan['actual_encoder_source_contact_guard']=actual_guard
    c.grip(ctx, arm, parameters.get('close_width', 0.))
    # Refresh the source RGB/q16 epoch after the bounded approach. The packet
    # is accepted only when CYAN is actually present in that packet. When the
    # grasp occludes CYAN (the normal post-grasp case), retain the exact
    # pregrasp packet already accepted by VisualObserver; never splice a fresh
    # frame identity onto an old pose. The latter would fail the owned-record
    # provenance check and would falsely turn UNKNOWN visibility into a pose.
    fresh_source=c.state(ctx)
    dated_source, source_epoch = accepted_postgrasp_source(
        fresh_source, plan['s'], name='cyan')
    attachment = ctx.tool('puzzle.capture_conditional_attachment', source_rgb=dated_source,
        assumption=parameters['conditional_attachment_prior'], arm_id=arm)
    lift_start,lift_q=c.hand(ctx,arm)
    c.move(ctx, lift_start+[0, 0, parameters['lift']], lift_q, arm)
    actual, actual_q = c.hand(ctx, arm)
    # The initial source remained still until closure is a labelled optimizer
    # proposal. It is never promoted to an accepted same-frame held source.
    conditional = ctx.tool('puzzle.propagate_conditional_attachment',
        attachment_id=attachment['attachment_id'], arm_id=arm)
    prediction = conditional['predicted_body_pose']
    observed = c.state(ctx, {'cyan': prediction})
    inspections = []
    if 'cyan' not in observed['poses']:
        from .gripper_geometry import scene_clearance
        # One bounded upward observation move, from current encoders. Other
        # objects retain their explicitly historical4mm occupancy assumption.
        inspection_rise=float(parameters.get('postlift_inspection_rise_m',.035))
        if not .0<inspection_rise<=.035:
            raise c.PlanningRefusal('Source inspection rise exceeds bounded35mm proposal')
        target = actual+[0, 0, inspection_rise]
        inspection_ik=ctx.tool('puzzle.plan_pose',position=target.tolist(),rotation=actual_q,arm_id=arm,min_joint_limit_margin_rad=.035)
        if (inspection_ik['position_error_m']>=.002 or inspection_ik['rotation_error_rad']>=.03
                or inspection_ik.get('joint_limit_margin_rad',0.)<.035):
            raise c.PlanningRefusal('Actual bounded source inspection native IK/reserve refused')
        scene = deepcopy(state['planning_scene'])
        for name, item in scene['occupancy'].items():
            if name != 'cyan':
                item['inflation_m'] = max(.004, item['inflation_m'])
                item['source'] += '_historical_during_source_grasp_only'
        source_R = Rotation.from_quat(c.qarr(actual_q)).as_matrix()
        checks = [conditional_scene.hand_clearance(actual+(target-actual)*u, source_R,
            .055, scene, exclude=('cyan',), plane_z=float(np.asarray(
                state['parameters']['historical_complete_rgb']['poses']['board'])[2]-.003))
            for u in np.linspace(0, 1, 9)]
        inspections.append(dict(target_tcp=target.tolist(), current_robot_source=actual.tolist(),
            clearance=checks, pose_proposal_only=prediction))
        if min(row['clearance_m'] for row in checks) >= .003:
            c.move(ctx, target, actual_q, arm, cartesian=True, accuracy=.0025)
            moved, moved_q = c.hand(ctx, arm)
            prediction = _rigid_hint(prediction, actual, actual_q, moved, moved_q)
            observed = c.state(ctx, {'cyan': prediction})
    memory = visual_memory.observe(state['memory'], observed, parameters)
    common = {**state, 'memory': memory, 'observation': rgb_record(observed),
        'anchor_evidence': anchor, 'prediction': prediction, 'inspection_evidence': inspections,
        'conditional_attachment': attachment,
        'source_epoch': source_epoch,
        'postgrasp_observation': rgb_record(fresh_source)}
    if 'cyan' not in observed['poses']:
        conditional = ctx.tool('puzzle.propagate_conditional_attachment',
            attachment_id=attachment['attachment_id'], arm_id=arm)
        return _plain(dict(route='conditional', state={**common, 'held_status': 'CONDITIONAL_ATTACHMENT_UNKNOWN_3D',
            'no_repeat_grasp': True, 'parameters': {**parameters, 'conditional_attachment': attachment}},
            evidence=dict(kind='modular_grasp_conditional',
            timestamp=observed['timestamp'], frame_sha256=observed['frame_sha256'],
            target_3d_status='UNKNOWN', no_held_pose_source_created=True,
            conditional_attachment=attachment, conditional_control_proposal=conditional)))
    tcp, _ = c.hand(ctx, arm)
    body = observed['poses']['cyan']
    rise = float(body[2]-plan['body'][2])
    gap = c.part_hand_gap('cyan', body, tcp)
    if rise < .025 or gap > .045:
        return dict(route='lost', state={**common, 'held_status': 'RGB_NOT_LIFTED_NEAR_HAND'},
            evidence=dict(kind='modular_grasp_lost', timestamp=observed['timestamp'],
                frame_sha256=observed['frame_sha256'], rise_m=rise, hand_bead_gap_m=gap))
    source = _source(observed, 'cyan')
    return dict(route='verified', state={**common, 'held_status': 'CURRENT_RGB_VERIFIED',
        'parameters': {**parameters, 'owned_held_belief_source': source}},
        evidence=dict(kind='modular_grasp', timestamp=observed['timestamp'],
            frame_sha256=observed['frame_sha256'], rise_m=rise, hand_bead_gap_m=gap,
            source=source))


def carry_target(ctx, state):
    """Choose a current-held-transform target frame using public CAD symmetry."""
    parameters = state['parameters']
    arm = state['arm']
    if 'owned_held_belief_source' in parameters:
        belief = ctx.tool('puzzle.propagate_pyramid_held_belief',
            source=parameters['owned_held_belief_source'], arm_id=arm)
    else:
        belief = ctx.tool('puzzle.propagate_conditional_attachment',
            attachment_id=parameters['conditional_attachment']['attachment_id'], arm_id=arm)
    body = np.asarray(belief['predicted_body_pose'])
    board = np.asarray(parameters.get('current_board_for_inverse_planning', {}).get('pose', parameters['historical_complete_rgb']['poses']['board']))
    br = Rotation.from_quat(board[3:]).as_matrix()
    tcp, q = c.hand(ctx, arm)
    source_R = Rotation.from_quat(body[3:]).as_matrix()
    public_center, public_R = goals('pyramid', parameters['pyramid_plan'])['cyan']
    local = model.local_beads('cyan')
    candidates = []
    for turn in range(4):
        for flip in (False, True):
            symmetry = model.rotation(turn*np.pi/2)@(np.diag([1., -1., -1.]) if flip else np.eye(3))
            if np.max(np.min(np.linalg.norm((local@symmetry.T)[:, None]-local[None], axis=2), axis=1)) > 1.e-7:
                continue
            relative = public_R@symmetry
            desired = br@relative
            delta = desired@source_R.T
            rotation = c.qdict(Rotation.from_matrix(delta@Rotation.from_quat(c.qarr(q)).as_matrix()).as_quat())
            target = board[:3]+br@public_center
            zero_tcp = target+delta@(tcp-body[:3])
            hover = zero_tcp+[0, 0, parameters['transport']]
            check = ctx.tool('puzzle.plan_pose', position=hover.tolist(), rotation=rotation, arm_id=arm)
            candidates.append(dict(relative_target_R=relative.tolist(), target_tcp=zero_tcp.tolist(),
                hover=hover.tolist(), rotation=rotation, native_ik=check))
    feasible = [row for row in candidates if row['native_ik']['position_error_m'] < .002
        and row['native_ik']['rotation_error_rad'] < .03
        and row['native_ik'].get('joint_limit_margin_rad', 0.) >= .035]
    if not feasible:
        raise c.PlanningRefusal('Actual held RGB/FK transform has no public-goal-symmetric hover IK')
    feasible.sort(key=lambda row: row['native_ik']['position_error_m']+.05*row['native_ik']['rotation_error_rad'])
    selected = feasible[0]
    return selected, candidates, belief


def move_held_screened(ctx, state, target, belief):
    """Plan from current FK, check hand/held CAD, stream that newly solved path."""
    arm = state['arm']; parameters = state['parameters']
    plan = ctx.tool('puzzle.plan_path', position=target['hover'], rotation=target['rotation'],
        arm_id=arm, min_joint_limit_margin_rad=.035)
    if not plan.get('feasible') or len(plan.get('tcp_path', ())) != len(plan.get('path', ())):
        raise c.PlanningRefusal('Current-held public-target Cartesian IK path refused')
    samples = plan['tcp_path']
    if len(samples) < 2:
        raise c.PlanningRefusal('Missing robot FK path samples')
    scene = deepcopy(state['planning_scene'])
    for name, entry in scene['occupancy'].items():
        if name != 'cyan':
            entry['inflation_m'] = max(.004, entry['inflation_m'])
            entry['source'] += '_historical4mm_during_bounded_grasp'
    obstacles, sizes = conditional_scene.spheres(scene, exclude=('cyan',))
    board = np.asarray(parameters.get('current_board_for_inverse_planning', {}).get('pose', parameters['historical_complete_rgb']['poses']['board']))
    body = np.asarray(belief['predicted_body_pose'])
    tcp, q = c.hand(ctx, arm)
    robot_R = Rotation.from_quat(c.qarr(q)).as_matrix()
    relative_p = robot_R.T@(body[:3]-tcp)
    relative_R = robot_R.T@Rotation.from_quat(body[3:]).as_matrix()
    local, local_sizes = conditional_scene.local_sphere_cover('cyan')
    q16 = belief['current_robot_encoder_values']; offset = 0 if arm == 0 else 8
    actual_width = float(q16[offset+6]-q16[offset+7])
    from .gripper_geometry import clearance, scene_clearance
    checks = []
    for sample in samples:
        position = np.asarray(sample['position'])
        rotation = Rotation.from_quat([sample['rotation'][key] for key in 'xyzw']).as_matrix()
        hand = conditional_scene.hand_clearance(position, rotation, actual_width, scene,
            exclude=('cyan',), plane_z=float(board[2]-.003+.004))
        centers = local@(rotation@relative_R).T+position+rotation@relative_p
        pair = np.linalg.norm(centers[:, None]-obstacles[None], axis=2)-local_sizes[:, None]-sizes[None]-.006
        held = min(float(pair.min()), float((centers[:, 2]-local_sizes-.006-(board[2]-.003+.004)).min()))
        checks.append(dict(hand_clearance_m=hand['clearance_m'], held_cad_clearance_m=held,
            held_pose_role='Conditional sourceRGB/robotFK6mm; no current pose fill'))
    if min(min(row['hand_clearance_m'], row['held_cad_clearance_m']) for row in checks) < .002:
        raise c.PlanningRefusal('Current FK carry path violates hand or6mm conditional held CAD clearance')
    # The path and every waypoint originate from this live native IK call.
    waypoints = []
    for aa, bb in zip(np.asarray(plan['path'])[:-1], np.asarray(plan['path'])[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(bb-aa))/.018)))
        waypoints.extend(dict(positions=(aa+(bb-aa)*u).tolist()) for u in np.linspace(0, 1, count+1)[1:])
    moved = ctx.tool('robot.execute_trajectory', trajectory=dict(waypoints=waypoints),
        arm_id=arm, max_steps_per_waypoint=2)
    if moved.get('status') != 'ok':
        raise RuntimeError('Current native IK carry trajectory failed')
    c.settle_motion(ctx, target['hover'], target['rotation'], arm, False, .0025)
    return dict(contract='actual_rgb_fk_public_geometry_carry_v1', native_ik=plan,
        cad_sample_checks=checks, sample_clearance_scope='Native sampled FK;4mm historical scene and6mm rigid-held assumptions, not whole-arm or between-sample collision proof.',
        source=parameters.get('owned_held_belief_source', parameters.get('conditional_attachment')), held_belief=belief, selected_goal=target,
        native_motor_result=moved)


def plan_checkpoint(world, outputs):
    try:
        state = outputs['state']; evidence = outputs['evidence']; observed = world.sim_state['vision']
        return (outputs.get('decision') == 'planned' and evidence['kind'] == 'modular_geometry_plan'
            and evidence['timestamp'] == observed['timestamp'] and evidence['frame_sha256'] == observed['frame_sha256']
            and state['plan']['body'] == observed['poses']['cyan']
            and state['planning_scene'] == planning_scene(observed, state)
            and all(row['passed'] for row in state['plan']['source_contact_checks'])
            and state['plan']['source_grasp_only'] is True
            and state['plan']['target_release_validated'] is False)
    except (KeyError, ValueError, TypeError):
        return False


def grasp_checkpoint(world, outputs):
    try:
        evidence = outputs['evidence']; state = outputs['state']; observed = world.sim_state['vision']
        if evidence['timestamp'] != observed['timestamp'] or evidence['frame_sha256'] != observed['frame_sha256']:
            return False
        if outputs.get('decision') == 'unknown':
            return (evidence['kind'] == 'modular_grasp_unknown' and 'cyan' not in observed['poses']
                and evidence['no_held_pose_source_created'] is True)
        if outputs.get('decision') == 'conditional':
            token = evidence['conditional_attachment']
            proposal = evidence['conditional_control_proposal']
            return (evidence['kind'] == 'modular_grasp_conditional'
                and 'cyan' not in observed['poses'] and evidence['target_3d_status'] == 'UNKNOWN'
                and token['rgb_and_close_are_same_epoch'] is False
                and token == state['parameters']['conditional_attachment']
                and proposal in world.sim_state['pyramid_held_belief_history']
                and proposal['source'] == token and proposal['current_target_measured'] is False
                and world.sim_state['robot_tcp_poses'][str(state['arm'])][2]-state['plan']['grasp'][2] >= .025)
        if outputs.get('decision') == 'lost':
            return evidence['kind'] == 'modular_grasp_lost' and 'cyan' in observed['poses']
        source = state['parameters']['owned_held_belief_source']
        proprio = observed['robot_geometry_provenance']
        tcp = np.asarray(world.sim_state['robot_tcp_poses'][str(state['arm'])][:3])
        return (outputs.get('decision') == 'verified' and evidence['kind'] == 'modular_grasp'
            and source == _source(observed, 'cyan')
            and source['robot_encoder_values'] == proprio['robot_encoder_values']
            and observed['poses']['cyan'][2]-state['plan']['body'][2] >= .025
            and c.part_hand_gap('cyan', observed['poses']['cyan'], tcp) <= .045)
    except (KeyError, ValueError, TypeError):
        return False
