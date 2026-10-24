"""Observable state and semantic stages for the shape-RGB GaP policy.

Only public CAD, image-derived poses and robot proprioception enter this module.
Physics evaluation belongs to the external rehearsal observer. A confirmed loss
is a legal stage outcome; an uncertain attachment or failed motor action aborts.
"""
import json
from typing import Any, TypedDict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from puzzler import model
from puzzler import vision_controller as c
from puzzler import visual_memory


class StageParameters(TypedDict):
    mode: str
    anchor: bool
    min_ik_joint_margin_rad: float


class StageState(TypedDict):
    piece: str
    attempt: int
    parameters: StageParameters
    arm: int
    plan: dict[str, Any]
    observation: dict[str, Any]
    prediction: list[float]
    history: list[dict[str, Any]]
    last_stage: str
    last_outcome: str
    memory: visual_memory.VisualMemory
    pending_placement_receipt: dict[str, Any]


class StageEvidence(TypedDict, total=False):
    kind: str
    timestamp: float
    frame_sha256: dict[str, str]
    rise_m: float
    hand_bead_gap_m: float
    loss_confirmed: bool
    scene_complete: bool
    max_displacement_m: float
    visual_error_m: float
    stable_released: bool
    hands_clear: bool
    minimum_hand_bead_gap_m: float
    hand_tcp_positions: list[list[float]]
    prior_timestamp: float
    prior_frame_sha256: dict[str, str]
    prior_hand_tcp_poses: list[list[float]]
    retry_history_count: int
    all_goal_errors_m: dict[str, float]
    max_hand_drift_m: float
    max_hand_angular_drift_rad: float


class StageResult(TypedDict):
    route: str
    state: StageState
    evidence: StageEvidence


# This is the pinned engine's public type-name registry. Register project types
# before its native validator introspects subgraph I/O; no runtime is upgraded.
from gap_core.schema import TYPE_REGISTRY
for _stage_type in (visual_memory.PlacementReceipt, visual_memory.ReceiptTrack,
                    visual_memory.PrefixAssumption, visual_memory.VisualMemory,
                    StageParameters, StageState, StageEvidence, StageResult):
    TYPE_REGISTRY[_stage_type.__name__] = _stage_type


class StageRefusal(RuntimeError):
    """There is insufficient observable evidence for the next stage."""


def initial_state(piece: str, parameters: StageParameters) -> StageState:
    return dict(piece=piece, attempt=0, parameters=dict(parameters), arm=-1,
                plan={}, observation={}, prediction=[], history=[],
                last_stage='', last_outcome='', memory=visual_memory.new_memory(),
                pending_placement_receipt={})


def _advance_piece(state):
    """Reset piece-local state while retaining an explicitly bound ledger."""
    if state['last_stage'] != 'handoff' or state['last_outcome'] != 'carried':
        return state
    order = state['parameters']['assembly_order']
    index = order.index(state['piece'])+1
    if index >= len(order):
        raise StageRefusal('The last piece handoff must enter finish')
    next_state = initial_state(order[index], state['parameters'])
    next_state['memory'] = _plain(state['memory'])
    return next_state


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _plain(value):
    return json.loads(json.dumps(value, default=_json_value, allow_nan=False))


def _observation(observed):
    # Refused-fit diagnostics can contain unbounded optimization residuals.
    # Persist finite accepted poses and their sensor provenance in graph state.
    packet = {k: observed[k] for k in
              ('poses', 'detections', 'timestamp', 'frame_sha256', 'assessment', 'observation_source',
               'bounded_rgb_source', 'robot_geometry_provenance', 'goal_pixel_support')
              if k in observed}
    if 'assembly_completion' in observed:
        packet['assembly_completion'] = {k: v for k, v in observed['assembly_completion'].items() if k != 'views'}
    return packet


def _result(route, state, observed, **evidence) -> StageResult:
    if 'timestamp' not in observed or not observed.get('frame_sha256'):
        raise StageRefusal('A stage outcome requires recorded fresh RGB provenance')
    memory = visual_memory.observe(state['memory'], observed, state['parameters'])
    if evidence.get('kind')=='finish' and route=='complete':
        requested=state['parameters'].get('diagnostic_subset') or list(model.CELLS)
        if any(name not in memory['placements'] or memory['placements'][name]['status']=='unknown_or_conflicted'
               for name in requested):
            raise StageRefusal('Finish requires an uncontradicted actual release receipt for each requested piece')
    return _plain(dict(route=route, state={**state, 'memory': memory, 'observation': _observation(observed),
                                         'last_stage': evidence.get('kind', ''), 'last_outcome': route},
                       evidence=dict(timestamp=observed['timestamp'],
                                     frame_sha256=observed['frame_sha256'], **evidence)))


def _require(observed, names):
    missing=sorted(set(names)-set(observed.get('poses', {})))
    if missing:
        raise StageRefusal('Fresh RGB pose support is missing for '+str(missing))
    return observed


def _pause(ctx, steps):
    result=ctx.tool('robot.wait_steps', steps=steps)
    if result is not None and result.get('waited', steps)<steps:
        raise StageRefusal('The observation stability wait did not complete')


def _piece_parameters(state):
    p=state['parameters']
    if p.get('anchor', False):
        raise StageRefusal('The staged policy currently requires the support hand parked')
    return {**p, **p.get('pieces', {}).get(state['piece'], {})}


def _scene_names():
    return ['board', *model.CELLS]


def _beads(name, pose):
    pose=np.asarray(pose)
    return model.local_beads(name)@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]


def _displacement(before, after, names):
    movement={}
    for name in names:
        aa=_beads(name, before['poses'][name]);bb=_beads(name, after['poses'][name])
        cost=np.linalg.norm(aa[:,None]-bb[None,:], axis=2)
        ii,jj=linear_sum_assignment(cost)
        movement[name]=float(cost[ii,jj].max())
    return movement


def _board_displacement(before, after):
    # The fitted outline is the public 220 x 315 mm case CAD, offset
    # 27.5 mm from the board origin. Track corners so yaw cannot hide behind
    # an unchanged center position.
    corners=np.array([[.0275+sx*.110,sy*.1575,.013]
                      for sx in (-1,1) for sy in (-1,1)])
    positions=[]
    for observed in (before,after):
        board=np.asarray(observed['poses']['board'])
        positions.append(corners@Rotation.from_quat(board[3:]).as_matrix().T+board[:3])
    return float(np.linalg.norm(positions[1]-positions[0],axis=1).max())


def _held_or_lost(ctx, state, observed, prediction, source_z):
    name=state['piece'];arm=state['arm']
    _require(observed, [name])
    body=np.asarray(observed['poses'][name]);tcp,_=c.hand(ctx, arm)
    gap=c.part_hand_gap(name, body, tcp)
    rise=float(body[2]-source_z)
    # A board-visible detection alone does not prove the object is held.
    if gap>.045:
        return 'lost', dict(loss_confirmed=True, rise_m=rise, hand_bead_gap_m=gap)
    if rise<.025 or observed.get('grasp_prediction_error_m', 0.)>.025:
        raise StageRefusal('Target near the hand has no verified lifted attachment')
    _require(observed, ['board'])
    return 'verified', dict(loss_confirmed=False, rise_m=rise, hand_bead_gap_m=gap)


def observe_plan(ctx, state: StageState) -> StageResult:
    """Fresh scene -> selected source arm and screened public-CAD grasp."""
    state = _advance_piece(state)
    name=state['piece'];p=_piece_parameters(state)
    observed=_require(c.state(ctx), _scene_names())
    arm=0 if observed['poses'][name][1]>=0 else 1
    tcp,q=c.hand(ctx, arm)
    if tcp[2]<.865:
        c.move(ctx, [tcp[0],tcp[1],.865], q, arm)
    # A cleared retry carries its history explicitly; do not repeat a failed
    # bead grasp merely because the graph called the same skill again.
    excluded={h['grasp_index'] for h in state['history'] if 'grasp_index' in h}
    if excluded:
        allowed=set(p.get('grasp_indices', range(len(model.local_beads(name)))))-excluded
        if not allowed:
            raise StageRefusal('The bounded retry has no unused bead grasp')
        p={**p, 'grasp_indices': sorted(allowed)}
    plan=c.plan(ctx, name, p, arm)
    observed=_require(plan['s'], _scene_names())
    plan={**plan, 's': _observation(observed)}
    updated={**state, 'arm': arm, 'plan': plan, 'prediction': []}
    return _result('planned', updated, observed, kind='plan', scene_complete=True,
                   retry_history_count=len(state['history']))


def grasp_and_verify(ctx, state: StageState) -> StageResult:
    """Execute the screened source grasp, then measure lift and hand proximity."""
    name=state['piece'];p=_piece_parameters(state);arm=state['arm'];plan=state['plan']
    g=np.asarray(plan['grasp']);q=plan['rotation']
    c.grip(ctx, arm, p.get('open_width', .055))
    c.move(ctx, g+[0,0,p.get('source_hover', .11)], q, arm)
    c.move(ctx, g, q, arm, cartesian=True, contact='grasp')
    c.grip(ctx, arm, p.get('close_width', 0.))
    lift=p.get('lift', .085)
    c.move(ctx, g+[0,0,lift], q, arm)
    prediction=np.asarray(plan['body']).copy();prediction[2]+=lift
    observed=c.inspect_held(ctx, name, prediction, arm)
    route,evidence=_held_or_lost(ctx, state, observed, prediction, plan['body'][2])
    return _result(route, {**state, 'prediction': prediction.tolist()}, observed,
                   kind='grasp', **evidence)


def transport_and_verify(ctx, state: StageState) -> StageResult:
    """Measure the current grasp transform, carry, and register again from RGB."""
    name=state['piece'];p=_piece_parameters(state);arm=state['arm'];plan=state['plan']
    current=c.inspect_held(ctx, name, np.asarray(state['observation']['poses'][name]), arm)
    route,evidence=_held_or_lost(ctx, state, current, state['prediction'], plan['body'][2])
    if route=='lost':
        return _result('lost', state, current, kind='transport', **evidence)
    body=np.asarray(current['poses'][name]);board=np.asarray(current['poses']['board'])
    rb=Rotation.from_quat(board[3:]).as_matrix();desired=rb@np.asarray(plan['target_R'])
    delta=desired@Rotation.from_quat(body[3:]).as_matrix().T
    tcp,q=c.hand(ctx, arm)
    qtarget=c.qdict(Rotation.from_matrix(delta@Rotation.from_quat(c.qarr(q)).as_matrix()).as_quat())
    target=board[:3]+rb@np.asarray(plan['target_center'])
    target_tcp=target+delta@(tcp-body[:3]);height=p.get('transport', .09)
    c.move(ctx, target_tcp+[0,0,height], qtarget, arm)
    prediction=np.r_[target+[0,0,height], Rotation.from_matrix(desired).as_quat()]
    observed=c.inspect_held(ctx, name, prediction, arm)
    route,evidence=_held_or_lost(ctx, state, observed, prediction, plan['body'][2])
    return _result(route, {**state, 'prediction': prediction.tolist()}, observed,
                   kind='transport', **evidence)


def release_and_verify(ctx, state: StageState) -> StageResult:
    """Release only with current obstacle support, then measure a stable scene."""
    name=state['piece'];p=_piece_parameters(state);arm=state['arm'];plan=state['plan']
    # An image that localizes the carried part can still leave an obstacle
    # ambiguous. Bounded screened observation motions may reacquire that scene;
    # their predictions/alternative obstacle poses never replace measurements.
    observed=c.inspect_scene_held(ctx, name, np.asarray(state['observation']['poses'][name]), arm, p)
    # Active inspection can move the hand or reveal a loss. Recompute heldness
    # from its returned current RGB and actual encoders before precision motion.
    route,evidence=_held_or_lost(ctx, state, observed, state['prediction'], plan['body'][2])
    if route=='lost':
        return _result('lost', state, observed, kind='release', **evidence)
    # Never fill an occluded neighbor from a pre-grasp snapshot. Contact may
    # have moved it, and a stale point pose does not bound its uncertainty.
    _require(observed, _scene_names())
    body=np.asarray(observed['poses'][name]);board=np.asarray(observed['poses']['board'])
    rb=Rotation.from_quat(board[3:]).as_matrix();desired=rb@np.asarray(plan['target_R'])
    correction=desired@Rotation.from_quat(body[3:]).as_matrix().T
    tcp,q=c.hand(ctx, arm)
    qtarget=c.qdict(Rotation.from_matrix(correction@Rotation.from_quat(c.qarr(q)).as_matrix()).as_quat())
    target=board[:3]+rb@np.asarray(plan['target_center'])
    target_tcp=target+correction@(tcp-body[:3])
    height=c.clear_release_height(name, observed['poses'], target_tcp, qtarget, p)
    c.move(ctx, target_tcp+[0,0,height], qtarget, arm, accuracy=.0025)
    c.grip(ctx, arm, p.get('open_width', .055))
    c.withdraw(ctx, arm)
    _pause(ctx, 50)
    prediction=np.r_[target, Rotation.from_matrix(desired).as_quat()]
    before=c.inspect_released(ctx, name, prediction, arm)
    _pause(ctx, 50)
    after=_require(c.state(ctx, {name: before['poses'][name]}), ['board', name])
    if after['timestamp']<=before['timestamp']:
        raise StageRefusal('Released-state evidence did not advance sensor time')
    displacement=max(_displacement(before, after, [name])[name],_board_displacement(before,after))
    tcp,_=c.hand(ctx, arm);gap=c.part_hand_gap(name, after['poses'][name], tcp)
    if displacement>=.0015 or gap<=.055:
        raise StageRefusal('The released target is not stable and separated from the hand')
    error=after['assessment']['bead_error_m'][name]
    if error is None:
        raise StageRefusal('The released target has no fresh visual goal residual')
    # This separate pending record contains actual RGB pair/encoder evidence.
    # It is not an accepted receipt until the success-only placement boundary.
    pending = (visual_memory.release_receipt(name, before, after, tcp, p)
               if error < .003 else {})
    state = {**state, 'pending_placement_receipt': pending}
    return _result('placed' if error<.003 else 'missed', state, after,
                   kind='release', stable_released=True, loss_confirmed=False,
                   visual_error_m=float(error), max_displacement_m=displacement,
                   hand_bead_gap_m=gap, prior_timestamp=before['timestamp'],
                   prior_frame_sha256=before['frame_sha256'])


def confirm_placement(ctx, state: StageState) -> StageResult:
    """Success-only graph boundary; no missed attempt reaches this checkpoint."""
    observed=state['observation'];name=state['piece']
    error=observed['assessment']['bead_error_m'][name]
    if error is None or error>=.003:
        raise StageRefusal('Placement confirmation was reached without a valid goal residual')
    if state['last_stage'] != 'release' or state['last_outcome'] != 'placed' or not state['pending_placement_receipt']:
        raise StageRefusal('Placement receipt requires a completed successful RGB release')
    memory = visual_memory.mint_placement(state['memory'], state['pending_placement_receipt'],
                                          observed, _piece_parameters(state))
    return _result('accepted', {**state, 'memory': memory, 'pending_placement_receipt': {}},
                   observed, kind='placement', visual_error_m=float(error))


def handoff_piece(ctx, state: StageState, expected_piece: str) -> StageResult:
    """Join successful bounded retry paths through one unique typed output."""
    if (state['piece'] != expected_piece or state['last_stage'] != 'placement'
            or state['last_outcome'] != 'accepted'):
        raise StageRefusal('Piece handoff did not receive its successful placement state')
    track = state['memory']['placements'].get(expected_piece)
    if not track or track['status'] != 'measured_now' or track['invalidation_reason']:
        raise StageRefusal('Piece handoff has no valid actual visual placement receipt')
    if track['receipt']['after']['frame_sha256'] != state['observation']['frame_sha256']:
        raise StageRefusal('Piece handoff receipt has different image provenance')
    return _result('carried', state, state['observation'], kind='handoff')


def recover_cleared(ctx, state: StageState) -> StageResult:
    """Clear only an image-confirmed lost/released part, then certify a retry."""
    name=state['piece'];arm=state['arm'];p=_piece_parameters(state)
    # Re-check the physical premise from a fresh image before opening. A graph
    # error or uncertain held object does not authorize this recovery path.
    legal=(state['last_stage'] in ('grasp','transport','release') and state['last_outcome']=='lost')
    legal=legal or (state['last_stage']=='release' and state['last_outcome']=='missed')
    if not legal:
        raise StageRefusal('Only a confirmed loss or released placement miss can request recovery')
    current=_require(c.state(ctx, {name: state['observation']['poses'][name]}), ['board',name])
    tcp,_=c.hand(ctx, arm)
    if c.part_hand_gap(name, current['poses'][name], tcp)<=.045:
        raise StageRefusal('Recovery cannot prove that the target has left the hand')
    c.grip(ctx, arm, p.get('open_width', .055))
    if state['last_outcome']=='lost':
        c.withdraw(ctx, arm)
    ctx.tool('robot.go_home', arm_id=arm)
    _pause(ctx, 25)
    before=_require(c.state(ctx), _scene_names())
    before_hands=[c.hand(ctx, a) for a in (0,1)]
    _pause(ctx, 50)
    after=_require(c.state(ctx), _scene_names())
    if after['timestamp']<=before['timestamp']:
        raise StageRefusal('Recovery requires two observations separated in sensor time')
    movement=_displacement(before, after, list(model.CELLS))
    board_drift=_board_displacement(before,after)
    maximum=max([board_drift,*movement.values()])
    if maximum>=.0015:
        raise StageRefusal('Recovery scene is still moving in RGB')
    all_beads=np.concatenate([_beads(n, after['poses'][n]) for n in model.CELLS])
    after_hands=[c.hand(ctx, a) for a in (0,1)]
    hand_drift=max(float(np.linalg.norm(a[0]-b[0])) for a,b in zip(before_hands,after_hands))
    angular_drift=max(float((Rotation.from_quat(c.qarr(a[1])).inv()*
                             Rotation.from_quat(c.qarr(b[1]))).magnitude())
                      for a,b in zip(before_hands,after_hands))
    if hand_drift>=.0004 or angular_drift>=.015:
        raise StageRefusal('Parked hands have not settled in robot encoder feedback')
    hands=[h[0] for h in after_hands]
    minimum=min(float(np.linalg.norm(all_beads-h, axis=1).min()) for h in hands)
    if minimum<=.055:
        raise StageRefusal('Recovery cannot prove that both parked hands are free of parts')
    plan=state['plan']
    history=[*state['history'], dict(attempt=state['attempt'], grasp_index=plan['index'],
                                   heading=plan['heading'], lean=plan['lean'],
                                   source_height_m=plan['grasp_height_m'])]
    updated={**state, 'attempt': state['attempt']+1, 'history': history, 'prediction': []}
    return _result('cleared', updated, after, kind='recovery', scene_complete=True,
                   stable_released=True, hands_clear=True,
                   minimum_hand_bead_gap_m=minimum, max_displacement_m=maximum,
                   max_hand_drift_m=hand_drift, max_hand_angular_drift_rad=angular_drift,
                   hand_tcp_positions=[h.tolist() for h in hands],
                   prior_hand_tcp_poses=[np.r_[h,c.qarr(q)].tolist() for h,q in before_hands],
                   prior_timestamp=before['timestamp'], prior_frame_sha256=before['frame_sha256'],
                   retry_history_count=len(history))


def finish_staged(ctx, state: StageState) -> StageResult:
    p=_piece_parameters(state)
    for arm in (1,0):
        tcp,q=c.hand(ctx, arm)
        c.grip(ctx, arm, p.get('open_width',.055))
        if tcp[2]<.9:
            c.move(ctx, tcp+[0,0,.05], q, arm)
        ctx.tool('robot.go_home', arm_id=arm)
    _pause(ctx,100)
    names=p.get('diagnostic_subset') or list(model.CELLS)
    before=_require(c.state(ctx), ['board',*names])
    _pause(ctx,50)
    observed=_require(c.state(ctx), ['board',*names])
    displacement=_displacement(before, observed, names)
    errors={n:observed['assessment']['bead_error_m'][n] for n in names}
    maximum=max([_board_displacement(before,observed),*displacement.values()])
    if (observed['timestamp']<=before['timestamp'] or maximum>=.0015
            or any(e is None or e>=.003 for e in errors.values())):
        raise StageRefusal('The repeated final RGB observations are unstable or outside the goal region')
    return _result('complete', state, observed, kind='finish',
                   stable_released=True, all_goal_errors_m=errors,
                   max_displacement_m=maximum,
                   prior_timestamp=before['timestamp'], prior_frame_sha256=before['frame_sha256'])


def stage_checkpoint(kind, world, outputs):
    """Recompute factual guards from sensor records and encoder snapshots.

    Authored booleans and claimed distances are diagnostics, never sufficient
    evidence. External physical truth audits remain separate from this policy.
    """
    evidence=outputs.get('evidence', {});state=outputs.get('state', {})
    observed=world.sim_state.get('vision', {})
    if (evidence.get('kind')!=kind or evidence.get('timestamp')!=observed.get('timestamp')
            or not evidence.get('frame_sha256')
            or evidence.get('frame_sha256')!=observed.get('frame_sha256')):
        return False
    if (state.get('observation',{}).get('timestamp')!=observed.get('timestamp')
            or state.get('observation',{}).get('frame_sha256')!=observed.get('frame_sha256')):
        return False
    hands=world.sim_state.get('robot_tcp_poses', {})
    if set(hands)!={'0','1'}:
        return False
    try:
        tcp=[np.asarray(hands[str(a)],dtype=float) for a in (0,1)]
        if any(h.shape!=(7,) or not np.all(np.isfinite(h)) for h in tcp):
            return False
        poses=observed.get('poses',{});name=state['piece'];arm=state['arm']
        gap=c.part_hand_gap(name,poses[name],tcp[arm][:3]) if name in poses and arm in (0,1) else None
    except (KeyError,ValueError,TypeError):
        return False

    def prior_observation(names):
        for previous in reversed(world.sim_state.get('vision_history', [])):
            if (previous.get('timestamp')==evidence.get('prior_timestamp')
                    and previous.get('frame_sha256')==evidence.get('prior_frame_sha256')
                    and previous.get('timestamp',1e99)<observed['timestamp']
                    and set(names).issubset(previous.get('poses',{}))
                    and set(names).issubset(poses)):
                return previous
        return None

    def measured_movement(previous,names):
        values=list(_displacement(previous,observed,names).values())
        if 'board' in previous['poses'] and 'board' in poses:
            values.append(_board_displacement(previous,observed))
        return max(values) if values else float('inf')

    route=outputs.get('decision')
    if kind=='handoff':
        track=state.get('memory',{}).get('placements',{}).get(name,{})
        receipt=track.get('receipt',{})
        try:
            recomputed=visual_memory.release_receipt(name,receipt['before'],receipt['after'],
                receipt['released_tcp_position'],state['parameters'])
            prior=next((entry for entry in world.sim_state.get('vision_history',[])
                if entry.get('timestamp')==receipt['before']['timestamp']
                and entry.get('frame_sha256')==receipt['before']['frame_sha256']),None)
            current=visual_memory._observation(observed,name)
            prior_record=visual_memory._observation(prior,name) if prior else None
        except (KeyError,ValueError,TypeError,visual_memory.MemoryRefusal):
            return False
        return (route=='carried' and track.get('status')=='measured_now'
                and gap is not None and gap>.055
                and not track.get('invalidation_reason')
                and receipt.get('piece')==name and receipt.get('source')=='rgb_release_pair'
                and receipt==recomputed and prior_record==receipt['before'] and current==receipt['after']
                and receipt.get('after',{}).get('frame_sha256')==observed.get('frame_sha256')
                and receipt.get('after',{}).get('timestamp')==observed.get('timestamp'))
    if kind=='plan':
        return (route=='planned' and set(_scene_names()).issubset(poses)
                and state['plan'].get('body')==poses.get(name)
                and 0<=state['attempt']<state['parameters']['max_attempts_per_piece'])
    if kind in ('grasp','transport'):
        if route=='lost':
            return gap is not None and gap>.045
        return (route=='verified' and 'board' in poses and gap is not None and gap<=.045
                and poses[name][2]-state['plan']['body'][2]>=.025)
    if kind=='release':
        if route=='lost':
            return gap is not None and gap>.045
        previous=prior_observation(['board',name])
        error=observed.get('assessment',{}).get('bead_error_m',{}).get(name)
        return (previous is not None and measured_movement(previous,[name])<.0015
                and gap is not None and gap>.055 and error is not None
                and ((route=='placed' and error<.003) or (route=='missed' and error>=.003)))
    if kind=='placement':
        error=observed.get('assessment',{}).get('bead_error_m',{}).get(state.get('piece'))
        return route=='accepted' and error is not None and error<.003
    if kind=='recovery':
        previous=prior_observation(_scene_names())
        if route!='cleared' or previous is None or measured_movement(previous,list(model.CELLS))>=.0015:
            return False
        bead_centers=np.concatenate([_beads(n,poses[n]) for n in model.CELLS])
        if min(float(np.linalg.norm(bead_centers-h[:3],axis=1).min()) for h in tcp)<=.055:
            return False
        prior_hands=evidence.get('prior_hand_tcp_poses', [])
        if len(prior_hands)!=2:
            return False
        for a,(old,current) in enumerate(zip(prior_hands,tcp)):
            old=np.asarray(old)
            if old.shape!=(7,) or not np.all(np.isfinite(old)):
                return False
            samples=world.sim_state.get('robot_tcp_history',{}).get(str(a),[])
            if not any(evidence['prior_timestamp']<=sample.get('timestamp',-1.)<observed['timestamp']
                       and np.asarray(sample.get('pose',[])).shape==(7,)
                       and np.allclose(sample['pose'],old,atol=1e-9,rtol=0.) for sample in samples):
                return False
            if (np.linalg.norm(old[:3]-current[:3])>=.0004
                    or (Rotation.from_quat(old[3:]).inv()*Rotation.from_quat(current[3:])).magnitude()>=.015):
                return False
        return True
    if kind=='finish':
        names=state['parameters'].get('diagnostic_subset') or list(model.CELLS)
        previous=prior_observation(['board',*names])
        errors=observed.get('assessment',{}).get('bead_error_m',{})
        centers=np.concatenate([_beads(n,poses[n]) for n in names]) if set(names).issubset(poses) else None
        hands_clear=(centers is not None and
                     min(float(np.linalg.norm(centers-h[:3],axis=1).min()) for h in tcp)>.055)
        return (route=='complete' and previous is not None and hands_clear
                and measured_movement(previous,names)<.0015
                and all(errors.get(n) is not None and errors[n]<.003 for n in names))
    return False
