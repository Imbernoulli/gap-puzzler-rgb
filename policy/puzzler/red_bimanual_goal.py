"""Red goal delivery after a physically verified two-hand regrasp chain.

Control inputs are actual RGB, public robot q16/FK, attachment provenance and
public CAD/goals. Independent transport/release/handoff checkpoints decide
physical success. A conditional attachment and a visual UNKNOWN stay explicit.
"""
from copy import deepcopy
import json
import numpy as np
from scipy.spatial.transform import Rotation

from . import model, conditional_scene, visual_memory, vision_controller as c
from . import modular_red_placement as base
from . import red_bimanual_handoff as handoff
from . import unified_contract
from .modular_controller import result, save_observation, scene_belief
from .public_bimanual_material_geometry import PublicBimanualMaterialGeometry
from .red_source_geometry import target_nonfinger_clearance
from .gripper_geometry import clearance, scene_clearance
from .vision_observer import goals
from .vision_stages import StageRefusal

RESERVE=.035
FLOOR=.0005
HOVER=.036
DROP=.005
RELEASE_WIDTH=.090
_GEOMETRY=None


def geometry():
    global _GEOMETRY
    if _GEOMETRY is None:
        _GEOMETRY=PublicBimanualMaterialGeometry(device='cuda:0',minimum_clearance_m=FLOOR)
    return _GEOMETRY


def _arm(state):
    arm=int(state['arm'])
    if state.get('piece')!='red' or arm!=1:
        raise StageRefusal('Red final delivery requires the actual right receiver attachment')
    return arm


def _selected_bead(state):
    candidate=state.get('handoff',{}).get('candidate',{})
    bead=int(candidate.get('bead',4))
    if bead not in range(len(model.local_beads('red'))):
        raise StageRefusal('Final Red contact bead must be a declared public CAD index')
    return bead


def _belief(ctx,state):
    return handoff._belief(ctx,state,_arm(state))


def _bound(belief):
    return handoff._held_envelope(belief)


def _body_at(g,q,arm,belief):
    g.fk(q)
    p,r=g.robot.iks['left' if arm==0 else 'right'].ee_pose(g.scratch)
    return handoff._body(dict(tcp=[*np.asarray(p).tolist(),*np.asarray(r).tolist()]),belief)


def _board_margin(scene):
    margin=float(scene.get('board_for_planning',{}).get('uncertainty_m',0.))
    if not np.isfinite(margin) or margin<0.:
        raise StageRefusal('Planning-board uncertainty must be finite and nonnegative')
    return margin


def _guard_safe(guard,scene,*,material=False):
    if not guard.get('robot_material_safe' if material else 'safe'):
        return False
    needed=FLOOR+_board_margin(scene)
    for row in guard.get('rows',[]):
        groups=row.get('groups',{})
        for name in ('robot_board79',) if material else ('robot_board79','held_envelope_board79'):
            if name not in groups or groups[name]['clearance_m']<needed:
                return False
    return bool(guard.get('rows'))


def _hand_guard(point,rotation,width,scene,board,belief,*,all_target=False):
    """Retain the original local-hand guards with the declared held bound."""
    rr=Rotation.from_quat(c.qarr(rotation)).as_matrix()
    centers,radii=base._target_centers(np.asarray(point),rr,belief)
    obstacles,sizes=conditional_scene.spheres(scene,exclude=('red',))
    other=clearance(point,rr,width,obstacles,radius=sizes)
    fixed=scene_clearance(point,rr,width,[],table_z=float(board[2]-.003),
                          plane_z=float(board[2]-.003),board=board)
    finite=base._board_query().query(board,point,rr,width)
    target=(clearance if all_target else target_nonfinger_clearance)(
        point,rr,width,centers,radius=radii+_bound(belief))
    guards=dict(non_target=other,table_bases_hinge=fixed,finite_board79=finite,
                target_all_material=target) if all_target else dict(
                non_target=other,table_bases_hinge=fixed,finite_board79=finite,target_nonfinger=target)
    name=min(guards,key=lambda key:guards[key]['clearance_m'])
    return dict(safe=bool(guards[name]['clearance_m']>=FLOOR and finite['clearance_m']>=FLOOR+_board_margin(scene)),
        clearance_m=float(guards[name]['clearance_m']),held_envelope_m=_bound(belief),
        groups={key:float(value['clearance_m']) for key,value in guards.items()},
        witness=dict(component=name,detail=guards[name].get('witness')))


def _terminal_external(state,scene,board,belief,body_pose):
    """Original named assembly corridor, with full held uncertainty elsewhere.

    The named public cover-pair set and 1mm nominal penetration limit are the
    original Red contract. An 8mm conditional attachment keeps 8mm separation
    on every unallowed pair; its larger uncertainty does not expand that set.
    Board79 and table keep the entire held envelope even in this corridor.
    """
    g=geometry();bound=_bound(belief)
    body=np.asarray(body_pose);bodyR=Rotation.from_quat(body[3:]).as_matrix()
    local,radii=conditional_scene.local_sphere_cover('red')
    centers=local@bodyR.T+body[:3]
    obstacles,sizes,nominal,support,names=base._named_obstacles(state,scene,board,bodyR)
    br=Rotation.from_quat(board[3:]).as_matrix()
    target=board[:3]+br@np.asarray(state['plan']['target_center'])
    desired=br@np.asarray(state['plan']['target_R'])
    public_red=local@desired.T+target
    public=goals('pyramid',state['parameters']['pyramid_plan'])
    mask=np.zeros_like(support)
    for col,name in enumerate(names):
        piece=name['piece'];item=scene['occupancy'][piece];gc,gr=public[piece]
        goalpose=np.r_[board[:3]+br@gc,Rotation.from_matrix(br@gr).as_quat()]
        valid=(item['source'] in ('conditional_given_stage_prefix_goal_plus_raw',
                'conditional_executed_goal_persistence_plus_current_raw') and item['inflation_m']>=.004)
        observed_option=item['pose_hypotheses'][name['hypothesis']]
        if not valid or visual_memory.shape_displacement(piece,goalpose,observed_option)>.0040001:
            continue
        other,other_radii=conditional_scene.local_sphere_cover(piece);j=name['cover_index']
        otherpoint=(br@gr)@other[j]+goalpose[:3]
        public_gap=np.linalg.norm(public_red-otherpoint,axis=1)-radii-other_radii[j]
        redbeads=model.local_beads('red')@desired.T+target
        otherbeads=model.local_beads(piece)@(br@gr).T+goalpose[:3]
        if not np.any(np.abs(np.linalg.norm(redbeads[:,None]-otherbeads[None],axis=2)-2*model.RADIUS)<1.e-6):
            continue
        delta=public_red-otherpoint;swept=delta.copy()
        swept[:,2]+=np.clip(-delta[:,2],.003,.036)
        swept_gap=np.linalg.norm(swept,axis=1)-radii-other_radii[j]
        mask[:,col]=(public_gap>=-1.e-6)&(swept_gap<=.0105001)
    pair=g._sphere_pair_distances(centers,radii+bound,obstacles,sizes)
    nominal_pair=g._sphere_pair_distances(centers,radii,obstacles,nominal)
    on_axis=(np.linalg.norm(body[:2]-target[:2])<=.001 and
             (Rotation.from_matrix(desired).inv()*Rotation.from_matrix(bodyR)).magnitude()<=.04)
    height=float(body[2]-target[2])
    allowed=mask if on_axis and -.00025<=height<=.0365 else np.zeros_like(mask)
    failed=np.where(allowed,nominal_pair<-.001,pair<FLOOR)
    board_guard=g._sphere_board(board,centers,radii+bound)
    table_clearance=float((centers[:,2]-radii-bound-.750).min())
    safe=bool(not failed.any() and board_guard['clearance_m']>=FLOOR+_board_margin(scene) and table_clearance>=FLOOR)
    return dict(safe=safe,held_envelope_m=bound,assembly_axis=bool(on_axis),assembly_height_m=height,
        declared_assembly_cover_pairs=int(allowed.sum()),
        minimum_unallowed_envelope_m=float(pair[~allowed].min()) if (~allowed).any() else None,
        minimum_allowed_nominal_m=float(nominal_pair[allowed].min()) if allowed.any() else None,
        held_board79=board_guard,held_table_clearance_m=table_clearance,
        scope='Original public named Red cover-pair corridor only; all other pairs retain declared held bound; board/table never exempted')


def _plan(ctx,state,scene,belief,waypoints,*,terminal=False,joint_endpoint=False):
    arm=_arm(state);actual=base.proprio(ctx,arm)
    if joint_endpoint:
        if terminal or len(waypoints)!=1:
            raise StageRefusal('Joint endpoint proposal is a single ordinary full-envelope carry leg only')
        endpoint=waypoints[0]
        path=ctx.tool('puzzle.plan_joint_path',position=endpoint['position'],rotation=endpoint['rotation'],
            arm_id=arm,min_joint_limit_margin_rad=RESERVE)
    else:
        path=ctx.tool('puzzle.plan_pose_sequence',waypoints=waypoints,arm_id=arm,min_joint_limit_margin_rad=RESERVE)
    if not path.get('feasible') or path.get('min_joint_limit_margin_rad',-1.)<RESERVE:
        raise c.PlanningRefusal('Red goal reserved Cartesian path refused '+json.dumps((path.get('errors') or [{}])[-1]))
    g=geometry();q=g.densify_q16_path(handoff._expand(actual['robot_encoder_values'],path['path'],arm))
    bodyposes=np.asarray([_body_at(g,row,arm,belief) for row in q])
    board=np.asarray(scene['board_for_planning']['pose']);contract={arm:[_selected_bead(state)]}
    if terminal:
        delta=bodyposes[-1,:3]-bodyposes[0,:3]
        initialR=Rotation.from_quat(bodyposes[0,3:])
        if delta[2]>=-.001 or np.linalg.norm(delta[:2])>.001:
            raise c.PlanningRefusal('Named assembly corridor requires a strictly vertical downward body path')
        if max((initialR.inv()*Rotation.from_quat(pose[3:])).magnitude() for pose in bodyposes)>.04:
            raise c.PlanningRefusal('Named assembly corridor requires retained constant body orientation')
        guard=g.screen_robot_material_path(q,scene,target='red',target_body_poses=bodyposes,
            finger_target_contract=contract,board_anchor_exit=None)
        outside=[_terminal_external(state,scene,board,belief,pose) for pose in bodyposes]
        if not _guard_safe(guard,scene,material=True) or not all(row['safe'] for row in outside):
            raise c.PlanningRefusal('Red terminal material/held-environment contract refused '+json.dumps(
                dict(material=guard,external=next((r for r in outside if not r['safe']),None)),separators=(',',':')))
    else:
        guard=g.screen_q16_path(q,scene=scene,moving_arm=arm,held_arm=arm,
            held_body_pose=bodyposes[0],object_in_tcp_position=belief['object_in_tcp_position'],
            object_in_tcp_rotation=belief['object_in_tcp_rotation'],target='red',
            finger_target_contract=contract,board_anchor_exit=None,held_envelope_m=_bound(belief))
        outside=[]
        if not _guard_safe(guard,scene):
            raise c.PlanningRefusal('Red normal full robot/held path refused '+json.dumps(guard,separators=(',',':')))
    local=[]
    for sample in q:
        g.fk(sample);point,quat=g.robot.iks['right'].ee_pose(g.scratch)
        row=_hand_guard(np.asarray(point),c.qdict(quat),float(sample[14]-sample[15]),scene,board,belief)
        if not row['safe']:
            raise c.PlanningRefusal('Original local Red hand guard refused '+json.dumps(row))
        local.append(row)
    return dict(arm=arm,native_path=path,q16_path=q.tolist(),guard=guard,held_external=outside,
        held_envelope_m=_bound(belief),local_hand_guards=local,endpoint=waypoints[-1],
        source_actual=actual,terminal_contract=terminal,
        planning_space='live_IK_endpoint_joint_interpolation' if joint_endpoint else 'live_Cartesian_pose_sequence')


def _drive(ctx,state,motion,*,intermediate=False,scene=None,belief=None):
    stable=None
    if intermediate:
        if scene is None or belief is None or motion.get('terminal_contract'):
            raise StageRefusal('Stable actual adaptation requires an ordinary screened intermediate and its current scene/belief')
        stable=dict(scene=scene,belief=belief,held_arm=_arm(state),contract={_arm(state):[_selected_bead(state)]},
            purpose='intermediate_goal_carry')
    execution=handoff._execute(ctx,state,motion,retained_arm=_arm(state),stable_actual_handoff=stable)
    adaptation=execution.get('stable_actual_endpoint')
    if adaptation is not None and intermediate:
        local=[]
        for actual in(adaptation['actual_initial_robot'],adaptation['actual_robot']):
            row=_hand_guard(np.asarray(actual['tcp'][:3]),c.qdict(actual['tcp'][3:]),actual['width_m'],
                scene,np.asarray(scene['board_for_planning']['pose']),belief)
            if not row['safe']:
                raise StageRefusal('Stable intermediate actual endpoint failed original local hand guard '+json.dumps(row))
            local.append(row)
        adaptation['original_local_hand_guards']=local
        adaptation['fresh_RGB_required_before_next_motion']=True
    return execution


def _refresh(ctx,state,label,*,optimizer_belief=None):
    attempts=[]
    for attempt in range(2):
        state,belief,record=handoff._fresh_sender(ctx,state,_arm(state),optimizer_belief=optimizer_belief)
        attempts.append(record)
        if record.get('status')=='actual_RGB_q16_rebound':break
    from .red_computed_carry import _repair_prefix_rgb
    # The existing skill detects both a raw-present contradiction and a
    # previously invalidated missing prefix. No conflict means no sensor call.
    state,prefix_reacquisition=_repair_prefix_rgb(ctx,state,label=label+'_prefix_reacquisition')
    if prefix_reacquisition is not None:
        state,belief,rebound=handoff._fresh_sender(ctx,state,_arm(state),optimizer_belief=belief)
        attempts.append(rebound)
    try:
        state,scene,scene_refresh=handoff.scene_for_motion(ctx,state,belief)
    except StageRefusal as error:
        if (prefix_reacquisition is not None or
                not str(error).startswith('Invalidated prefix requires current reacquisition: ')):raise
        # Board reacquisition may introduce a new actual raw contradiction.
        # Permit one bounded prefix repair, never a cleared ledger or a loop.
        state,prefix_reacquisition=_repair_prefix_rgb(ctx,state,label=label+'_prefix_reacquisition')
        if prefix_reacquisition is None:raise
        state,belief,rebound=handoff._fresh_sender(ctx,state,_arm(state),optimizer_belief=belief)
        attempts.append(rebound)
        state,scene,scene_refresh=handoff.scene_for_motion(ctx,state,belief)
    state['planning_scene']=scene
    return state,belief,scene,dict(label=label,attempts=attempts,held_envelope_m=_bound(belief),
        scene_refresh=scene_refresh,prefix_reacquisition=prefix_reacquisition)


def _goal(state,scene):
    board=np.asarray(scene['board_for_planning']['pose']);br=Rotation.from_quat(board[3:]).as_matrix()
    center,rr=goals('pyramid',state['parameters']['pyramid_plan'])['red']
    relative=np.asarray(state.get('plan',{}).get('target_R',rr))
    return board,board[:3]+br@center,br@relative,center,relative


def _body_waypoint(center,rotation,belief):
    rr=rotation@np.asarray(belief['object_in_tcp_rotation']).T
    point=np.asarray(center)-rr@np.asarray(belief['object_in_tcp_position'])
    return handoff._waypoint(point,rr)


def _hover_error(state,scene,belief,actual):
    board,target,desired,_,_=_goal(state,scene)
    target_hover=target+[0,0,HOVER]
    body=handoff._body(actual,belief)
    reference=np.r_[target_hover,Rotation.from_matrix(desired).as_quat()]
    bead_error=visual_memory.shape_displacement('red',reference,body)
    rotation_error=float((Rotation.from_matrix(desired).inv()*Rotation.from_quat(body[3:])).magnitude())
    return dict(bead_error_m=float(bead_error),body_translation_error_m=float(np.linalg.norm(body[:3]-target_hover)),
        lateral_error_m=float(np.linalg.norm(body[:2]-target_hover[:2])),rotation_error_rad=rotation_error,
        actual_body_estimate=body.tolist(),true_goal_hover_pose=reference.tolist(),
        estimated_from='Fresh actual RGB/q16 attachment propagated through current robot FK',
        ready_for_original_terminal=bool(np.linalg.norm(body[:2]-target_hover[:2])<=.001 and rotation_error<=.03),
        transport_six_mm_predicate=bool(bead_error<=.006))


def _latest_measured(refresh):
    attempts=refresh.get('attempts',[])
    return bool(attempts and attempts[-1].get('status')=='actual_RGB_q16_rebound')


def _align_goal_hover(ctx,state,belief,scene,initial_refresh,*,label,maximum_corrections=3,
                      require_terminal_axis=False):
    """Bounded actual-RGB feedback, with every correction explicitly screened.

    Refreshing an attachment changes the world object estimate; it does not
    itself move the object to the public goal. Exact-target replanning handles
    a changed attachment first. Repeated residuals permit only a small explicit
    command compensation, never a hidden motor correction or a changed goal.
    """
    if maximum_corrections!=3:
        raise StageRefusal('This reviewed goal feedback contract permits exactly three bounded corrections')
    records=[];current_refresh=initial_refresh;previous=None;corrections=[]
    for attempt in range(maximum_corrections+1):
        if not _latest_measured(current_refresh):
            rec=dict(attempt=attempt,status='RGB_UNKNOWN',current_target_measured=False,
                held_envelope_m=_bound(belief),scope='No visual error invented; original independent physical checkpoint still required')
            records.append(rec)
            return state,belief,scene,dict(label=label,rounds=records,corrections=corrections,
                status='visual_unknown_no_claim',current_target_measured=False,latest_refresh=current_refresh)
        actual=base.proprio(ctx,_arm(state));error=_hover_error(state,scene,belief,actual)
        records.append(dict(attempt=attempt,status='actual_RGB_q16_goal_error',**error))
        if error['transport_six_mm_predicate'] and error['ready_for_original_terminal']:
            return state,belief,scene,dict(label=label,rounds=records,corrections=corrections,
                status='actual_RGB_goal_hover_aligned',current_target_measured=True)
        if attempt==maximum_corrections:
            if not error['transport_six_mm_predicate'] or require_terminal_axis:
                raise StageRefusal('Bounded actual RGB goal-hover correction exhausted '+json.dumps(dict(
                    label=label,rounds=records,corrections=corrections),separators=(',',':')))
            return state,belief,scene,dict(label=label,rounds=records,corrections=corrections,
                status='transport_six_mm_estimate_only_terminal_axis_not_yet_aligned',current_target_measured=True)
        board,target,desired,_,_=_goal(state,scene);true_center=target+[0,0,HOVER]
        body=np.asarray(error['actual_body_estimate']);measuredR=Rotation.from_quat(body[3:]).as_matrix()
        # First exploit the newly measured attachment, without an arbitrary
        # compensating offset. Only a residual that stopped improving receives
        # explicit command feedback on the next bounded, fully screened path.
        stalled=(previous is not None and error['bead_error_m']>=previous*.85)
        translation=np.zeros(3);angle=np.zeros(3)
        if stalled:
            translation=true_center-body[:3]
            norm=float(np.linalg.norm(translation))
            if norm>.006:translation*=.006/norm
            angle=Rotation.from_matrix(desired@measuredR.T).as_rotvec()
            norm=float(np.linalg.norm(angle))
            if norm>.06:angle*=.06/norm
        command_center=true_center+translation
        command_rotation=Rotation.from_rotvec(angle).as_matrix()@desired
        proposal=dict(strategy='explicit_bounded_command_feedback' if stalled else 'new_actual_attachment_exact_goal',
            true_goal_center=true_center.tolist(),true_goal_rotation=desired.tolist(),
            command_center=command_center.tolist(),command_rotation=command_rotation.tolist(),
            translation_compensation_m=translation.tolist(),rotation_compensation_rotvec_rad=angle.tolist(),
            maximum_translation_compensation_m=.006,maximum_rotation_compensation_rad=.06,
            previous_bead_error_m=previous,actual_bead_error_m=error['bead_error_m'],
            goal_is_unchanged=True,proposal_is_measurement=False)
        waypoint=_body_waypoint(command_center,command_rotation,belief)
        motion=_plan(ctx,state,scene,belief,[waypoint],terminal=False)
        execution=_drive(ctx,state,motion)
        corrections.append(dict(proposal=proposal,execution=execution))
        previous=error['bead_error_m']
        state,belief,scene,current_refresh=_refresh(ctx,state,label+'_after_correction_'+str(attempt+1),optimizer_belief=belief)
    raise AssertionError('Unreachable bounded goal feedback exit')


def _carry_goal_endpoint_candidates(ctx,state,scene,belief):
    """Qualify safe hover transport; keep future drop qualification explicit.

    Every executed transport sample still needs the ordinary full robot/held
    path guards. A current attachment's future drop failure does not predict
    the attachment at a later, freshly observed hover. It never authorizes
    descent, opening, release or any placement checkpoint.
    """
    arm=_arm(state)
    board=np.asarray(scene['board_for_planning']['pose']);br=Rotation.from_quat(board[3:]).as_matrix()
    center,goalR=goals('pyramid',state['parameters']['pyramid_plan'])['red']
    target=board[:3]+br@center
    attempts=[];accepted=[]
    actual=base.proprio(ctx,arm)
    for index,sym in enumerate(base.public_symmetries('red')):
        desired=br@goalR@sym
        trial_state={**state,'plan':{**state['plan'],'target_center':center.tolist(),'target_R':(goalR@sym).tolist()}}
        checks=[];hand_checks=[]
        for height in (HOVER,DROP):
            waypoint=_body_waypoint(target+[0,0,height],desired,belief)
            ik=ctx.tool('puzzle.plan_pose',position=waypoint['position'],rotation=waypoint['rotation'],arm_id=arm,
                        min_joint_limit_margin_rad=RESERVE)
            checks.append(dict(height_m=height,native_IK=ik))
            if (ik.get('position_error_m',1.)>.002 or ik.get('rotation_error_rad',1.)>.03 or
                    ik.get('joint_limit_margin_rad',-1.)<RESERVE):break
            widths=(actual['width_m'],.055,RELEASE_WIDTH) if height==DROP else (actual['width_m'],)
            hand_checks.extend(_hand_guard(np.asarray(waypoint['position']),waypoint['rotation'],width,
                                           scene,board,belief) for width in widths)
        row=dict(symmetry_index=index,public_symmetry=sym.tolist(),target_R=(goalR@sym).tolist(),
                 native_endpoint_IK=checks,local_endpoint_hand=hand_checks)
        def valid_ik(check):return (check['native_IK'].get('position_error_m',1.)<=.002 and
              check['native_IK'].get('rotation_error_rad',1.)<=.03 and
              check['native_IK'].get('joint_limit_margin_rad',-1.)>=RESERVE)
        hover_good=bool(checks and checks[0]['height_m']==HOVER and valid_ik(checks[0]) and
            hand_checks and hand_checks[0]['safe'])
        if not hover_good:
            row['refusal']='native_hover_IK_or_original_loaded_hover_hand_guard'
            attempts.append(row);continue
        drop_good=bool(len(checks)==2 and valid_ik(checks[1]) and len(hand_checks)==4 and
            all(check['safe'] for check in hand_checks[1:]))
        proposed_body=np.r_[target+[0,0,DROP],Rotation.from_matrix(desired).as_quat()]
        terminal=_terminal_external(trial_state,scene,board,belief,proposed_body)
        row['public_drop5_external_contract']=terminal
        row['current_future_drop5_endpoint_qualified']=bool(drop_good and terminal['safe'])
        row['hover_endpoint_qualified']=True
        row['safe_hover_only_not_release_qualified']=not row['current_future_drop5_endpoint_qualified']
        if not row['current_future_drop5_endpoint_qualified']:
            row['future_release_requires_fresh_RGB_bound']=dict(
                current_held_envelope_m=_bound(belief),
                current_declared_terminal_contract_passed=False,
                current_drop_native_IK_and_actualjaw_to90_qualified=drop_good,
                scope='Only ordinary full-envelope safe hover transport may proceed. Fresh actual RGB/q16 and every unchanged drop5 loaded/jaw-sweep/material/external/source gate must pass before descent or opening.')
        row['minimum_endpoint_reserve_rad']=float(checks[0]['native_IK']['joint_limit_margin_rad'])
        accepted.append(row);attempts.append(row)
    if not accepted:
        raise c.PlanningRefusal('No actual-attachment strict public-goal hover candidate '+json.dumps(attempts))
    selected=max(accepted,key=lambda row:row['minimum_endpoint_reserve_rad'])
    return selected,attempts


def carry(ctx,state):
    state=deepcopy(state);arm=_arm(state)
    if (state.get('handoff',{}).get('stage')!='verified' or
            int(state.get('handoff',{}).get('hop',0))!=2):
        raise StageRefusal('Goal carry requires preceding typed receiver binding and its independent physical checkpoint')
    state,belief,scene,refresh=_refresh(ctx,state,'goal_entry',optimizer_belief=state.get('handoff',{}).get('receiver_attachment',{}).get('belief'))
    selected,attempts=_carry_goal_endpoint_candidates(ctx,state,scene,belief)
    center,_=goals('pyramid',state['parameters']['pyramid_plan'])['red']
    p={**state['parameters'],'transport':HOVER,'release_height':DROP,'release_open_width':RELEASE_WIDTH}
    state.update(parameters=p,plan={**state['plan'],'target_center':center.tolist(),'target_R':selected['target_R'],
        'red_target_selection':selected,'red_target_candidates':attempts})
    board,target,desired,_,_=_goal(state,scene)
    br=Rotation.from_quat(board[3:]).as_matrix()
    actual=base.proprio(ctx,arm);current=handoff._body(actual,belief);currentR=Rotation.from_quat(current[3:]).as_matrix()
    motions=[];rebindings=[refresh];refusals=[];outside=None;route=None;pretranslation=None
    # Separate an orientation change from the neighboring prefix first. The
    # proposal is a short current-attitude translation in the measured board
    # frame, not a stored world waypoint. Its full path may be refused before
    # motion, in which case the other current-state route families remain.
    try:
        shifted=current[:3]+br@np.array([0.,-.040,0.])
        pretranslation=_plan(ctx,state,scene,belief,[_body_waypoint(shifted,currentR,belief)])
    except c.PlanningRefusal as error:
        refusals.append(dict(route='current_attitude_board_negativeY40mm',refusal=str(error)))
    if pretranslation is not None:
        motions.append(_drive(ctx,state,pretranslation,intermediate=True,scene=scene,belief=belief))
        state,belief,scene,r=_refresh(ctx,state,'after_current_attitude_pretranslation',optimizer_belief=belief)
        rebindings.append(r)
        if motions[-1].get('stable_actual_endpoint')is not None and not _latest_measured(r):
            raise StageRefusal('Adapted current-attitude pretranslation requires new actual Red RGB/q16')
        board,target,desired,_,_=_goal(state,scene);br=Rotation.from_quat(board[3:]).as_matrix()
        actual=base.proprio(ctx,arm);current=handoff._body(actual,belief);currentR=Rotation.from_quat(current[3:]).as_matrix()
    # A continuous Cartesian SLERP can stay on a singular wrist branch even
    # when both endpoints are reachable.  Native IK restarts and joint-space
    # interpolation offer another online path proposal.  Every dense public
    # robot/held sample still passes the exact same ordinary full guards.
    for side,extra in ((-.10,.085),(-.10,.075),(-.12,.10),(-.14,.12)):
        outer=target+br@np.array([0.,side,0.])+np.array([0.,0.,extra])
        try:
            outside=_plan(ctx,state,scene,belief,[_body_waypoint(outer,desired,belief)],joint_endpoint=True)
            route=dict(kind='live_joint_endpoint_outside_then_Cartesian_hover',board_lateral_offset_m=side,
                world_vertical_offset_m=extra,saved_world_or_joint_path=False)
            break
        except c.PlanningRefusal as error:
            refusals.append(dict(route='joint_endpoint_outside',board_lateral_offset_m=side,
                world_vertical_offset_m=extra,refusal=str(error)))
    # Keep the previously declared geometric family available if every new
    # joint proposal is refused before any movement.
    if outside is None:
        for side,extra in ((-.13,.10),(.13,.10),(-.17,.14),(.17,.14),(-.10,.18)):
            high=max(float(current[2]),float(target[2])+extra)
            raised=current[:3].copy();raised[2]=high
            outer=target+br@np.array([0.,side,0.]);outer[2]=high
            ways=[_body_waypoint(raised,currentR,belief),_body_waypoint(outer,currentR,belief)]
            try:
                outside=_plan(ctx,state,scene,belief,ways)
                route=dict(kind='Cartesian_raise_outside_then_rotate',board_lateral_offset_m=side,
                    minimum_world_height_above_goal_m=extra,saved_world_or_joint_path=False)
                break
            except c.PlanningRefusal as error:refusals.append(dict(route='Cartesian_outside',refusal=str(error)))
    if outside is None:
        raise c.PlanningRefusal('Red raised outside route refused '+json.dumps(refusals))
    motions.append(_drive(ctx,state,outside,intermediate=True,scene=scene,belief=belief))
    state,belief,scene,r=_refresh(ctx,state,'after_outside_translation',optimizer_belief=belief);rebindings.append(r)
    if motions[-1].get('stable_actual_endpoint') is not None and not _latest_measured(r):
        raise StageRefusal('Adapted intermediate goal endpoint requires new actual Red RGB/q16 before subsequent motion')
    board,target,desired,_,_=_goal(state,scene)
    if route['kind']=='live_joint_endpoint_outside_then_Cartesian_hover':
        try:
            above=target+[0,0,max(.085,HOVER+.045)]
            ingress=_plan(ctx,state,scene,belief,[_body_waypoint(above,desired,belief),
                _body_waypoint(target+[0,0,HOVER],desired,belief)])
        except c.PlanningRefusal as error:
            # Keep the attained outside pose and its newly measured attachment.
            # A failed new path does not restore the source or replay a prefix.
            refusals.append(dict(route='Cartesian_hover_from_attained_outside',refusal=str(error)))
            ingress=_plan(ctx,state,scene,belief,[_body_waypoint(target+[0,0,HOVER],desired,belief)],joint_endpoint=True)
    else:
        actual=base.proprio(ctx,arm);current=handoff._body(actual,belief)
        rotate=_plan(ctx,state,scene,belief,[_body_waypoint(current[:3],desired,belief)])
        motions.append(_drive(ctx,state,rotate,intermediate=True,scene=scene,belief=belief))
        state,belief,scene,r=_refresh(ctx,state,'after_outside_rotation',optimizer_belief=belief);rebindings.append(r)
        if motions[-1].get('stable_actual_endpoint') is not None and not _latest_measured(r):
            raise StageRefusal('Adapted intermediate rotation requires new actual Red RGB/q16 before subsequent motion')
        board,target,desired,_,_=_goal(state,scene)
        above=target+[0,0,max(.085,HOVER+.045)]
        ingress=_plan(ctx,state,scene,belief,[_body_waypoint(above,desired,belief),
                                          _body_waypoint(target+[0,0,HOVER],desired,belief)])
    motions.append(_drive(ctx,state,ingress))
    state,belief,scene,r=_refresh(ctx,state,'at_public_goal_hover',optimizer_belief=belief);rebindings.append(r)
    state,belief,scene,hover_feedback=_align_goal_hover(ctx,state,belief,scene,r,label='final_transport_hover')
    state['transport_record']=dict(contract='Red_actual_receiver_attachment_goal_transport_v2',arm=arm,
        current_attachment=belief,held_envelope_m=_bound(belief),selected_public_goal=selected,
        public_transport_hover_m=HOVER,motions=motions,actual_robot=base.proprio(ctx,arm),route_proposal=route,
        current_attitude_pretranslation=dict(board_relative_m=[0.,-.040,0.],executed=pretranslation is not None),
        source_rebindings=rebindings,refusals=refusals,goal_hover_feedback=hover_feedback,desired_recipe_is_not_measured_attachment=True)
    return result('carried',state,state['observation'],'transport',transport_record=state['transport_record'])


def _opening_at_q(state,scene,belief,q):
    arm=_arm(state);g=geometry()
    q=np.asarray(q);body=_body_at(g,q,arm,belief)
    end=q.copy();end[14]=RELEASE_WIDTH/2.;end[15]=-RELEASE_WIDTH/2.
    dense=g.densify_q16_path([q,end])
    poses=np.tile(body,(len(dense),1))
    material=g.screen_robot_material_path(dense,scene,target='red',target_body_poses=poses,
        finger_target_contract={arm:[_selected_bead(state)]},board_anchor_exit=None)
    if not _guard_safe(material,scene,material=True):
        raise c.PlanningRefusal('Actual reached drop5-to90mm solid-jaw sweep refused '+json.dumps(
            dict(material=material,
                external=dict(status='UNKNOWN',reason='Not evaluated after original material gate refusal'),
                local=dict(status='UNKNOWN',reason='Not evaluated after original material gate refusal')),
            separators=(',',':')))
    # Jaw samples change finger widths only. The existing external predicate
    # depends on this one fixed body, scene, board and held bound, not jaw q.
    # Reuse that exact result within this invocation; never cache across calls.
    external_body=_terminal_external(state,scene,np.asarray(scene['board_for_planning']['pose']),belief,body)
    external=[deepcopy(external_body) for _ in poses]
    local=[]
    for row in dense:
        g.fk(row);point,quat=g.robot.iks['right'].ee_pose(g.scratch)
        local.append(_hand_guard(point,c.qdict(quat),float(row[14]-row[15]),scene,
            np.asarray(scene['board_for_planning']['pose']),belief))
    if not all(row['safe'] for row in external) or not all(row['safe'] for row in local):
        raise c.PlanningRefusal('Actual reached drop5-to90mm solid-jaw sweep refused '+json.dumps(
            dict(material=material,external=next((r for r in external if not r['safe']),None),
                 local=next((r for r in local if not r['safe']),None)),separators=(',',':')))
    return dict(q16_before_open=q.tolist(),body_before_open=body.tolist(),q16_jaw_sweep=dense.tolist(),
        material=material,held_external=external,local_hand_guards=local,
        held_envelope_m=_bound(belief),width_sweep_m=[float(q[14]-q[15]),RELEASE_WIDTH])


def _opening_sweep(ctx,state,scene,belief):
    actual=base.proprio(ctx,_arm(state))
    record=_opening_at_q(state,scene,belief,actual['robot_encoder_values'])
    record['actual_preopen']=actual
    return record


def _released_robot_guard(g,q,body_pose,uncertainty):
    """All robot material versus a dated released-body uncertainty envelope."""
    local,radii=conditional_scene.local_sphere_cover('red');pose=np.asarray(body_pose)
    centers=Rotation.from_quat(pose[3:]).apply(local)+pose[:3]
    g.fk(q);distances=g._robot_sphere_distances(centers,radii+uncertainty)
    i,j=np.unravel_index(int(np.argmin(distances)),distances.shape)
    return dict(safe=bool(distances[i,j]>=FLOOR),clearance_m=float(distances[i,j]),
        released_body_envelope_m=float(uncertainty),witness=dict(
        robot_collider=g.shape_labels[int(g.physical_ids[i])],target_sphere=int(j)))


def _empty_plan(ctx,state,scene,arm,waypoints,released_pose,released_bound,*,joint_endpoint=False):
    actual=base.proprio(ctx,arm)
    if joint_endpoint:
        if len(waypoints)!=1:raise StageRefusal('Empty-hand joint endpoint proposal requires one current public endpoint')
        path=ctx.tool('puzzle.plan_joint_path',position=waypoints[0]['position'],rotation=waypoints[0]['rotation'],
            arm_id=arm,min_joint_limit_margin_rad=RESERVE)
    else:
        path=ctx.tool('puzzle.plan_pose_sequence',waypoints=waypoints,arm_id=arm,min_joint_limit_margin_rad=RESERVE)
    if not path.get('feasible') or path.get('min_joint_limit_margin_rad',-1.)<RESERVE:
        raise c.PlanningRefusal('Released Red empty-hand native path refused '+json.dumps((path.get('errors') or [{}])[-1]))
    g=geometry();q=g.densify_q16_path(handoff._expand(actual['robot_encoder_values'],path['path'],arm))
    material=g.screen_robot_material_path(q,scene,target='red',target_body_poses=released_pose,
                                         finger_target_contract={},board_anchor_exit=None)
    target=[_released_robot_guard(g,row,released_pose,released_bound) for row in q]
    if not _guard_safe(material,scene,material=True) or not all(row['safe'] for row in target):
        raise c.PlanningRefusal('Released Red full robot/uncertain-target withdrawal refused '+json.dumps(
            dict(material=material,target=next((r for r in target if not r['safe']),None)),separators=(',',':')))
    return dict(arm=arm,native_path=path,q16_path=q.tolist(),endpoint=waypoints[-1],guard=material,
        released_target_guards=target,released_body_pose_hypothesis=np.asarray(released_pose).tolist(),
        released_body_envelope_m=released_bound,
        planning_space='live_IK_endpoint_joint_interpolation' if joint_endpoint else 'live_Cartesian_pose_sequence',
        scope='Empty-hand path; fixed dated release hypothesis is conditional, never a new measured target pose')


def _released_withdrawal_plan(ctx,state,scene,arm,released_pose,released_bound,current,rotation):
    """Choose an online vertical withdrawal before issuing any motor command.

    The original 60mm proposal can cross a wrist reserve boundary. A 40mm
    proposal retains the same attitude and all original material, released
    body, board and reserve gates. A planning refusal moves no hand.
    """
    refusals=[]
    for distance in (.060,.040):
        try:
            motion=_empty_plan(ctx,state,scene,arm,
                [handoff._waypoint(np.asarray(current)+[0.,0.,distance],rotation)],
                released_pose,released_bound)
        except c.PlanningRefusal as error:
            refusals.append(dict(vertical_distance_m=distance,refusal=str(error)))
            continue
        motion['withdrawal_proposal']=dict(vertical_distance_m=distance,
            planning_refusals=refusals,source='Current actual open-hand TCP and public vertical direction',
            saved_world_or_joint_path=False,execution_not_yet_issued=True)
        return motion
    raise c.PlanningRefusal('No original-gate released Red vertical withdrawal '+json.dumps(refusals))


def _empty_execute(ctx,motion):
    arm=motion['arm'];path=np.asarray(motion['q16_path']);off=8*arm
    before=base.proprio(ctx,arm)
    if before['width_m']<=.045 or np.max(np.abs(np.asarray(before['robot_encoder_values'])-path[0]))>.002:
        raise StageRefusal('Released empty-hand actual q16/open-jaw no longer matches screened path')
    runs=[]
    for begin in range(1,len(path),8):
        if base.proprio(ctx,arm)['width_m']<=.045:
            raise StageRefusal('Released hand unexpectedly closed during empty withdrawal')
        commands=[dict(positions=row[off:off+6].tolist()) for row in path[begin:begin+8]]
        moved=ctx.tool('robot.execute_trajectory',trajectory=dict(waypoints=commands),arm_id=arm,max_steps_per_waypoint=3)
        if moved.get('status')!='ok':raise StageRefusal('Native empty-hand Red delivery trajectory failed')
        runs.append(moved)
    endpoint=motion['endpoint']
    c.await_motion(ctx,np.asarray(endpoint['position']),endpoint['rotation'],arm,False,.0025)
    return dict(screen=motion,motor_runs=runs,actual_after=base.proprio(ctx,arm))


def _home(ctx,state,scene,arm,released_pose,released_bound):
    actual=base.proprio(ctx,arm);q=np.asarray(actual['robot_encoder_values']);g=geometry()
    ik=g.robot.iks['left' if arm==0 else 'right'];home=q.copy();home[8*arm:8*arm+6]=ik._home_q
    g.fk(home);point,quat=ik.ee_pose(g.scratch)
    endpoint=dict(position=np.asarray(point).tolist(),rotation=c.qdict(quat))
    current=np.asarray(actual['tcp'][:3]);rotation=Rotation.from_quat(actual['tcp'][3:]).as_matrix()
    proposals=[[endpoint],
        [handoff._waypoint(current+[0,0,.060],rotation),endpoint],
        [handoff._waypoint(current+[0,0,.100],rotation),endpoint]]
    refusals=[]
    for ways in proposals:
        try:
            motion=_empty_plan(ctx,state,scene,arm,ways,released_pose,released_bound)
            return dict(execution=_empty_execute(ctx,motion),refusals=refusals,
                        home_source='Public robot neutral home joint coordinates, recomputed native path from actual q16')
        except c.PlanningRefusal as error:refusals.append(str(error))
    # A Cartesian line to the public home TCP may graze a same-arm link even
    # when the home joint configuration and a joint-space path are clear.
    # Solve that same endpoint online and keep all material/released-body gates.
    try:
        motion=_empty_plan(ctx,state,scene,arm,[endpoint],released_pose,released_bound,joint_endpoint=True)
        return dict(execution=_empty_execute(ctx,motion),refusals=refusals,
            home_source='Public robot home TCP, current native IK endpoint and dense joint interpolation',
            planning_space='live_IK_endpoint_joint_interpolation')
    except c.PlanningRefusal as error:refusals.append(str(error))
    raise c.PlanningRefusal('No fully screened released-Red home path '+json.dumps(refusals))


def _record_release_pair(ctx,state,target,desired):
    hint={'red':np.r_[target,Rotation.from_matrix(desired).as_quat()].tolist()}
    first=c.state(ctx,hint);state=save_observation(state,first)
    ctx.tool('robot.wait_steps',steps=50)
    second=c.state(ctx,{'red':first['poses']['red']} if 'red' in first.get('poses',{}) else hint)
    state=save_observation(state,second);receipt=None;refusal=None;memory=state['memory']
    try:
        receipt=visual_memory.release_receipt('red',first,second,
            np.asarray(base.proprio(ctx,_arm(state))['tcp'][:3]),state['parameters'])
        memory=visual_memory.mint_placement(visual_memory.observe(memory,second,state['parameters']),
                                          receipt,second,state['parameters'])
    except visual_memory.MemoryRefusal as error:refusal=str(error)
    return state,first,second,receipt,refusal,memory


def _release_height_feedback(ctx,state,scene,belief):
    """Derive a vertical command proposal from the current observed hover.

    The public assembly goal and nominal release height stay explicit. The
    current RGB-owned body and actual q16 provide a bounded vertical command
    correction, never a measurement of a future descent or release outcome.
    """
    if _bound(belief)>.0060000001:
        raise StageRefusal('Release height feedback requires the original owned6mm attachment')
    board,target,desired,_,_=_goal(state,scene)
    actual=base.proprio(ctx,_arm(state));body=handoff._body(actual,belief)
    height=float(body[2]-target[2])
    residual=float(HOVER-height)
    compensation=float(np.clip(residual,-.006,.006))
    command_height=float(DROP+compensation)
    if not FLOOR<command_height<HOVER-.001:
        raise StageRefusal('Bounded current-hover release command leaves the original vertical corridor')
    return dict(nominal_release_height_m=DROP,reference_hover_height_m=HOVER,
        current_body_height_estimate_m=height,current_body_pose_estimate=body.tolist(),
        current_actual_robot=actual,observed_hover_height_residual_m=residual,
        state_estimate_kind='Authenticated visual source propagated through actual current encoders',
        vertical_command_compensation_m=compensation,maximum_compensation_m=.006,
        command_release_height_m=command_height,public_goal_unchanged=True,
        future_endpoint_or_release_measured=False,
        scope='Current authenticated RGB-owned attachment plus actual public q16/FK; explicit screened vertical proposal only')


def _preview_release_height(ctx,state,scene,belief,height):
    """Pure online native endpoint and the identical solid-jaw geometry gate."""
    arm=_arm(state);actual=base.proprio(ctx,arm)
    board,target,desired,_,_=_goal(state,scene)
    waypoint=_body_waypoint(target+[0,0,height],desired,belief)
    ik=ctx.tool('puzzle.plan_pose',position=waypoint['position'],rotation=waypoint['rotation'],
        arm_id=arm,min_joint_limit_margin_rad=RESERVE)
    metrics=np.asarray([ik.get('position_error_m',float('nan')),
        ik.get('rotation_error_rad',float('nan')),ik.get('joint_limit_margin_rad',float('nan'))],float)
    if (not np.isfinite(metrics).all() or
            ik.get('position_error_m',1.)>.002 or ik.get('rotation_error_rad',1.)>.03 or
            ik.get('joint_limit_margin_rad',-1.)<RESERVE):
        raise c.PlanningRefusal('Future release-height native endpoint violates original reserve/accuracy')
    q=np.asarray(actual['robot_encoder_values'],float).copy();joints=np.asarray(ik.get('joints',[]),float)
    if joints.shape!=(6,) or not np.isfinite(joints).all():
        raise c.PlanningRefusal('Future release-height endpoint requires six real native IK joints')
    q[8*arm:8*arm+6]=joints
    sweep=_opening_at_q(state,scene,belief,q)
    return dict(requested_body_height_m=float(height),native_IK=ik,q16_endpoint=q.tolist(),
        achieved_body_pose=sweep['body_before_open'],jaw_samples=len(sweep['q16_jaw_sweep']),
        minimum_local_hand_m=min(row['clearance_m'] for row in sweep['local_hand_guards']),
        minimum_unallowed_held_external_m=min(row['minimum_unallowed_envelope_m'] for row in sweep['held_external']),
        original_material_external_local_opening_passed=True,proposal_only_no_motor=True)


def _select_release_height(ctx,state,scene,belief,feedback):
    """Choose the first current-CAD opening-safe vertical release proposal.

    Both the command and a 2.5mm lower tracking endpoint use native FK and
    exactly the same ordinary0.5mm/held6mm/full79/prefix gates as actual open.
    Every executed descent and actual reached opening is screened again.
    """
    baseline=max(DROP,float(feedback['command_release_height_m']))
    candidates=[baseline]
    candidates.extend(float(v)/2000. for v in range(int(np.ceil(baseline*2000.)),23)
        if float(v)/2000.>baseline+1.e-9)
    attempts=[]
    for height in candidates:
        if height>DROP+.0060000001:continue
        row=dict(command_release_height_m=height,lower_tracking_height_m=height-.0025)
        try:
            # The lower endpoint is the restrictive jaw/board case. Qualify
            # it first; a refused endpoint cannot authorize this candidate.
            row['lower_tracking_opening']=_preview_release_height(ctx,state,scene,belief,height-.0025)
            row['command_opening']=_preview_release_height(ctx,state,scene,belief,height)
        except c.PlanningRefusal as error:
            row.update(qualified=False,refusal=str(error)[:1600]);attempts.append(row);continue
        row['qualified']=True;attempts.append(row)
        return height,dict(selected=row,candidates=attempts,maximum_command_height_m=DROP+.006,
            tracking_vertical_shortfall_m=.0025,public_goal_unchanged=True,no_motor_in_selection=True)
    raise c.PlanningRefusal('No bounded current-CAD release height has safe original full opening and lower tracking endpoint '+json.dumps(attempts))


def _await_release_width(ctx,arm):
    """Wait only for the same registered90mm command; never reissue a grip.

    Actual encoders must cross the unchanged89.5mm gate within150steps.
    The opened source is already revoked. No held source is queried or reused.
    """
    samples=[];waited=0
    while True:
        actual=base.proprio(ctx,arm);width=float(actual['width_m'])
        samples.append(dict(waited_steps=waited,actual_width_m=width,actual_timestamp=actual.get('timestamp')))
        if not np.isfinite(width):
            raise StageRefusal('Actual Red release opening encoder width must be finite')
        if width>=.0895:
            return actual,dict(maximum_wait_steps=150,actual_wait_steps=waited,actual_samples=samples,
                unchanged_open_width_gate_m=.0895,same_original90mm_command=True,
                held_source_reused=False,new_grip_command_issued=False)
        if width<.045 or waited>=150:
            raise StageRefusal('Actual Red release opening has not reached original90mm width within bounded150steps '+json.dumps(samples))
        ctx.tool('robot.wait_steps',steps=10);waited+=10


def _owned_terminal_control(ctx,state,scene,*,maximum_source_age_s=10.):
    """Reauthenticate an unrevoked, recent original6mm visual attachment.

    The native propagator owns source/epoch/TTL/jaw validation. This function
    never turns an UNKNOWN into a current pose, resets a date, or changes the
    declared6mm envelope. All later geometry and physical gates still apply.
    """
    parameters=state['parameters']
    if parameters.get('conditional_attachment_id') or not parameters.get('owned_held_belief_source'):
        raise StageRefusal('Conditional terminal entry requires an original owned visual source')
    belief=_belief(ctx,state)
    if (belief.get('contract')!='pyramid_rgb_owned_held_control_v1' or
            not np.isclose(_bound(belief),.006,rtol=0.,atol=1.e-10) or
            belief.get('source')!=parameters['owned_held_belief_source']):
        raise StageRefusal('Conditional terminal entry requires native-authenticated original6mm ownership')
    source=belief['source'];age=float(belief.get('source_age_s',float('inf')))
    now=float(belief.get('current_timestamp',float('nan')))
    source_time=float(source.get('timestamp',float('nan')))
    if (source.get('piece')!='red' or source.get('mode')!='pyramid' or
            not np.isfinite(age) or not 0.<=age<=maximum_source_age_s or
            not np.isfinite(now) or not np.isfinite(source_time) or abs(now-source_time-age)>.03):
        raise StageRefusal('Conditional terminal visual source exceeds its authenticated age scope')
    board=scene.get('board_for_planning',{})
    board_age=now-float(board.get('source_timestamp',float('-inf')))
    if (board.get('source')!='current_rgb_measurement' or
            not 0.<=board_age<=1. or float(board.get('uncertainty_m',float('inf')))!=0.):
        raise StageRefusal('Conditional terminal entry requires a current actual board RGB measurement')
    jaw=belief.get('retained_jaw_proprioception',{})
    source_width=float(jaw.get('source_jaw_width_m',float('nan')))
    width=float(jaw.get('current_jaw_width_m',float('nan')))
    if (jaw.get('contradicted') is not False or not np.isfinite(source_width) or source_width<.010 or
            not np.isfinite(width) or not .008<width<.045 or abs(width-source_width)>.0025):
        raise StageRefusal('Conditional terminal entry refuses a jaw contradiction or lost width retention')
    actual=base.proprio(ctx,_arm(state))
    actual_width=float(actual.get('width_m',float('nan')))
    q=np.asarray(actual.get('robot_encoder_values',[]),float)
    authenticated=np.asarray(belief.get('current_robot_encoder_values',[]),float)
    if (q.shape!=(16,) or authenticated.shape!=(16,) or
            not np.isfinite(q).all() or not np.isfinite(authenticated).all() or
            np.max(np.abs(q-authenticated))>.002 or not np.isfinite(actual_width) or
            abs(actual_width-width)>.0004):
        raise StageRefusal('Conditional terminal entry requires current actual encoder/jaw agreement')
    return belief,actual


def _terminal_source_admission(ctx,state,scene,belief,refresh):
    if _latest_measured(refresh):
        if not np.isclose(_bound(belief),.006,rtol=0.,atol=1.e-10):
            raise StageRefusal('Original terminal descent requires the original6mm bound')
        return belief,dict(kind='actual_RGB_q16_rebound',current_target_measured=True,
            conditional_control=False,visual_receipt_earned=False)
    if (not refresh.get('attempts') or
            any(row.get('status')!='RGB_UNKNOWN' for row in refresh.get('attempts',[])) or
            'red' in state.get('observation',{}).get('poses',{})):
        raise StageRefusal('Explicit current Red RGB contradiction cannot use a retained terminal belief')
    belief,before=_owned_terminal_control(ctx,state,scene)
    before_body=handoff._body(before,belief)
    ctx.tool('robot.wait_steps',steps=6)
    belief,after=_owned_terminal_control(ctx,state,scene)
    after_body=handoff._body(after,belief)
    drift=float(visual_memory.shape_displacement('red',before_body,after_body))
    angle=float((Rotation.from_quat(before_body[3:]).inv()*Rotation.from_quat(after_body[3:])).magnitude())
    width_drift=abs(float(after['width_m']-before['width_m']))
    if drift>.0004 or angle>.015 or width_drift>.0004:
        raise StageRefusal('Conditional terminal entry requires actual short encoder/jaw stability')
    error=_hover_error(state,scene,belief,after)
    if not error['transport_six_mm_predicate'] or not error['ready_for_original_terminal']:
        raise StageRefusal('Conditional terminal current-FK hover prediction is outside original axis/6mm bounds')
    error['estimated_from']='Authenticated dated visual source and current public FK, conditional rigid persistence'
    return belief,dict(kind='authenticated_recent_owned6_prediction',current_target_measured=False,
        conditional_control=True,visual_receipt_earned=False,source=belief['source'],
        source_age_s=belief['source_age_s'],held_envelope_m=.006,
        actual_short_stability=dict(before=before,after=after,bead_drift_m=drift,
            rotation_drift_rad=angle,jaw_width_drift_m=width_drift),conditional_hover_error=error,
        scope='UNKNOWN remains UNKNOWN. Native ownership/epoch and current robot stability qualify conditional control only; object rigidity and6mm are declared assumptions, no new measurement or physical success.')


def release(ctx,state):
    state=deepcopy(state);arm=_arm(state)
    if state.get('last_outcome')!='carried':raise StageRefusal('Red goal release requires preceding transport and physical hover checkpoint')
    state,belief,scene,refresh=_refresh(ctx,state,'before_goal_drop5',optimizer_belief=state.get('transport_record',{}).get('current_attachment'))
    belief,source_admission=_terminal_source_admission(ctx,state,scene,belief,refresh)
    if source_admission['conditional_control']:
        pre_descent_feedback=dict(status='conditional_authenticated_owned_hover_prediction',
            current_target_measured=False,conditional_control=True,
            source_admission=source_admission,rounds=[source_admission['conditional_hover_error']])
    else:
        state,belief,scene,pre_descent_feedback=_align_goal_hover(ctx,state,belief,scene,refresh,
            label='before_original_vertical_drop5',require_terminal_axis=True)
        if not pre_descent_feedback.get('current_target_measured'):
            # A screened correction can occlude a previously visible Red.
            # Reauthenticate at this new boundary through the identical owned6
            # admission; the initial measured flag cannot authorize motion.
            belief,readmission=_terminal_source_admission(ctx,state,scene,belief,
                pre_descent_feedback.get('latest_refresh',{}))
            source_admission=dict(readmission,initial_source_admission=source_admission,
                qualification_phase='after_hover_feedback',conditional_control_qualified=True)
            pre_descent_feedback=dict(pre_descent_feedback,
                status='visual_unknown_authenticated_owned_hover_prediction_after_feedback',
                current_target_measured=False,conditional_control=True,
                conditional_control_qualified=True,source_admission=source_admission)
    if ((not pre_descent_feedback.get('current_target_measured') and not source_admission['conditional_control']) or
            _bound(belief)>.0060000001 or scene['board_for_planning'].get('source')!='current_rgb_measurement'):
        raise StageRefusal('Original terminal descent requires actual fresh Red6mm and current board RGB after feedback')
    board,target,desired,_,_=_goal(state,scene)
    height_feedback=_release_height_feedback(ctx,state,scene,belief)
    height_feedback.update(current_target_measured=source_admission['current_target_measured'],
        conditional_control=source_admission['conditional_control'],
        visual_source=belief.get('source'),
        current_estimate_role='Same authenticated visual attachment propagated through current actual q16; UNKNOWN is not a fresh pose')
    height_feedback['initial_feedback_command_height_m']=height_feedback['command_release_height_m']
    command_height,height_selection=_select_release_height(ctx,state,scene,belief,height_feedback)
    height_feedback.update(command_release_height_m=command_height,online_height_selection=height_selection)
    requested=_body_waypoint(target+[0,0,command_height],desired,belief)
    descent=_plan(ctx,state,scene,belief,[requested],terminal=True)
    descent_execution=_drive(ctx,state,descent)
    # Rebind the actual reached object only when true RGB supports it; otherwise
    # preserve the same authenticated conditional attachment and its full bound.
    state,belief,scene,reached_refresh=_refresh(ctx,state,'at_actual_drop5',optimizer_belief=belief)
    board,target,desired,_,_=_goal(state,scene)
    actual_body=handoff._body(base.proprio(ctx,arm),belief)
    actual_height=float(actual_body[2]-target[2])
    if source_admission['conditional_control'] and not _latest_measured(reached_refresh):
        if (any(row.get('status')!='RGB_UNKNOWN' for row in reached_refresh.get('attempts',[])) or
                'red' in state.get('observation',{}).get('poses',{})):
            raise StageRefusal('New explicit Red contradiction prevents retained-belief release opening')
        # After descent retain the original propagator's30s scope; the new
        # 10s condition applies only to pre-descent conditional admission.
        belief,_=_owned_terminal_control(ctx,state,scene,maximum_source_age_s=30.)
        actual_body=handoff._body(base.proprio(ctx,arm),belief)
        actual_height=float(actual_body[2]-target[2])
    if (np.linalg.norm(actual_body[:2]-target[:2])>.001 or
            (Rotation.from_quat(actual_body[3:]).inv()*Rotation.from_matrix(desired)).magnitude()>.04 or
            abs(actual_height-command_height)>.0025):
        raise StageRefusal('Actual reached Red attachment does not satisfy original vertical drop5 release corridor')
    sweep=_opening_sweep(ctx,state,scene,belief)
    opening=ctx.tool('robot.set_grip',arm_id=arm,width_m=RELEASE_WIDTH,ramp_steps=18,settle_steps=15)
    opened,opening_width_wait=_await_release_width(ctx,arm)
    # Registered opening invalidates the old held source forever. Post-opening
    # plans use only the distinctly labelled dated release hypothesis below.
    released_pose=np.asarray(sweep['body_before_open'])
    released_height_envelope=max(DROP,command_height,actual_height)
    released_bound=_bound(belief)+released_height_envelope
    actual=base.proprio(ctx,arm);rr=Rotation.from_quat(actual['tcp'][3:]).as_matrix()
    withdrawal=_released_withdrawal_plan(ctx,state,scene,arm,released_pose,released_bound,
        np.asarray(actual['tcp'][:3]),rr)
    withdrawal_execution=_empty_execute(ctx,withdrawal)
    home_active=_home(ctx,state,scene,arm,released_pose,released_bound)
    home_other=_home(ctx,state,scene,1-arm,released_pose,released_bound)
    # Both public-home TCPs and open jaws must settle for the original full12
    # two-second physical postcondition before this phase exits.
    ctx.tool('robot.wait_steps',steps=110)
    state,first,second,receipt,refusal,memory=_record_release_pair(ctx,state,target,desired)
    earned=list(state.get('executed_placements',[]))
    if receipt is not None and 'red' not in earned:earned.append('red')
    record=dict(contract='Red_two_regrasp_actual_attachment_strict_goal_release_v2',arm=arm,
        runtime_attachment_before_open=belief,held_envelope_m=_bound(belief),
        actual_drop5_refresh=refresh,pre_descent_hover_feedback=pre_descent_feedback,actual_reached_refresh=reached_refresh,
        requested_release_tcp=requested['position'],release_height_m=DROP,
        nominal_release_height_m=DROP,command_release_height_m=command_height,
        terminal_source_admission=source_admission,
        actual_reached_body_height_estimate_m=actual_height,
        actual_reached_target_measured=_latest_measured(reached_refresh),
        release_height_feedback=height_feedback,released_height_envelope_m=released_height_envelope,
        descent=descent_execution,opening_sweep=sweep,opening_result=opening,actual_postopen=opened,
        opening_width_wait=opening_width_wait,
        old_held_source_permanently_revoked=True,
        released_target_hypothesis=dict(pose=released_pose.tolist(),inflation_m=released_bound,
            source='Dated actual preopen RGB-owned/conditional attachment plus maximum nominal/command/reached vertical release envelope',
            current_target_measured=False,physical_goal_or_stability_claim=False),
        withdrawal=withdrawal_execution,home_active=home_active,home_other=home_other,
        release_rgb_pair=[first,second],new_visual_release_receipt=receipt,receipt_refusal=refusal,
        target_3d_status='measured_in_new_release_pair' if 'red' in first.get('poses',{}) and 'red' in second.get('poses',{}) else 'UNKNOWN',
        independent_physical_release_required=True)
    p={**state['parameters']};p.pop('owned_held_belief_source',None);p.pop('conditional_attachment_id',None)
    state.update(parameters=p,memory=memory,release_execution=record,executed_placements=earned,
        execution_tokens={**state.get('execution_tokens',{}),'red':record})
    # The unchanged physical checkpoint must prove goal/speed/no-load/home/
    # stable-tail before either route advances; UNKNOWN remains uncertain.
    return result('placed' if receipt is not None else 'uncertain',state,second,'release',release_execution=record,
                  visual_status=record['target_3d_status'],current_target_measured='red' in second.get('poses',{}),
                  visual_receipt_verified=receipt is not None)


def verify_release(ctx,state):
    state=deepcopy(state)
    if (state.get('piece')!='red' or state.get('last_outcome') not in ('placed','uncertain') or
            'red' not in state.get('execution_tokens',{})):
        raise StageRefusal('Final Red handoff requires preceding checked physical release execution')
    previous=state['observation'];hint={}
    if 'red' in previous.get('poses',{}):hint['red']=previous['poses']['red']
    current=c.state(ctx,hint);state=save_observation(state,current)
    if current['timestamp']<previous['timestamp']:
        raise StageRefusal('Final Red release verification cannot roll back RGB time')
    ctx.tool('robot.wait_steps',steps=6)
    token=state['execution_tokens']['red']
    if token.get('new_visual_release_receipt') is None:
        if not (unified_contract.conditional_handoff_allowed(state) or
                unified_contract.terminal_visual_unknown_allowed(state)):
            raise StageRefusal('Final visual UNKNOWN requires the canonical explicitly enabled terminal execution scope')
        state,visual=unified_contract.commit_execution_hypothesis(state)
        state['pending_placement_receipt']=dict(piece='red',target_3d_status=visual['status'],
            visual_diagnostics=visual,terminal_only=True,visual_receipt_earned=False,
            independent_final_verification_required=True)
    answer=unified_contract.verified_handoff(ctx,state,'red')
    answer['evidence'].update(visual_status=state['release_execution']['target_3d_status'],
        independent_physical_handoff_required=True,canonical_ledger_preserved=True)
    return answer
