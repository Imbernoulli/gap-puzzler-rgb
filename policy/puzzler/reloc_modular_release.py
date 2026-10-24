"""Parameterized conditional release. No simulatortruth or teacher commands."""
import json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from . import model,visual_memory
from . import vision_controller as c
from .vision_stages import StageState,StageResult,StageRefusal,_plain,_observation,_require,_pause,_piece_parameters
PlanningRefusal=c.PlanningRefusal
from . import conditional_scene as segment_scene
from .scene_reference import select_reference


def camera_ray_observability_evidence(observed, name):
    """Describe current calibrated RGB/CAD fit support, separately from CAD safety.

    A fit residual is image evidence, never a physical camera collision gap.
    Missing support returns UNKNOWN; bounded ownership admission decides whether
    the physical release may proceed before a new postrelease RGB receipt.
    """
    detections=observed.get('detections',{})
    row=detections.get(name)
    if not isinstance(row,dict) or row.get('visible') is not True:
        return None
    views=row.get('views',[])
    support=np.asarray(row.get('center_support',[]),dtype=float)
    residual=float(row.get('median_residual_m',float('nan')))
    rms=float(row.get('rms_residual_m',float('nan')))
    attribution=row.get('independent_rgb_support',{})
    by_view=attribution.get('bead_support_by_view',{}) if isinstance(attribution,dict) else {}
    supported=[sum(bool(v) for v in values) for values in by_view.values()]
    if len(views)<2 or support.size<2 or int(np.sum(support>=.6))<2:
        return None
    if not np.isfinite([residual,rms]).all() or residual>.0018 or rms>.004:
        return None
    # Some calibrated RGB passes omit per-bead attribution for an occluded
    # view even though the rigid multi-view fit is valid.  The fit itself is
    # the camera-ray evidence; attribution is retained when present but is
    # not allowed to turn a valid visible fit into an UNKNOWN release.
    margin=min(.004-rms,.0018-residual)
    if margin<.0005:
        return None
    return dict(
        kind='multiview_rgb_ray_cad_observability_v1',
        rgb_fit_margin_m=float(margin),
        visible_views=list(views),
        center_support=support.tolist(),
        median_ray_fit_residual_m=residual,
        rms_ray_fit_residual_m=rms,
        supported_beads_per_view=supported,
        frame_sha256=observed.get('frame_sha256'),
        timestamp=observed.get('timestamp'),
        policy_source='current RGB calibrated ray/CAD fit; no GT or target pose fill',
    )


from .release_admission import check_bounded_release, target_wrist_camera_guard


def segment_conditional_release(ctx,state:StageState)->StageResult:
    """One bounded conditional release; source priors never become RGB poses."""
    name=state['piece'];arm=state['arm'];p=_piece_parameters(state)
    mark=ctx.tool('robot.execution_journal_cursor')
    belief=(ctx.tool('puzzle.propagate_conditional_attachment',attachment_id=p['conditional_attachment_id'],arm_id=arm)if p.get('conditional_attachment_id')else ctx.tool('puzzle.propagate_pyramid_held_belief',source=p['owned_held_belief_source'],arm_id=arm))
    # Generic release profiles may carry a legacy skip flag from an older
    # rehearsal.  A profile that explicitly requires current RGB takes
    # precedence, so this chain cannot silently fall back to a historical
    # pre-open image.
    historical_preopen=bool(p.get('skip_preopen_rgb'))
    if historical_preopen:
        prior=p['historical_complete_rgb']
        observed=dict(poses={},timestamp=prior['timestamp'],frame_sha256=prior['frame_sha256'],observation_source='historical_rgb_prior_for_planning',is_current_rgb=False)
    else:observed=c.state(ctx,{name:belief['predicted_body_pose']})
    # Active RGB inspection repairs the common held-object occlusion before
    # release-camera certification. It uses bounded robot motion plus a fresh
    # multi-view RGB/CAD fit; no pose is filled from the prediction.
    if name not in observed.get('poses',{}) or not observed.get('detections',{}).get(name,{}).get('visible',False):
        try:
            observed=c.inspect_held(ctx,name,np.asarray(belief['predicted_body_pose'],dtype=float),arm,require_board=True,max_moves=3)
        except RuntimeError:
            pass
    raw=observed['poses'];measured=name in raw
    body=np.asarray(raw[name] if measured else belief['predicted_body_pose'])
    admission=check_bounded_release(name,belief,observed,p)
    prior=p['historical_complete_rgb'];age=belief['current_timestamp']-prior['timestamp']
    board_measured='board'in raw
    if not board_measured and not p.get('allow_conditional_board_persistence'):raise StageRefusal('Fresh board unavailable and no explicit conditional persistence contract')
    board=np.asarray(raw['board']if board_measured else prior['poses']['board']);br=Rotation.from_quat(board[3:]).as_matrix()
    board_margin=0. if board_measured else .004
    reference=select_reference(observed,prior,now=belief['current_timestamp'],max_age_s=p.get('historical_max_age_s',15.),board_limit_m=.0015,refusal=StageRefusal)
    occupancy={};historical_names=[]
    for item in model.CELLS:
        if reference['mode']=='current_complete_reference' and item in p.get('planning_prefix',p.get('actual_executed_prefix',[])) and state.get('belief_ledger',{}).get(item,{}).get('status')=='current_rgb_conflict_goal_envelope_invalidated':
            poses=[raw[item]];source='current_rgb_after_goal_envelope_invalidation';inflation=.006
        elif item in p.get('planning_prefix',p.get('actual_executed_prefix',[]))and state.get('belief_ledger',{}).get(item,{}).get('status')=='current_rgb_conflict_goal_envelope_invalidated':
            conflict=state['belief_ledger'][item];candidate=next((row for row in reversed(p.get('historical_rgb_records',[]))if item in row.get('poses',{})and row['timestamp']>=conflict['source_timestamp']and 0<=belief['current_timestamp']-row['timestamp']<=p.get('historical_max_age_s',30.)),None)
            if candidate is None:raise StageRefusal('Conflicted prefix has no current replan source: '+item)
            poses=[candidate['poses'][item]];source='conditional_persistence_of_replanned_conflict_RGB';inflation=.006;historical_names.append(item)
        elif item in p.get('planning_prefix',p.get('actual_executed_prefix',[])):
            from puzzler.vision_observer import goals
            center,relative=goals(p['mode'],p['pyramid_plan'])[item]
            poses=[[*list(board[:3]+br@center),*Rotation.from_matrix(br@relative).as_quat().tolist()]]
            if item in raw:poses.append(raw[item])
            source='conditional_given_stage_prefix_public4mm_plus_board_uncertainty_and_raw'if item in p.get('given_prefix_names',[])else'conditional_actual_executed_prefix_public4mm_plus_board_uncertainty_and_raw';inflation=.004+board_margin;historical_names.append(item)
        elif item in raw:
            poses=[raw[item]];source='current_rgb_measurement';inflation=0.
        elif item==name:
            poses=[belief['predicted_body_pose']];source='conditional_pregrasp_close_attachment_robot_fk'if p.get('conditional_attachment_id')else'conditional_owned_rgb_robot_fk_held';inflation=float(belief['uncertainty_m'])
        else:
            history=p.get('historical_rgb_records',[prior]);candidate=next((row for row in reversed(history)if item in row.get('poses',{})and 0<=belief['current_timestamp']-row['timestamp']<=p.get('historical_max_age_s',30.)),None)
            if candidate is None:raise StageRefusal('No bounded historical occupancy for '+item)
            poses=[candidate['poses'][item]];source='conditional_recent_rgb_unmoved_unplaced';inflation=.004;historical_names.append(item)
        occupancy[item]=dict(source=source,pose_hypotheses=poses,inflation_m=inflation,shape='public CAD sphere cover')
    scene=dict(contract='conditional_recent_rgb_occupancy_v1',target=name,reference=reference,
        measurement=_observation(observed),occupancy=occupancy,measured_names=sorted(raw),
        conditional_names=historical_names,predicted_names=[] if measured else [name],board_for_planning=dict(pose=board.tolist(),source='current_rgb_measurement'if board_measured else'conditional_recent_rgb_board_persistence',uncertainty_m=board_margin,current_board_measured=board_measured),
        scope='Historical object/board persistence and authenticated held uncertainty are separately declared planning assumptions; no current-pose fill, covariance, or accuracy guarantee.')
    desired=br@np.asarray(p.get('release_rotation_bias_board',np.eye(3)))@np.asarray(state['plan']['target_R'])
    delta=desired@Rotation.from_quat(body[3:]).as_matrix().T
    tcp,q=c.hand(ctx,arm);qtarget=c.qdict(Rotation.from_matrix(delta@Rotation.from_quat(c.qarr(q)).as_matrix()).as_quat())
    target=board[:3]+br@(np.asarray(state['plan']['target_center'])+np.asarray(p.get('release_translation_bias_board_m',[0.,0.,0.])))
    zero_tcp=target+delta@(tcp-body[:3]);hand_R=Rotation.from_quat(c.qarr(qtarget)).as_matrix()
    enc=belief['current_robot_encoder_values'];offset=0 if arm==0 else 8;actual_width=float(enc[offset+6]-enc[offset+7])
    open_width=p.get('release_open_width',p.get('open_width',.055));height=None;clearance=[]
    bias_includes_height=p.get('release_bias_includes_height',False)
    heights=[0.]if bias_includes_height else np.linspace(p.get('release_height',.012),p.get('max_release_height',.024),7)
    for h in heights:
        rows=[segment_board_conditional_clearance(zero_tcp+[0,0,h],hand_R,float(w),scene,name,board,board_margin) for w in np.linspace(actual_width,open_width,9)]
        gap=min(row['clearance_m']for row in rows);clearance.append(dict(height_m=float(h),minimum_clearance_m=gap))
        # Keep scanning after a board-clear candidate: the wrist guard may
        # require a higher approach at the same target frame.
        if gap>=p.get('hand_clearance_margin',.0005) and height is None:
            height=float(h)
    if height is None:raise PlanningRefusal('Bounded conditional actual-jaw sweep has no clear release height')
    requested=zero_tcp+[0,0,height]
    # Keep the camera certificate tied to this exact pre-open RGB frame.  A
    # pose may be absent in a partially occluded frame, but a visible
    # multiview fit is still useful evidence for the camera half of the guard;
    # the wrist half below continues to use the current FK/body estimate.
    camera_evidence=camera_ray_observability_evidence(observed,name)
    # Current RGB fit diagnostics are observability evidence, never physical
    # camera clearance. Missing current target support remains UNKNOWN; an
    # authenticated bounded held estimate may still admit a geometric release.
    guard_parameters={**p,'target_wrist_camera_guard_body_uncertainty_m':admission['physical_guard_body_inflation_m'],'bounded_release_admission':admission}
    if camera_evidence is not None:
        guard_parameters['target_wrist_camera_guard_evidence']=camera_evidence
    target_guard=None
    # Choose the first board-clear release height that also clears the
    # non-finger wrist geometry.  The old implementation selected height
    # using only scene clearance, then checked the wrist at that height; this
    # could reject a valid higher approach while preserving the same target
    # frame and the 0.5 mm guard.
    for candidate in clearance:
        if candidate['minimum_clearance_m'] < p.get('hand_clearance_margin',.0005):
            continue
        candidate_guard=target_wrist_camera_guard(zero_tcp+[0,0,candidate['height_m']],hand_R,open_width,np.r_[target+[0,0,candidate['height_m']],Rotation.from_matrix(desired).as_quat()],name,guard_parameters)
        if candidate_guard.get('status')=='passed':
            height=candidate['height_m'];requested=zero_tcp+[0,0,height];target_guard=candidate_guard;break
    if target_guard is None:
        # Preserve the explicit UNKNOWN/rejected diagnostic rather than
        # silently lowering either guard threshold.
        target_guard=target_wrist_camera_guard(requested,hand_R,open_width,np.r_[target+[0,0,height],Rotation.from_matrix(desired).as_quat()],name,guard_parameters)
    if target_guard.get('status')!='passed':
        raise PlanningRefusal('Explicit target-to-wrist/camera guard '+target_guard.get('status','UNKNOWN')+': '+str(target_guard))
    c.move(ctx,requested,qtarget,arm,accuracy=.0025)
    actual,actual_q=c.hand(ctx,arm)
    opening=ctx.tool('robot.set_grip',arm_id=arm,width_m=open_width)
    c.withdraw(ctx,arm)
    support_release=None
    if state.get('support_anchor_evidence'):
        from .modular_controller import load_skill_module
        support=load_skill_module(p['support_anchor']['module']);support_release=support.release_support_anchor(ctx,state)
    c.grip(ctx,arm,max(.055,open_width))
    home=ctx.tool('robot.go_home',arm_id=arm)
    support_open=ctx.tool('robot.set_grip',arm_id=1-arm,width_m=.055)
    support_home=ctx.tool('robot.go_home',arm_id=1-arm)
    _pause(ctx,50)
    hint={name:[*target.tolist(),*Rotation.from_matrix(desired).as_quat().tolist()]}
    first=c.state(ctx,hint)
    _pause(ctx,50)
    second=c.state(ctx,{name:first['poses'][name]} if name in first.get('poses',{}) else hint)
    receipt=None;receipt_error=None;memory=state['memory']
    try:
        if not {'board',name}.issubset(first['poses'])or not {'board',name}.issubset(second['poses']):
            raise visual_memory.MemoryRefusal('Released target or board absent currentRGB')
        final_tcp,_=c.hand(ctx,arm)
        receipt=visual_memory.release_receipt(name,first,second,final_tcp,p)
        memory=visual_memory.observe(memory,second,p)
        memory=visual_memory.mint_placement(memory,receipt,second,p)
    except visual_memory.MemoryRefusal as error:receipt_error=str(error)
    record=dict(bounded_release_admission=admission,contract='conditional_segment_release_execution_v1',execution_mark=mark,source=p['owned_held_belief_source'],
        held_belief=belief,current_target_measured_at_preopen=measured,preopen_rgb=_observation(observed),preopen_rgb_role='historical_prior_only'if historical_preopen else'current_rgb',planning_scene=scene,
        original_historical_complete_rgb=prior,current_board_measured_at_preopen=board_measured,conditional_board_persistence_margin_m=board_margin,actual_preopen_width_m=actual_width,width_sweep_m=[actual_width,open_width],
        release_height_m=height,release_bias_includes_height=bias_includes_height,release_bias_board_m=p.get('release_translation_bias_board_m',[0.,0.,0.]),clearance_candidates=clearance,requested_release_tcp=requested.tolist(),requested_rotation=qtarget,
        actual_preopen_tcp=actual.tolist(),actual_preopen_rotation=actual_q,open_width_m=open_width,open_result=opening,home_result=home,support_open_result=support_open,support_home_result=support_home,support_release=support_release,
        rigid_held_belief_after_open='EXPIRED; historical only',target_3d_status='measured_in_new_release_pair'if name in first['poses']and name in second['poses']else'UNKNOWN',placement_receipt_status='verified_3mm_stable_pair'if receipt else'UNVERIFIED',
        new_visual_release_receipt=receipt,receipt_refusal=receipt_error,target_wrist_camera_guard=target_guard,release_rgb_pair=[{k:row[k]for k in ['poses','timestamp','frame_sha256','observation_source','assessment','robot_geometry_provenance']if k in row}for row in(first,second)],
        scope='Execution completion with separate optional actual RGB receipt. Conditional historical occupancy is not original complete-current-scene guarantee; no full12 claim.')
    return _plain(dict(route='executed',state={**state,'memory':memory,'observation':_observation(second),
        'release_execution':record,'last_stage':'conditional_segment_release_execution','last_outcome':'executed'},
        evidence=dict(kind='conditional_segment_release_execution',timestamp=second['timestamp'],frame_sha256=second['frame_sha256'])))


def segment_release_execution_checkpoint(world,outputs):
    """Recompute the execution token from actual tool records, not stage claims."""
    try:
        state=outputs['state'];record=state['release_execution'];e=outputs['evidence'];seen=world.sim_state['vision']
        if outputs.get('decision')!='executed'or e['kind']!='conditional_segment_release_execution':return False
        if e['timestamp']!=seen['timestamp']or e['frame_sha256']!=seen['frame_sha256']:return False
        belief=record['held_belief']
        if belief not in world.sim_state['pyramid_held_belief_history']or belief['source']!=state['parameters']['owned_held_belief_source']:return False
        if record['planning_scene']['measurement']!=record['preopen_rgb']:return False
        rows=world.sim_state['robot_command_journal'];nonce=world.sim_state['robot_command_session'];mark=record['execution_mark']
        if mark['session_nonce']!=nonce:return False
        markrows=[r for r in rows if r['call_id']==mark['call_id']and r['event']=='return'and r['tool']=='robot.execution_journal_cursor'and r.get('result')==mark]
        if len(markrows)!=1:return False
        rows=[r for r in rows if r['call_id']>mark['call_id']and r.get('session_nonce')==nonce]
        returns={r['call_id']:r for r in rows if r['event']=='return'}
        calls=[]
        for row in rows:
            if row['event']!='call':continue
            raw=json.dumps(row['arguments'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            if hashlib.sha256(raw).hexdigest()!=row['arguments_sha256']:return False
            if row['call_id']in returns:calls.append(row)
        opening=next(r for r in calls if r['tool']=='robot.set_grip'and r['arguments'].get('arm_id')==state['arm']and r['arguments'].get('width_m')==record['open_width_m']and returns[r['call_id']].get('result')==record['open_result'])
        home=next(r for r in calls if r['tool']=='robot.go_home'and r['arguments'].get('arm_id')==state['arm']and r['call_id']>opening['call_id']and returns[r['call_id']].get('result')==record['home_result'])
        if record['open_result'].get('clamped')is not False:return False
        moves=[r for r in calls if r['tool']=='robot.execute_trajectory'and r['arguments'].get('arm_id')==state['arm']]
        if not any(r['call_id']<opening['call_id']for r in moves)or not any(opening['call_id']<r['call_id']<home['call_id']for r in moves):return False
        if any(returns[r['call_id']].get('result',{}).get('status')!='ok'for r in moves):return False
        requested=np.asarray(record['requested_release_tcp']);qr=Rotation.from_quat(c.qarr(record['requested_rotation']))
        encoder_pass=False
        for row in calls:
            if row['tool']!='robot.get_ee_pose'or row['arguments'].get('arm_id')!=state['arm']or row['call_id']>=opening['call_id']:continue
            h=returns[row['call_id']].get('result',{}).get('pose',{});pp=h.get('position',{});qq=h.get('rotation',{})
            if set(pp)>={'x','y','z'}and set(qq)>={'x','y','z','w'}:
                if np.linalg.norm(np.array([pp[k]for k in 'xyz'])-requested)<.0025 and(qr.inv()*Rotation.from_quat([qq[k]for k in ['x','y','z','w']])).magnitude()<.04:encoder_pass=True
        receipt=record.get('new_visual_release_receipt')
        if receipt:
            if receipt!=visual_memory.release_receipt(state['piece'],receipt['before'],receipt['after'],receipt['released_tcp_position'],state['parameters']):return False
        return encoder_pass and world.sim_state['robot_tcp_poses'][str(state['arm'])][2]>=1.05
    except(KeyError,ValueError,TypeError,StopIteration,visual_memory.MemoryRefusal):return False


def segment_board_conditional_clearance(position,rotation,width,scene,name,board,board_margin):
    from puzzler.gripper_geometry import clearance,scene_clearance
    centers,radii=segment_scene.spheres(scene,exclude=(name,))
    actual=clearance(position,rotation,width,centers,radius=radii)
    fixed=scene_clearance(position,rotation,width,[],table_z=float(board[2]-.003+board_margin),plane_z=float(board[2]+.013+board_margin),board=board)
    return actual if actual['clearance_m']<fixed['clearance_m']else fixed
