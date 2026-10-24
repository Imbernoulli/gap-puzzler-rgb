"""Observation/state-driven puzzle skills. No teacher actions or simulator truth."""
from copy import deepcopy
import hashlib,importlib,json
from typing import Any,TypedDict
import numpy as np
from scipy.spatial.transform import Rotation
from . import model,visual_memory,conditional_scene
from . import vision_controller as c
from .vision_observer import goals
from .vision_stages import StageRefusal,_plain,_observation,_require,_pause
from .modular_release import segment_conditional_release

ORDER=('azure','orange','yellow','teal','purple','cyan','mint','lime','maroon','navy','pink','red')

class ModularState(TypedDict,total=False):
    piece:str
    parameters:dict[str,Any]
    plan:dict[str,Any]
    transport_record:dict[str,Any]
    source_contact_evidence:dict[str,Any]
    conditional_attachment:dict[str,Any]
    arm:int
    memory:dict[str,Any]
    observation:dict[str,Any]
    rgb_history:list[dict[str,Any]]
    planning_scene:dict[str,Any]
    executed_placements:list[str]
    initialized_prefix:list[str]
    execution_tokens:dict[str,Any]
    belief_ledger:dict[str,Any]
    history:list[dict[str,Any]]
    prediction:list[float]
    last_stage:str
    last_outcome:str
    release_execution:dict[str,Any]
    support_anchor_evidence:dict[str,Any]
    pending_placement_receipt:dict[str,Any]
    attempt:int
    repair_attempts:list[dict[str,Any]]
    released_placements:list[str]
    release_visual_status:str
    transfer:dict[str,Any]
    receiver_delivery:dict[str,Any]
    handoff:dict[str,Any]
    handoff_history:list[dict[str,Any]]
    held_status:str
    scene_refresh_required_before_next_motion:bool

class ModularResult(TypedDict):
    route:str
    state:ModularState
    evidence:dict[str,Any]

from gap_core.schema import TYPE_REGISTRY
TYPE_REGISTRY['ModularState']=ModularState
TYPE_REGISTRY['ModularResult']=ModularResult


def load_skill_module(name):
    approved={'puzzler.modular_red_contact','puzzler.modular_cyan_anchor','puzzler.modular_maroon_anchor','puzzler.cyan_geometry','puzzler.modular_maroon_geometry','puzzler.modular_maroon_adapter','puzzler.reloc_modular_release','puzzler.red_computed_carry','puzzler.cyan_modular_adapter','puzzler.cyan_receiver_delivery','puzzler.cyan_receiver_feedback_delivery','puzzler.cyan_receiver_release_feedback_delivery'}
    if name not in approved:raise StageRefusal('Unapproved modular skill hook: '+str(name))
    return importlib.import_module(name)


def profile_parameters(base,name):
    profiles=base['skill_profiles']
    piece_keys=set().union(*(row.keys()for row in profiles.values()))
    transient={'owned_held_belief_source','conditional_attachment_id','conditional_attachment','conditional_prefix','current_board_for_inverse_planning','historical_complete_rgb','historical_rgb_records','historical_max_age_s','actual_executed_prefix','planning_prefix','given_prefix_names'}
    shared={key:value for key,value in base.items()if key not in piece_keys and key not in transient}
    p={**shared,**profiles[name]};p['skill_profiles']=profiles
    p['open_width']=p.get('source_open_width',.055)
    p['pieces']={};p['actual_executed_prefix']=[]
    return p


def initial_state(name,base,*,memory=None,executed=(),execution_tokens=None,history=(),given_prefix=()):
    ledger={n:dict(status='conditional_given_stage_prefix',conditional_goal_margin_m=.004)for n in given_prefix}
    ledger.update({n:dict(status='executed_unknown',conditional_goal_margin_m=.004)for n in executed})
    return dict(piece=name,parameters=profile_parameters(base,name),plan={},arm=-1,memory=deepcopy(memory)if memory is not None else visual_memory.new_memory(given_prefix,conditional_local_benchmark=bool(given_prefix)),observation={},rgb_history=deepcopy(list(history)),planning_scene={},executed_placements=list(executed),initialized_prefix=list(given_prefix),execution_tokens=deepcopy(execution_tokens or{}),belief_ledger=ledger,history=[],prediction=[],last_stage='',last_outcome='',release_execution={},support_anchor_evidence={},pending_placement_receipt={},attempt=0)


def save_observation(state,observed):
    history=deepcopy(state.get('rgb_history',[]));packet={k:deepcopy(observed[k])for k in ['poses','detections','timestamp','frame_sha256','observation_source','assessment','robot_geometry_provenance','bounded_rgb_source','prefix_compatibility']if k in observed};
    if not history or history[-1].get('timestamp')!=packet.get('timestamp')or history[-1].get('frame_sha256')!=packet.get('frame_sha256')or history[-1].get('poses')!=packet.get('poses'):history.append(_plain(packet))
    ledger=deepcopy(state.get('belief_ledger',{}))
    if 'board'in observed.get('poses',{}):
        for name in [*state.get('initialized_prefix',[]),*state.get('executed_placements',[]),*state.get('released_placements',[])]:
            if name not in observed['poses']:continue
            residual=visual_memory.goal_error(name,observed,state['parameters'])
            if residual>=.004:ledger[name]={**ledger.get(name,{}),'status':'current_rgb_conflict_goal_envelope_invalidated','source_timestamp':observed['timestamp'],'frame_sha256':observed['frame_sha256'],'visual_goal_residual_m':residual}
            elif ledger.get(name,{}).get('status')=='current_rgb_conflict_goal_envelope_invalidated':ledger[name]={**ledger[name],'status':'reacquired_current_goal_compatible','source_timestamp':observed['timestamp'],'frame_sha256':observed['frame_sha256'],'visual_goal_residual_m':residual,'previous_invalidation':deepcopy(ledger[name])}
    return{**state,'observation':_observation(observed),'rgb_history':history[-48:],'belief_ledger':ledger}


def prior_for(state,name,now,max_age=30.):
    for row in reversed(state.get('rgb_history',[])):
        if name in row.get('poses',{})and 0<=now-row['timestamp']<=max_age:return row
    raise StageRefusal('No bounded RGB source for '+name)


def board_belief(state,observed):
    if'board'in observed['poses']:return dict(pose=observed['poses']['board'],source='current_rgb_measurement',uncertainty_m=0.,source_timestamp=observed['timestamp'],source_frame_sha256=observed['frame_sha256'])
    prior=prior_for(state,'board',observed['timestamp']);return dict(pose=prior['poses']['board'],source='conditional_recent_rgb_board_persistence',uncertainty_m=.004,source_timestamp=prior['timestamp'],source_frame_sha256=prior['frame_sha256'])


def scene_belief(state,observed):
    from .prefix_track_reassociation import scene_track,SOURCE as MAP_SOURCE
    p=state['parameters'];name=state['piece'];raw=observed['poses'];board=board_belief(state,observed);bp=np.asarray(board['pose']);br=Rotation.from_quat(bp[3:]).as_matrix();targets=goals(p['mode'],p['pyramid_plan']);occupancy={};ledger=deepcopy(state['belief_ledger'])
    for item in model.CELLS:
        if item in [*state.get('initialized_prefix',[]),*state['executed_placements'],*state.get('released_placements',[])]:
            center,rr=targets[item];goal=[*list(bp[:3]+br@center),*Rotation.from_matrix(br@rr).as_quat().tolist()]
            conditional_map=scene_track(state,item,targets)
            if conditional_map is not None:
                # Preserve a dated conditional public-goal track. All raw
                # measurements and aliases remain in history/diagnostics;
                # neither an ambiguous fit nor the goal becomes measured.
                occupancy[item]=dict(source=MAP_SOURCE,pose_hypotheses=[goal],
                    inflation_m=.004+board['uncertainty_m'],shape='public CAD sphere cover',
                    conditional_MAP_public_goal_only=True,
                    conditional_MAP_witness_timestamp=conditional_map['timestamp'],
                    raw_aliases_retained_in_conditional_MAP_witness=conditional_map,
                    independent_prefix_physical_checkpoint_required=True)
                continue
            conflict=item in raw and visual_memory.goal_error(item,{'poses':{'board':bp.tolist(),item:raw[item]}},p)>=.004
            if conflict:
                # Explicitly invalidate the goal envelope and replan from current RGB.
                ledger[item]={**ledger.get(item,{}),'status':'current_rgb_conflict_goal_envelope_invalidated','source_timestamp':observed['timestamp']}
                options=[raw[item]];margin=.006;source='current_rgb_after_goal_envelope_invalidation'
            else:
                if ledger.get(item,{}).get('status')=='current_rgb_conflict_goal_envelope_invalidated'and item not in raw:raise StageRefusal('Invalidated prefix requires current reacquisition: '+item)
                options=[goal]+([raw[item]]if item in raw else[]);margin=.004+board['uncertainty_m'];source='conditional_given_stage_prefix_goal_plus_raw'if item in state.get('initialized_prefix',[])else'conditional_executed_goal_persistence_plus_current_raw'
                ledger.setdefault(item,dict(status='executed_unknown',conditional_goal_margin_m=.004))
        elif item in raw:options=[raw[item]];margin=0.;source='current_rgb_measurement'
        else:
            prior=prior_for(state,item,observed['timestamp']);options=[prior['poses'][item]];margin=.004;source='conditional_recent_unplaced_rgb_persistence'
        occupancy[item]=dict(source=source,pose_hypotheses=options,inflation_m=margin,shape='public CAD sphere cover')
    return dict(contract='modular_rgb_scene_belief_v1',target=name,measurement=_observation(observed),occupancy=occupancy,measured_names=sorted(raw),conditional_names=[n for n,v in occupancy.items()if v['source']!='current_rgb_measurement'],board_for_planning=board,belief_ledger=ledger)


def result(route,state,observed,kind,**extra):
    updated=save_observation(state,observed)
    updated.update(last_stage=kind,last_outcome=route)
    return _plain(dict(route=route,state=updated,evidence=dict(kind=kind,timestamp=observed['timestamp'],frame_sha256=observed['frame_sha256'],**extra)))


def observe_and_plan(ctx,state:ModularState)->ModularResult:
    if state['parameters'].get('piece_module'):
        return load_skill_module(state['parameters']['piece_module']).observe_and_plan(ctx,state)
    observed=c.state(ctx);_require(observed,[state['piece']]);state=save_observation(state,observed);p=state['parameters']
    if state['piece']=='red' and p.get('source_contact_grasp',{}).get('module')=='puzzler.modular_red_contact':
        from .red_computed_carry import _repair_prefix_rgb
        state,prefix_reacquisition=_repair_prefix_rgb(ctx,state,label='contact_entry_before_scene')
        if prefix_reacquisition:
            # The newly measured source and its synchronous q16 are one packet;
            # the former raw source remains in the chronological RGB history.
            observed=state['observation'];_require(observed,['board','red'])
            state={**state,'source_entry_prefix_reacquisition':prefix_reacquisition}
    default_arm=0 if observed['poses'][state['piece']][1]>=0 else 1
    arm_candidates=[int(a) for a in p.get('arm_candidates',[default_arm]) if int(a) in (0,1)]
    if not arm_candidates: arm_candidates=[default_arm]
    arm=default_arm
    scene=scene_belief(state,observed);p={**p,'conditional_planning':True,'actual_executed_prefix':list(state['executed_placements'])+list(state.get('released_placements',[])),'planning_prefix':[*state.get('initialized_prefix',[]),*state['executed_placements'],*state.get('released_placements',[])],'given_prefix_names':list(state.get('initialized_prefix',[]))}
    state={**state,'parameters':p,'arm':arm,'planning_scene':scene,'belief_ledger':scene['belief_ledger']}
    if p.get('support_anchor'):
        module=load_skill_module(p['support_anchor']['module']);state['support_anchor_evidence']=module.prepare_support_anchor(ctx,state,observed)
        observed=c.state(ctx);_require(observed,[state['piece']]);state=save_observation(state,observed);scene=scene_belief(state,observed);state['planning_scene']=scene
    preferred_refusal=None;plan=None;planning_errors=[]
    for candidate_arm in arm_candidates:
        try:
            candidate_plan=(load_skill_module(p['geometry_module']).plan_from_rgb(ctx,state,observed)if p.get('geometry_module')else c.plan(ctx,state['piece'],p,candidate_arm,observation=observed,planning_scene=scene))
            arm=candidate_arm;plan=candidate_plan;break
        except (c.PlanningRefusal,RuntimeError) as error:
            planning_errors.append(dict(arm=candidate_arm,error=str(error)))
    if plan is None and p.get('fallback_grasp_parameters'):
        fallback={**p,**p['fallback_grasp_parameters']}
        if 'grasp_indices' not in p['fallback_grasp_parameters']:fallback.pop('grasp_indices',None)
        for candidate_arm in arm_candidates:
            try:
                plan=c.plan(ctx,state['piece'],fallback,candidate_arm,observation=observed,planning_scene=scene);arm=candidate_arm;preferred_refusal=planning_errors;break
            except (c.PlanningRefusal,RuntimeError) as error:
                planning_errors.append(dict(arm=candidate_arm,fallback=True,error=str(error)))
    if plan is None: raise c.PlanningRefusal('No feasible arm/grasp candidate: '+str(planning_errors))
    state['arm']=arm
    if 'source_open_width_m'in plan:
        p={**p,'source_open_width':plan['source_open_width_m'],'open_width':plan['source_open_width_m']};state['parameters']=p
    state['plan']={**plan,'s':_observation(observed),'preferred_geometry_refusal':preferred_refusal}
    return result('planned',state,observed,'perceive_plan',current_source_pose=observed['poses'][state['piece']],board_belief=scene['board_for_planning'])


def grasp_and_bind(ctx,state:ModularState)->ModularResult:
    if state['parameters'].get('piece_module'):
        return load_skill_module(state['parameters']['piece_module']).grasp_and_bind(ctx,state)
    name=state['piece'];p=state['parameters'];arm=state['arm'];plan=state['plan'];g=np.asarray(plan['grasp']);q=plan['rotation'];lift=p.get('lift',.085);attachment=None;attachment_refusal=None
    if p.get('source_contact_grasp',{}).get('module'):
        module=load_skill_module(p['source_contact_grasp']['module']);contact=module.execute_contact_source(ctx,state,state['observation'],plan)
        state={**state,'source_contact_evidence':contact,'memory':contact.get('updated_memory',state['memory']),'belief_ledger':contact.get('updated_belief_ledger',state['belief_ledger'])}
        for packet in contact.get('contact_endpoint',{}).get('source_RGB_epochs',[]):state=save_observation(state,packet)
        # Prefix fits are independent later RGB packets, never merged into
        # the selected RED source epoch. Replay their real chronological
        # evidence before the subsequent post-lift packet.
        for packet in contact.get('contact_endpoint',{}).get('prefix_reacquisition_rgb_packets',[]):state=save_observation(state,packet)
        if contact.get('postlift_rgb'):state=save_observation(state,contact['postlift_rgb'])
        attachment=contact.get('conditional_attachment')
    else:
        c.grip(ctx,arm,p.get('source_open_width',.055));c.move(ctx,g+[0,0,p.get('source_hover',.055)],q,arm);c.move(ctx,g,q,arm,cartesian=True,contact='grasp');c.grip(ctx,arm,p.get('close_width',0.))
        origin=state['observation'];source=dict(piece=name,pose=origin['poses'][name],timestamp=origin['timestamp'],frame_sha256=origin['frame_sha256'],observation_source=origin['observation_source'],role='Actualpregrasp RGB; notapostlift heldmeasurement')
        assumption={**p.get('conditional_attachment_prior',{}),**p.get('conditional_attachment_assumption',{}),'source_stationary_or_bounded_until_close':True,'bead_displacement_bound_m':p.get('conditional_attachment_bound_m',.008)}
        try:attachment=ctx.tool('puzzle.capture_conditional_attachment',source_rgb=source,assumption=assumption,arm_id=arm)
        except ValueError as error:attachment_refusal=str(error)
        c.move(ctx,g+[0,0,lift],q,arm)
    prediction=np.asarray(plan['body']).copy();prediction[2]+=lift
    observed=c.state(ctx,{name:prediction.tolist()})
    inspection_moves=p.get('held_inspection_max_moves',0 if attachment is not None else 1)
    if name not in observed['poses'] and inspection_moves>0:
        try:observed=c.inspect_held(ctx,name,prediction,arm,require_board=False,max_moves=inspection_moves)
        except RuntimeError:observed=c.state(ctx,{name:prediction.tolist()})
    state=save_observation(state,observed)
    if name in observed['poses']:
        tcp,_=c.hand(ctx,arm);rise=observed['poses'][name][2]-plan['body'][2];gap=c.part_hand_gap(name,observed['poses'][name],tcp)
        if rise<.025 or gap>.045:raise StageRefusal('Current RGB contradicts a lifted near-hand grasp')
        q16=observed['robot_geometry_provenance'];source=dict(mode='pyramid',piece=name,pose=observed['poses'][name],timestamp=observed['timestamp'],frame_sha256=observed['frame_sha256'],observation_source=observed['observation_source'],robot_encoder_values=q16['robot_encoder_values'],robot_encoder_sha256=q16['robot_encoder_sha256'],source_step=q16['step'],source_role='Actual accepted currentpostlift RGB and synchronousrobotq16')
        p={**p,'owned_held_belief_source':source};p.pop('conditional_attachment_id',None);measurement_status='measured_postlift_RGB'
    elif attachment is not None:
        p={**p,'conditional_attachment_id':attachment['attachment_id'],'owned_held_belief_source':attachment};rise=None;gap=None;measurement_status='UNKNOWN_conditional_attachment_only'
    else:raise StageRefusal('HeldRGB UNKNOWN and no authenticated conditionalattachment: '+str(attachment_refusal))
    boardrow=prior_for(state,'board',observed['timestamp']);p.update(historical_complete_rgb={k:boardrow[k]for k in ['poses','timestamp','frame_sha256','observation_source']},historical_rgb_records=deepcopy(state['rgb_history']),historical_max_age_s=30.)
    return result('verified',{**state,'parameters':p,'prediction':prediction.tolist(),'conditional_attachment':attachment},observed,'grasp',rise_m=rise,hand_bead_gap_m=gap,held_measurement_status=measurement_status,held_belief_source=p['owned_held_belief_source'],scope='Successfulgrasp requires independentactualphysical CP; conditionalattachment neverclaims measuredheld pose')


def transport(ctx,state:ModularState)->ModularResult:
    if state['parameters'].get('piece_module'):
        return load_skill_module(state['parameters']['piece_module']).execute_transport(ctx,state)
    if state['parameters'].get('transport_module'):
        return load_skill_module(state['parameters']['transport_module']).execute_transport(ctx,state)
    name=state['piece'];p=state['parameters'];arm=state['arm'];belief=(ctx.tool('puzzle.propagate_conditional_attachment',attachment_id=p['conditional_attachment_id'],arm_id=arm)if p.get('conditional_attachment_id')else ctx.tool('puzzle.propagate_pyramid_held_belief',source=p['owned_held_belief_source'],arm_id=arm))
    observed=state['observation'];board=board_belief(state,observed);bp=np.asarray(board['pose']);br=Rotation.from_quat(bp[3:]).as_matrix()
    target=bp[:3]+br@np.asarray(state['plan']['target_center'])
    from .modular_motion import move_held_screened
    carry_scene=scene_belief(state,observed)
    # Source-only planning deliberately deferred the target grasp transform.
    # Re-evaluate exact public CAD symmetries using the actual held transform;
    # an equivalent body frame changes wrist IK, not the physical goal shape.
    frames=[np.asarray(state['plan']['target_R'])];local=model.local_beads(name)
    public_R=np.asarray(goals(p['mode'],p['pyramid_plan'])[name][1])
    for k in range(4):
        for flip in(False,True):
            symmetry=model.rotation(k*np.pi/2)@(np.diag([1.,-1.,-1.])if flip else np.eye(3))
            distances=np.linalg.norm((local@symmetry.T)[:,None]-local[None],axis=2)
            candidate=public_R@symmetry
            if np.max(np.min(distances,axis=1))<1.e-7 and not any(np.allclose(candidate,old)for old in frames):frames.append(candidate)
    target_refusals=[];selected=None
    for frame in frames:
        desired=br@frame;hand_R=desired@np.asarray(belief['object_in_tcp_rotation']).T
        tcp=target-hand_R@np.asarray(belief['object_in_tcp_position']);q=c.qdict(Rotation.from_matrix(hand_R).as_quat())
        # The independent functional contract defines hover in the measured
        # board frame.  Convert that offset through the current board FK
        # instead of adding world-Z, which is wrong for a tilted board.
        hover_height=float(p.get('transport',.1))
        endpoint=tcp+br[:,2]*hover_height
        try:motion=move_held_screened(ctx,state,endpoint,q,belief,carry_scene)
        except c.PlanningRefusal as error:
            target_refusals.append(dict(public_equivalent_target_R=frame.tolist(),refusal=str(error)));continue
        selected=frame;break
    if selected is None:raise c.PlanningRefusal('No actual-held public-CAD equivalent target path: '+str(target_refusals))
    state={**state,'plan':{**state['plan'],'target_R':selected.tolist(),'actual_held_target_refusals':target_refusals,'actual_held_equivalent_target_frames':len(frames)}}
    actual,actual_q=c.hand(ctx,arm);pe=float(np.linalg.norm(actual-endpoint));re=float((Rotation.from_quat(c.qarr(q)).inv()*Rotation.from_quat(c.qarr(actual_q))).magnitude())
    if pe>.0025 or re>.04:raise StageRefusal('Actual robot carry endpoint outside bounds')
    boundary=ctx.tool('robot.execution_journal_cursor')
    state={**state,'belief_ledger':carry_scene['belief_ledger'],'transport_record':dict(source=belief,board_belief=board,target_tcp=endpoint.tolist(),target_rotation=q,transport_offset_frame='measured_board_frame',transport_height_m=hover_height,actual_tcp=actual.tolist(),actual_rotation=actual_q,position_error_m=pe,rotation_error_rad=re,execution_boundary=boundary,robot_proprioception_timestamp=boundary.get('timestamp'),rgb_source_timestamp=observed['timestamp'],motion=motion)}
    return result('carried',state,observed,'transport',transport_record=state['transport_record'])


def carry_and_release(ctx,state:ModularState)->ModularResult:
    # Release guard evidence must come from a current RGB observation.  The
    # old branch forced a historical pre-open image here, which made the
    # target-to-wrist/camera gate UNKNOWN even after a valid held q16/FK
    # propagation.
    p={**state['parameters'],'skip_preopen_rgb':False,'allow_conditional_board_persistence':True,'actual_executed_prefix':list(state['executed_placements'])+list(state.get('released_placements',[])),'planning_prefix':[*state.get('initialized_prefix',[]),*state['executed_placements'],*state.get('released_placements',[])],'given_prefix_names':list(state.get('initialized_prefix',[]))}
    state={**state,'parameters':p}
    out=(load_skill_module(p['release_module']).segment_conditional_release(ctx,state)if p.get('release_module')else segment_conditional_release(ctx,state))
    # Update actual dated RGB history from the real releasepair; no source pose fill.
    state=out['state'];record=state['release_execution'];name=state['piece']
    for observed in record['release_rgb_pair']:state=save_observation(state,observed)
    # Keep an execution token for provenance, but only a visually verified
    # placement may enter the earned/executed ledger.  An uncertain release
    # must go through re-observation and re-planning before another handoff.
    state['execution_tokens'][name]=record
    status='verified_release_pair'if record.get('new_visual_release_receipt')else'executed_unknown'
    if state['belief_ledger'].get(name,{}).get('status')!='current_rgb_conflict_goal_envelope_invalidated':state['belief_ledger'][name]=dict(status=status,conditional_goal_margin_m=.004,placement_verified=record.get('new_visual_release_receipt')is not None)
    pair=record['release_rgb_pair'];errors=[row.get('assessment',{}).get('bead_error_m',{}).get(name)for row in pair]
    if record.get('new_visual_release_receipt'):
        route='placed'
        if name not in state['executed_placements']:state['executed_placements'].append(name)
    elif len(errors)==2 and all(value is not None and value>.008 for value in errors):route='missed'
    else:route='uncertain'
    record['placement_observation_status']=route
    state['release_visual_status']=route
    state['last_outcome']=route;out['route']=route;out['state']=state
    out['evidence']['placement_observation_status']=route
    return _plain(out)


def _current_rgb_fk_gate(observed,name):
    """Require current RGB shape support plus a synchronous q16/FK packet."""
    if not isinstance(observed,dict) or observed.get('observation_source') not in ('rgb_shape_geometry','calibrated_rgb_cad_seeded'):return None
    poses=observed.get('poses',{})
    if not {'board',name}.issubset(poses):return None
    detection=observed.get('detections',{}).get(name,{})
    support=detection.get('independent_rgb_support',{})
    center=np.asarray(detection.get('center_support',[]),dtype=float)
    views=detection.get('views',[])
    by_view=support.get('bead_support_by_view',{}) if isinstance(support,dict) else {}
    bead_counts=np.asarray(support.get('bead_supported_views',[]),dtype=int) if isinstance(support,dict) else np.empty(0,dtype=int)
    if detection.get('visible') is not True or len(views)<2 or center.size<2:return None
    if observed.get('observation_source')=='rgb_shape_geometry':
        if (int(np.sum(center>=.6))<2 or bead_counts.size==0
                or int(np.min(bead_counts))<2 or not by_view):return None
    else:
        # Plain calibrated vision has no shape-identity attribution object;
        # require all four RGB views to support the fitted CAD centres.
        if center.size<4 or int(np.sum(center>=.6))<4:return None
    provenance=observed.get('robot_geometry_provenance',{})
    q16=np.asarray(provenance.get('robot_encoder_values',[]),dtype=np.float32)
    expected=provenance.get('robot_encoder_sha256')
    if q16.shape!=(16,) or not np.isfinite(q16).all() or not isinstance(expected,str):return None
    if hashlib.sha256(q16.tobytes()).hexdigest()!=expected:return None
    pose=np.asarray(poses[name],dtype=float);board=np.asarray(poses['board'],dtype=float)
    if pose.shape!=(7,) or board.shape!=(7,) or not np.isfinite(pose).all() or not np.isfinite(board).all():return None
    return dict(rgb_source=observed.get('observation_source'),visible=True,visible_views=list(views),
                center_support=center.tolist(),independent_bead_supported_views=bead_counts.tolist(),
                bead_support_by_view=deepcopy(by_view),q16_current=True,q16_step=provenance.get('step'),
                q16_timestamp=provenance.get('timestamp'),q16_sha256=expected,
                frame_sha256=deepcopy(observed.get('frame_sha256',{})))


def _cad_hint_fanout(name,board_pose,parameters):
    """Bounded public-CAD orientation seeds; they never become measurements."""
    if board_pose is None:return []
    center,relative=goals(parameters.get('mode','pyramid'),parameters.get('pyramid_plan','pyramid_robot6_plan.json'))[name]
    board=np.asarray(board_pose,dtype=float);br=Rotation.from_quat(board[3:]).as_matrix()
    base=br@np.asarray(relative);target=board[:3]+br@np.asarray(center);fanout=[]
    # MINT is a four-orientation planar pentomino.  Both multiplication
    # conventions are explicit hypotheses for the RGB optimizer.
    for turn in range(4):
        yaw=Rotation.from_euler('z',np.pi/2*turn).as_matrix()
        for side,rotation in (('postmultiply',base@yaw),('premultiply',br@yaw@np.asarray(relative))):
            fanout.append(dict(turn=turn,side=side,target=target.tolist(),
                               hint={name:[*target.tolist(),*Rotation.from_matrix(rotation).as_quat().tolist()]}))
    fanout.sort(key=lambda row:(0 if row['turn']==0 else 1,row['side']))
    return fanout


def reobserve_and_replan(ctx,state:ModularState)->ModularResult:
    """Reacquire an uncertain release with active CAD-hint RGB receipt repair."""
    name=state['piece'];parameters=state['parameters'];observed=None;pair_receipt=None;fanout=[]
    arm=int(state.get('arm',-1))
    # Empty-hand cleanup removes both possible hand occluders before fresh RGB.
    if arm in (0,1):
        ctx.tool('robot.set_grip',arm_id=arm,width_m=.055);ctx.tool('robot.go_home',arm_id=arm)
        support=1-arm;ctx.tool('robot.set_grip',arm_id=support,width_m=.055);ctx.tool('robot.go_home',arm_id=support)
        _pause(ctx,50)
    board_hint=state.get('observation',{}).get('poses',{}).get('board')
    candidates=_cad_hint_fanout(name,board_hint,parameters)
    if not candidates:candidates=[dict(turn=None,side='unhinted',target=None,hint=None)]
    for index,spec in enumerate(candidates):
        first=c.state(ctx,spec['hint']);_require(first,['board']);observed=first;state=save_observation(state,first)
        first_gate=_current_rgb_fk_gate(first,name)
        row=dict(index=index,turn=spec.get('turn'),side=spec.get('side'),hint=spec.get('hint'),
                 first_timestamp=first.get('timestamp'),first_frame_sha256=first.get('frame_sha256'),
                 first_gate=first_gate,first_visible=bool(first_gate))
        # First use the strict shape observer.  If its generic foreign-shape
        # attribution rejects the frame, run the separately labelled plain
        # calibrated RGB/CAD observer on the same public hint.  Both results
        # remain in the fanout record; only a guarded plain pair can mint the
        # active receipt.
        observer_rows=[('shape',first,first_gate)]
        first_goal_error=(visual_memory.goal_error(name,first,parameters)
                          if {'board',name}.issubset(first.get('poses',{})) else None)
        if first_gate is None or (first_goal_error is not None and first_goal_error>=.003):
            plain_hint=dict(spec['hint'] or {});plain_hint['__observer_mode__']='plain_rgb_cad'
            plain_first=c.state(ctx,plain_hint);_require(plain_first,['board']);observed=plain_first;state=save_observation(state,plain_first)
            plain_first_gate=_current_rgb_fk_gate(plain_first,name)
            observer_rows.append(('plain',plain_first,plain_first_gate))
        for mode,initial,initial_gate in observer_rows:
            row[f'{mode}_first_gate']=initial_gate;row[f'{mode}_first_visible']=bool(initial_gate)
            row[f'{mode}_first_timestamp']=initial.get('timestamp');row[f'{mode}_first_frame_sha256']=initial.get('frame_sha256')
            if initial_gate is None:continue
            _pause(ctx,50)
            second_hint={name:list(initial['poses'][name])}
            if mode=='plain':second_hint['__observer_mode__']='plain_rgb_cad'
            second=c.state(ctx,second_hint);_require(second,['board']);observed=second;state=save_observation(state,second)
            second_gate=_current_rgb_fk_gate(second,name)
            row[f'{mode}_second_gate']=second_gate;row[f'{mode}_second_visible']=bool(second_gate)
            row[f'{mode}_second_timestamp']=second.get('timestamp');row[f'{mode}_second_frame_sha256']=second.get('frame_sha256')
            if second_gate is not None:
                try:
                    tcp,_=c.hand(ctx,state['arm']);pair_receipt=visual_memory.release_receipt(name,initial,second,tcp,parameters)
                    row[f'{mode}_receipt']='passed';row['receipt_observer_mode']=mode;break
                except visual_memory.MemoryRefusal as error:row[f'{mode}_receipt_error']=str(error)
        fanout.append(row)
        if pair_receipt is not None:break
        _pause(ctx,10)
    token=deepcopy(state.get('execution_tokens',{}).get(name,{}));receipt=token.get('new_visual_release_receipt')
    measured=pair_receipt is not None
    if measured:
        memory=visual_memory.mint_placement(visual_memory.observe(state['memory'],observed,parameters),pair_receipt,observed,parameters)
        token['new_visual_release_receipt']=pair_receipt
        tokens=deepcopy(state.get('execution_tokens',{}));tokens[name]=token
        executed=list(state.get('executed_placements',[]))
        if name not in executed:executed.append(name)
        ledger=deepcopy(state.get('belief_ledger',{}));ledger[name]=dict(status='verified_release_pair',conditional_goal_margin_m=.004,placement_verified=True)
        state={**state,'memory':memory,'execution_tokens':tokens,'executed_placements':executed,'belief_ledger':ledger}
    pending={'piece':name,'prior_release_token':token,'prior_receipt':receipt,
        'current_rgb_timestamp':observed.get('timestamp'),'current_rgb_frame':observed.get('frame_sha256'),
        'target_3d_status':'measured' if measured else 'UNKNOWN','reobserve_receipt':pair_receipt,
        'replan_required':not measured,'cad_hint_fanout':fanout,
        'scope':'Fresh RGB/proprio observation only; no placement claim.'}
    state={**state,'pending_placement_receipt':pending,'last_stage':'reobserve','last_outcome':'placed' if measured else 'reobserved','attempt':int(state.get('attempt',0))+1}
    route='placed' if measured else 'unknown'
    if not measured:
        pending['replan_required']=False;pending['reobserve_limit_reached']=True
    return _plain(dict(route=route,state=state,evidence=dict(
        kind='reobserve_after_uncertain_release',timestamp=observed.get('timestamp'),frame_sha256=observed.get('frame_sha256'),
        placement_observation_status='measured_pair_receipt' if measured else 'UNKNOWN',receipt=pair_receipt,
        cad_hint_fanout=fanout,prior_release_token_present=bool(token),replan_required=not measured,
        scope='Only a fresh verified RGB pair can commit a real receipt; UNKNOWN preserves unearned status.'
    )))


def handoff_piece(ctx,state:ModularState,expected_piece:str)->ModularResult:
    if state['piece']!=expected_piece or expected_piece not in state['execution_tokens']or state['last_outcome'] not in ('placed','executed_visual_unknown','executed_visual_unverified'):raise StageRefusal('Handoff requires checked release execution and an explicit visual status')
    current=deepcopy(state['observation']);p=state['parameters'];order=p['assembly_order'];index=order.index(expected_piece)
    if index+1<len(order):
        nextpiece=order[index+1];base=p;new=initial_state(nextpiece,base,memory=state['memory'],executed=state['executed_placements'],execution_tokens=state['execution_tokens'],history=state.get('rgb_history',[]),given_prefix=state.get('initialized_prefix',[]));new['belief_ledger']=deepcopy(state['belief_ledger']);new['released_placements']=deepcopy(state.get('released_placements',[]));state=new
    return result('advanced',state,current,'handoff',expected_piece=expected_piece,executed_placements=state['executed_placements'])


def verify_restored_grasp(ctx,state:ModularState)->ModularResult:
    """Test-only: request new physical verification of an exactly restored grasp.

    Historical source remains historical. This function performs no pickup and
    does not itself claim that the preceding failed independent CP passed.
    """
    p=state['parameters']
    if p.get('conditional_attachment_id'):
        current=ctx.tool('puzzle.propagate_conditional_attachment',attachment_id=p['conditional_attachment_id'],arm_id=state['arm'])
        source=current['source'];p={**p,'conditional_attachment_id':source['attachment_id'],'owned_held_belief_source':source}
        state={**state,'parameters':p,'conditional_attachment':source}
    _pause(ctx,6)
    boundary=ctx.tool('robot.execution_journal_cursor')
    return result('verified',state,state['observation'],'grasp',test_initialization_only=True,source_role='Historical ownRGB/restoredgraphstate, not a freshmeasurement',current_robot_boundary=boundary,physical_success='Must be proven by new independent checkpoint')
