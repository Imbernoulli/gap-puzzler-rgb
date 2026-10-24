"""Computed joint-space transport for the live body-relative RED grasp.
Runtime inputs are current RGB, q16/FK, and public CAD/goal geometry only.
"""
import json
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from . import red_arm0_carry as ca, modular_red_placement as base, vision_controller as c, conditional_scene, visual_memory
from .vision_observer import goals
from .modular_controller import result,save_observation,prior_for
from .modular_red_contact import encoder_epoch
from .vision_stages import StageRefusal


def _assembly_corridor(state,scene,board,belief,point,rotation,width,endpoint):
    """Finite public assembled-CAD adjacency, never arbitrary scene contact.

    Existing 6+4 mm uncertainty bands cannot remain disjoint at an intended
    assembly. Only declared near cover pairs of compatible, earned prefix
    hypotheses use nominal penetration during an exact vertical insertion.
    Every other pair retains the complete uncertainty envelope.
    """
    rr=Rotation.from_quat(c.qarr(rotation)).as_matrix();centers,radii=base._target_centers(point,rr,belief)
    obs,sizes,nominal,support,names=base._named_obstacles(state,scene,board,rr)
    br=Rotation.from_quat(board[3:]).as_matrix();local,rad=conditional_scene.local_sphere_cover('red')
    target=board[:3]+br@np.asarray(state['plan']['target_center']);desired=br@np.asarray(state['plan']['target_R']);public_red=local@desired.T+target
    public=goals('pyramid',state['parameters']['pyramid_plan']);mask=np.zeros_like(support)
    for col,name in enumerate(names):
        piece=name['piece'];item=scene['occupancy'][piece];gc,gr=public[piece];goalpose=np.r_[board[:3]+br@gc,Rotation.from_matrix(br@gr).as_quat()]
        map_only=(item['source']=='conditional_executed_goal_MAP_persistence_current_rgb'
            and piece=='maroon' and item.get('conditional_MAP_public_goal_only') is True
            and len(item['pose_hypotheses'])==1)
        valid=(item['source'] in ('conditional_given_stage_prefix_goal_plus_raw','conditional_executed_goal_persistence_plus_current_raw') or map_only) and item['inflation_m']>=.004
        pose=item['pose_hypotheses'][name['hypothesis']]
        if not valid or visual_memory.shape_displacement(piece,goalpose,pose)>(1.e-9 if map_only else .0040001):continue
        other,other_radii=conditional_scene.local_sphere_cover(piece);j=name['cover_index'];otherpoint=(br@gr)@other[j]+goalpose[:3]
        public_gap=np.linalg.norm(public_red-otherpoint,axis=1)-rad-other_radii[j]
        redbeads=base.model.local_beads('red')@desired.T+target;otherbeads=base.model.local_beads(piece)@(br@gr).T+goalpose[:3]
        if not np.any(np.abs(np.linalg.norm(redbeads[:,None]-otherbeads[None],axis=2)-2*base.model.RADIUS)<1.e-6):continue
        delta=public_red-otherpoint;swept=delta.copy();swept[:,2]+=np.clip(-delta[:,2],.003,.036)
        swept_gap=np.linalg.norm(swept,axis=1)-rad-other_radii[j]
        # 10 mm is exactly the conditional held+prefix envelope. The extra
        # .5 mm is its existing requested separation floor.
        mask[:,col]=(public_gap>=-1.e-6)&(swept_gap<=.0105001)
    pair=np.linalg.norm(centers[:,None]-obs[None],axis=2)-radii[:,None]-sizes[None]-.006
    nominal_pair=np.linalg.norm(centers[:,None]-obs[None],axis=2)-radii[:,None]-nominal[None]
    body=point+rr@np.asarray(belief['object_in_tcp_position']);bodyR=rr@np.asarray(belief['object_in_tcp_rotation'])
    on_axis=np.linalg.norm(body[:2]-target[:2])<=.001 and (Rotation.from_matrix(desired).inv()*Rotation.from_matrix(bodyR)).magnitude()<=.04
    bounded_height=-.00025<=float(body[2]-target[2])<=.0365
    allowed=mask if on_axis and bounded_height else np.zeros_like(mask)
    bad=np.where(allowed,nominal_pair<-.001,pair<.0005)
    hand=base._guard(point,rr,width,scene,board,target_centers=centers,target_radii=radii)
    return dict(position=point.tolist(),hand_clearance_m=float(hand['clearance_m']),held_min_envelope_clearance_m=float(pair.min()),held_min_nominal_clearance_m=float(nominal_pair.min()),declared_assembly_cover_pairs=int(allowed.sum()),minimum_unallowed_envelope_m=float(pair[~allowed].min()),minimum_allowed_nominal_m=float(nominal_pair[allowed].min()) if allowed.any() else None,assembly_axis=bool(on_axis),assembly_height_m=float(body[2]-target[2]),feasible=bool(hand['clearance_m']>=.0005 and not bad.any()))


def screen_sequence(ctx,state,scene,board,belief,waypoints,width,descent=False):
    arm=int(state['arm']);path=ctx.tool('puzzle.plan_pose_sequence',waypoints=waypoints,arm_id=arm,min_joint_limit_margin_rad=.035)
    if not path.get('feasible') or path.get('min_joint_limit_margin_rad',0.)<.035:
        raise c.PlanningRefusal('Computed Cartesian sequence reserve refusal '+json.dumps(path.get('errors',[])[-1:],separators=(',',':')))
    checks=[]
    if descent:
        start=np.asarray(path['tcp_path'][0]['position']);end=np.asarray(waypoints[-1]['position'])
        if end[2]>=start[2]-.001 or np.linalg.norm(end[:2]-start[:2])>.001:
            raise c.PlanningRefusal('Assembly corridor requires a strictly downward vertical insertion')
    for sample in path['tcp_path']:
        point=np.asarray(sample['position'])
        row=_assembly_corridor(state,scene,board,belief,point,sample['rotation'],width,np.asarray(waypoints[-1]['position'])) if descent else ca._row(state,scene,board,belief,point,sample['rotation'],width)
        if (not row['feasible']) if descent else (row['hand_clearance_m']<.0005 or row['held_min_envelope_clearance_m']<.0005):
            raise c.PlanningRefusal('Computed Cartesian sampled CAD refusal '+json.dumps(row,separators=(',',':')))
        checks.append(row)
    return dict(endpoint=waypoints[-1]['position'],rotation=waypoints[-1]['rotation'],native_path=path,checks=checks,min_joint_margin_rad=path['min_joint_limit_margin_rad'],descent=descent)


def execute_screened(ctx,state,motion,width,reobserve_on_shrink=False):
    path=motion['native_path'];arm=int(state['arm']);commands=[]
    for a,b in zip(np.asarray(path['path'])[:-1],np.asarray(path['path'])[1:]):
        count=max(1,int(np.ceil(np.max(np.abs(b-a))/.012)))
        commands.extend(dict(positions=(a+(b-a)*u).tolist())for u in np.linspace(0,1,count+1)[1:])
    runs=[]
    for i in range(0,len(commands),8):
        actual=base.proprio(ctx,arm)
        if actual['width_m']<max(.008,width-.0025):raise StageRefusal('Computed carry live jaw retention failed')
        if reobserve_on_shrink and actual['width_m']<width-.002:
            motion.update(executed_segments=runs,actual_after=actual,interrupted_for_reobserve=True,
                          jaw_contraction_m=float(width-actual['width_m']))
            return motion
        motor=ctx.tool('robot.execute_trajectory',trajectory=dict(waypoints=commands[i:i+8]),arm_id=arm,max_steps_per_waypoint=3)
        if motor.get('status')!='ok':raise StageRefusal('Computed carry native motor failure')
        runs.append(motor)
    c.settle_motion(ctx,np.asarray(motion['endpoint']),motion['rotation'],arm,False,.0025)
    motion.update(executed_segments=runs,actual_after=base.proprio(ctx,arm));return motion


def move(ctx,state,scene,board,belief,point,rotation,width,descent=False):
    waypoints=[dict(position=np.asarray(point).tolist(),rotation=rotation)]
    return execute_screened(ctx,state,screen_sequence(ctx,state,scene,board,belief,waypoints,width,descent),width)


def _repair_prefix_rgb(ctx,state,*,label):
    """Resolve a declared prefix-image contradiction by fresh plain RGB only.

    Keep every original raw observation in history. Public goals seed an
    independent calibrated four-view fit; they are never copied as poses.
    """
    from .modular_controller import _current_rgb_fk_gate,board_belief
    names=set(state.get('initialized_prefix',[]))|set(state.get('executed_placements',[]))|set(state.get('released_placements',[]))
    original=state['observation'];raw=original.get('poses',{});parameters=state['parameters']
    conflicts=[name for name in names if state.get('belief_ledger',{}).get(name,{}).get('status')=='current_rgb_conflict_goal_envelope_invalidated' or ('board'in raw and name in raw and visual_memory.goal_error(name,original,parameters)>=.004)]
    if not conflicts:return state,None
    board=np.asarray(board_belief(state,original)['pose']);br=Rotation.from_quat(board[3:]).as_matrix();public=goals('pyramid',parameters['pyramid_plan']);hints={'__observer_mode__':'plain_rgb_cad'}
    # A source-contact scene still needs a real RED/board measurement. This
    # prior seeds the independent fit only and is never copied into its output.
    if 'red' in raw:hints['red']=raw['red']
    for name in conflicts:
        center,rotation=public[name];hints[name]=np.r_[board[:3]+br@center,Rotation.from_matrix(br@rotation).as_quat()].tolist()
    attempts=[]
    for attempt in range(2):
        observed=c.state(ctx,hints);state=save_observation(state,observed);checks={}
        for name in conflicts:
            gate=_current_rgb_fk_gate(observed,name)
            error=visual_memory.goal_error(name,observed,parameters) if gate is not None else None
            checks[name]=dict(current_four_view_q16_gate=gate,actual_RGB_goal_error_m=error)
        accepted=all(v['current_four_view_q16_gate'] is not None and v['actual_RGB_goal_error_m']<.004 for v in checks.values())
        attempts.append(dict(timestamp=observed.get('timestamp'),frame_sha256=observed.get('frame_sha256'),checks=checks,accepted=accepted))
        if accepted:return state,dict(contract='RED_prefix_fresh_plain_RGB_reacquisition_v1',label=label,conflicting_names=conflicts,original_raw_observation=original,attempts=attempts,scope='Independent current RGB and q16 resolves alias; original observations retained, no GT or goal-pose fill')
    # An earned Maroon track may keep a conditional MAP interpretation after
    # these bounded attempts. It is never a new globally unique RGB pose and
    # never makes an initialized fixture into an earned predecessor.
    if set(conflicts)=={'maroon'} or ('maroon' in conflicts and all(
            name=='maroon' or checks[name]['current_four_view_q16_gate'] is not None
            and checks[name]['actual_RGB_goal_error_m']<.004 for name in conflicts)):
        from .prefix_track_reassociation import earned_lineage,validate_record,SOURCE
        cursor=ctx.tool('robot.execution_journal_cursor')
        lineage=earned_lineage(state,cursor)
        if lineage is not None:
            old_poses=[]
            for packet in state.get('rgb_history',[]):
                if packet.get('timestamp',-1.)>=lineage['original_receipt']['after']['timestamp'] and 'maroon' in packet.get('poses',{}):
                    pose=packet['poses']['maroon']
                    if not any(visual_memory.shape_displacement('maroon',pose,old)<.003 for old in old_poses):old_poses.append(pose)
            request={'receipt_pose':lineage['original_receipt']['after']['poses']['maroon'],
                     'conflicting_poses':old_poses}
            map_hints={'__observer_mode__':'prefix_rgb_cad',
                '__conditional_prefix_tracks__':{'maroon':request}}
            if 'red' in raw:map_hints['red']=raw['red']
            current=c.state(ctx,map_hints)
            witness=current.get('prefix_track_reassociation',{}).get('maroon',{})
            board_now=np.asarray(current.get('poses',{}).get('board',[]))
            if board_now.shape==(7,):
                rb=Rotation.from_quat(board_now[3:]).as_matrix();gc,gr=public['maroon']
                goal_now=np.r_[board_now[:3]+rb@gc,Rotation.from_matrix(rb@gr).as_quat()]
                if validate_record('maroon',witness,current,goal_now):
                    state=save_observation(state,current)
                    ledger=dict(state['belief_ledger']);previous=ledger.get('maroon',{})
                    ledger['maroon']={**previous,'status':'conditional_MAP_track_persistence',
                        'conditional_goal_margin_m':.004,'conditional_MAP_reassociation':witness,
                        'conditional_MAP_lineage':lineage,'prior_invalidation':previous,
                        'raw_conflicts_retained':True,'new_visual_receipt_earned':False}
                    memory=deepcopy(state['memory'])
                    track=memory['placements']['maroon']
                    track.update(status='conditional_MAP_track_persistence',
                        prior_invalidation_reason=track.get('invalidation_reason',''),
                        conditional_MAP_reassociation=witness)
                    state={**state,'belief_ledger':ledger,'memory':memory}
                    attempts.append(dict(timestamp=current['timestamp'],frame_sha256=current['frame_sha256'],
                        accepted=False,conditional_MAP_persistence=True,
                        globally_unique_measurement=False,new_visual_receipt_earned=False))
                    return state,dict(contract='RED_earned_Maroon_conditional_MAP_prefix_v1',label=label,
                        conflicting_names=conflicts,original_raw_observation=original,attempts=attempts,
                        conditional_MAP_witness=witness,lineage=lineage,source=SOURCE,
                        scope='Current pixels and prior executed release promise; other aliases retained, no new pose/receipt or physical guarantee')
    raise StageRefusal('RED current prefix RGB conflict remains after bounded independent plain fits '+json.dumps(attempts,separators=(',',':')))


def _refresh_held_rgb(ctx,state,belief,*,label):
    """Rebind only a newly accepted RGB body and its SAME-image native q16.

    Predictions are optimizer hints, never returned as measured poses. A
    missed view permits at most one 6-mm screened outside-table translation;
    no fixed wrist fan, object-state oracle, or assumed goal pose is used.
    """
    arm=int(state['arm']);attempts=[];prior=state['parameters']['owned_held_belief_source']
    for attempt in range(3):
        predicted=ctx.tool('puzzle.propagate_pyramid_held_belief',source=prior,arm_id=arm)
        observed=c.state(ctx,{'red':predicted['predicted_body_pose']})
        row=dict(attempt=attempt,timestamp=observed.get('timestamp'),frame_sha256=observed.get('frame_sha256'),
                 visible_names=sorted(observed.get('poses',{})))
        if 'red' in observed.get('poses',{}):
            epoch=encoder_epoch(observed);actual=base.proprio(ctx,arm)
            gap=c.part_hand_gap('red',np.asarray(observed['poses']['red']),np.asarray(actual['tcp'][:3]))
            innovation=visual_memory.shape_displacement('red',predicted['predicted_body_pose'],observed['poses']['red'])
            if observed['timestamp']<=prior['timestamp'] or actual['width_m']<.008 or gap>.045 or innovation>.020:
                raise StageRefusal('Postrotation RGB contradicts a retained near-hand RED '+json.dumps(dict(gap_m=gap,innovation_m=innovation)))
            source=dict(mode='pyramid',piece='red',pose=observed['poses']['red'],timestamp=observed['timestamp'],
                frame_sha256=observed['frame_sha256'],observation_source=observed['observation_source'],
                robot_encoder_values=epoch['values'],robot_encoder_sha256=epoch['sha256'],source_step=epoch['step'],
                source_role='Actual accepted postrotation outside RGB and its same-image q16; refreshed attachment, never goal fill')
            state=save_observation(state,observed)
            # A current Red/same-q16 packet is sufficient to bind the body.
            # Board evidence remains a separately dated real RGB record.
            if 'board' not in observed['poses']:
                board_view=c.state(ctx,{'__observer_mode__':'plain_rgb_cad'})
                state=save_observation(state,board_view)
                row['separate_board_observation']=board_view
            board_packet=prior_for(state,'board',state['observation']['timestamp'],max_age=30.)
            row['board_source_timestamp']=board_packet['timestamp']
            row['board_current_measurement']='board' in state['observation'].get('poses',{})
            row['board_persistence_bound_m']=0. if row['board_current_measurement'] else .004
            parameters={**state['parameters'],'owned_held_belief_source':source,
                'historical_complete_rgb':{k:board_packet[k]for k in ('poses','timestamp','frame_sha256','observation_source')},
                'historical_rgb_records':state['rgb_history'],'historical_max_age_s':30.}
            parameters.pop('conditional_attachment_id',None);state={**state,'parameters':parameters}
            fresh=ctx.tool('puzzle.propagate_pyramid_held_belief',source=source,arm_id=arm)
            row.update(status='new_RGB_q16_attachment',actual_proprio=actual,hand_bead_gap_m=gap,
                       measured_vs_old_prediction_m=innovation,source=source,new_held_belief=fresh)
            attempts.append(row)
            return state,fresh,dict(contract='RED_postrotation_RGB_q16_rebind_v1',label=label,attempts=attempts)
        row['status']='RGB_UNKNOWN';attempts.append(row)
        if attempt==1:
            # Move only within the already-clear outside corridor. Every FK
            # sample retains the original CAD, limits and held envelope.
            scene=base.scene_belief(state,state['observation']);board=np.asarray(scene['board_for_planning']['pose']);br=Rotation.from_quat(board[3:]).as_matrix();cur=base.proprio(ctx,arm)
            target=np.asarray(cur['tcp'][:3])+br@np.array([0.,-.006,0.])
            try:row['bounded_view_motion']=move(ctx,state,scene,board,predicted,target,c.qdict(cur['tcp'][3:]),float(cur['width_m']))
            except c.PlanningRefusal as error:row['bounded_view_refusal']=str(error)
    raise StageRefusal('Postrotation RGB UNKNOWN after bounded screened observations '+json.dumps(attempts,separators=(',',':')))


def _ensure_current_board_rgb(ctx,state,belief):
    """Bounded outside looks; only real current board RGB removes persistence.

    The historical board remains a dated four-millimetre planning belief until
    an actual observer packet contains the board. Every look is independently
    screened, and every accepted RED source is rebound to its own image q16.
    """
    if 'board' in state['observation'].get('poses',{}):
        return state,belief,dict(status='current_board_already_measured',timestamp=state['observation']['timestamp'],attempts=[])
    arm=int(state['arm']);origin=base.proprio(ctx,arm);attempts=[]
    for distance in (.020,.040,.060):
        scene=base.scene_belief(state,state['observation']);board=np.asarray(scene['board_for_planning']['pose']);br=Rotation.from_quat(board[3:]).as_matrix();cur=base.proprio(ctx,arm)
        point=np.asarray(origin['tcp'][:3])+br@np.array([0.,-distance,0.])
        row=dict(outside_offset_y_m=-distance)
        waypoint=dict(position=point.tolist(),rotation=c.qdict(cur['tcp'][3:]))
        try:
            screened=screen_sequence(ctx,state,scene,board,belief,[waypoint],float(cur['width_m']))
        except c.PlanningRefusal as error:
            row.update(status='refused_before_motion',refusal=str(error));attempts.append(row);continue
        # An execution/settle error propagates, even if it is a PlanningRefusal.
        row['screened_motion']=execute_screened(ctx,state,screened,float(cur['width_m']))
        state,belief,rebound=_refresh_held_rgb(ctx,state,belief,label='current_board_outside_look')
        state,prefix_fix=_repair_prefix_rgb(ctx,state,label='current_board_outside_look_prefix')
        row.update(held_RGB_rebinding=rebound,prefix_RGB_reacquisition=prefix_fix,
                   observation_timestamp=state['observation']['timestamp'],current_board_measured='board' in state['observation'].get('poses',{}))
        attempts.append(row)
        if row['current_board_measured']:
            return state,belief,dict(status='fresh_current_board_RGB',timestamp=state['observation']['timestamp'],attempts=attempts)
    raise StageRefusal('Current board RGB UNKNOWN after bounded screened outside looks '+json.dumps(attempts,separators=(',',':')))


def carry(ctx,state):
    arm=int(state['arm']);p=state['parameters'];belief=ctx.tool('puzzle.propagate_pyramid_held_belief',source=p['owned_held_belief_source'],arm_id=arm)
    initial_belief=belief;prefix_reacquisitions=[]
    state,prefix_fix=_repair_prefix_rgb(ctx,state,label='pre_carry')
    if prefix_fix:prefix_reacquisitions.append(prefix_fix)
    scene=base.scene_belief(state,state['observation']);board=np.asarray(scene['board_for_planning']['pose']);br=Rotation.from_quat(board[3:]).as_matrix();cur=base.proprio(ctx,arm)
    selected,candidates=base.choose_target(ctx,state,belief,scene,cur,for_transport=True)
    state={**state,'plan':{**state['plan'],'target_R':selected['target_R'],'target_center':selected['target_center'],'red_target_selection':selected,'red_target_candidates':candidates}}
    relp=np.asarray(belief['object_in_tcp_position']);relR=np.asarray(belief['object_in_tcp_rotation']);handR=Rotation.from_quat(cur['tcp'][3:]).as_matrix();body=np.asarray(cur['tcp'][:3])+handR@relp;bodyR=handR@relR
    target=board[:3]+br@np.asarray(selected['target_center']);goalR=br@np.asarray(selected['target_R']);width=float(cur['width_m']);refusals=[];screened=None;motions=[];rebindings=[]
    def waypoint(center,orientation,relative_p,relative_R):
        hr=orientation@relative_R.T
        return dict(position=(center-hr@relative_p).tolist(),rotation=c.qdict(Rotation.from_matrix(hr).as_quat()))
    # Keep the solved source/outside/rotate geometry. Stop OUTSIDE before the
    # last ingress so gravity/contact-induced attachment changes are measured.
    for height,side in ((.036,-.127),(.036,-.167),(.035,-.127)):
        raised=body+[0,0,.035];outer=target+br@np.array([0.,side,0.]);outer[2]=raised[2];turned=outer.copy();turned[2]=target[2]+height
        waypoints=[waypoint(raised,bodyR,relp,relR),waypoint(outer,bodyR,relp,relR),waypoint(turned,goalR,relp,relR)]
        try:screened=screen_sequence(ctx,state,scene,board,belief,waypoints,width);break
        except c.PlanningRefusal as error:refusals.append(str(error))
    if screened is None:raise c.PlanningRefusal('Computed outside Cartesian routes refused '+json.dumps(refusals,separators=(',',':')))
    outside_motion=execute_screened(ctx,state,screened,width,reobserve_on_shrink=True);motions.append(outside_motion)
    state,belief,rebound=_refresh_held_rgb(ctx,state,belief,label='outside_rotation_or_jaw_contraction');rebindings.append(rebound)
    state,prefix_fix=_repair_prefix_rgb(ctx,state,label='postrotation_current_prefix')
    if prefix_fix:prefix_reacquisitions.append(prefix_fix)
    if outside_motion.get('interrupted_for_reobserve'):
        scene=base.scene_belief(state,state['observation']);board=np.asarray(scene['board_for_planning']['pose']);br=Rotation.from_quat(board[3:]).as_matrix();goalR=br@np.asarray(state['plan']['target_R']);target=board[:3]+br@np.asarray(state['plan']['target_center']);outside=target+br@np.array([0.,side,0.]);outside[2]=target[2]+height
        endpoint=waypoint(outside,goalR,np.asarray(belief['object_in_tcp_position']),np.asarray(belief['object_in_tcp_rotation']));cur=base.proprio(ctx,arm)
        motions.append(move(ctx,state,scene,board,belief,np.asarray(endpoint['position']),endpoint['rotation'],float(cur['width_m'])))
        state,belief,rebound=_refresh_held_rgb(ctx,state,belief,label='outside_rotation_completed');rebindings.append(rebound)
        state,prefix_fix=_repair_prefix_rgb(ctx,state,label='rotation_completed_current_prefix')
        if prefix_fix:prefix_reacquisitions.append(prefix_fix)
    # The final target is recomputed from refreshed actual held geometry and a
    # newly observed board; no source-frame TCP or recorded world pose remains.
    state,belief,current_board_acquisition=_ensure_current_board_rgb(ctx,state,belief)
    scene=base.scene_belief(state,state['observation']);board=np.asarray(scene['board_for_planning']['pose']);br=Rotation.from_quat(board[3:]).as_matrix();cur=base.proprio(ctx,arm)
    selected,candidates=base.choose_target(ctx,state,belief,scene,cur,for_transport=True)
    target=board[:3]+br@np.asarray(selected['target_center']);goalR=br@np.asarray(selected['target_R'])
    # One small ingress tilt, separately recorded from the true assembly goal.
    # The public intermediate body point avoids the retained prefix envelope;
    # every actual FK sample is screened with the original .035 reserve.
    nominal_hover=float(p['transport']);tilt_angle=.038
    tilted_goal=Rotation.from_rotvec([tilt_angle,0.,0.]).as_matrix()@goalR
    local=base.model.local_beads('red')
    public_bead_offset=float(np.linalg.norm(local@(tilted_goal-goalR).T,axis=1).max())
    if public_bead_offset>.0055:raise c.PlanningRefusal('Ingress tilt exceeds public nominal-hover proposal envelope')
    relative_p=np.asarray(belief['object_in_tcp_position']);relative_R=np.asarray(belief['object_in_tcp_rotation'])
    intermediate=target+br@np.array([0.,-.040,0.])+[0,0,.042]
    endpoint=waypoint(target+[0,0,nominal_hover],tilted_goal,relative_p,relative_R)
    ingress_waypoints=[waypoint(intermediate,tilted_goal,relative_p,relative_R),endpoint]
    ingress=screen_sequence(ctx,state,scene,board,belief,ingress_waypoints,float(cur['width_m']))
    motions.append(execute_screened(ctx,state,ingress,float(cur['width_m'])))
    adjustment=dict(contract='RED_small_tilt_ingress_true_goal_retained_v1',world_rotation_vector_rad=[tilt_angle,0.,0.],nominal_transport_hover_m=nominal_hover,public_max_bead_offset_m=public_bead_offset,intermediate_body_board_y_m=-.040,intermediate_body_hover_m=.042,true_goal_R_retained=True)
    selected={**selected,'hover_tcp':endpoint['position'],'rotation':endpoint['rotation'],'hover_body_height_m':nominal_hover,'ingress_pose_adjustment':adjustment}
    state={**state,'plan':{**state['plan'],'target_R':selected['target_R'],'target_center':selected['target_center'],'red_target_selection':selected,'red_target_candidates':candidates}}
    current=base.proprio(ctx,arm);record=dict(contract='RED_arm_aware_low_segmented_carry_v3',implementation='computed_current_board_RGB_small_tilt_ingress_v1',arm=arm,source_belief=initial_belief,postrotation_belief=belief,postrotation_RGB_rebindings=rebindings,prefix_RGB_reacquisitions=prefix_reacquisitions,current_board_acquisition=current_board_acquisition,nominal_transport_hover_m=nominal_hover,selected_ingress_hover_m=nominal_hover,ingress_pose_adjustment=adjustment,selected_source_candidate={**state['plan'].get('selected_contact_candidate',{}),'arm':arm},selected_target=selected,segments=motions,refusals=refusals,actual_robot=current,reserve_rad=.035)
    state['transport_record']=record
    return result('carried',state,state['observation'],'transport',transport_record=record)

def release(ctx,state):
    # Reuse the measured release/withdraw/receipt contract, replacing only its
    # path solver and selecting the lowest actual-width sweep-clear height.
    arm=int(state['arm']);p=state['parameters'];belief=ctx.tool('puzzle.propagate_pyramid_held_belief',source=p['owned_held_belief_source'],arm_id=arm);scene=base.scene_belief(state,state['observation']);board=np.asarray(scene['board_for_planning']['pose']);br=Rotation.from_quat(board[3:]).as_matrix();desired=br@np.asarray(state['plan']['target_R']);target=board[:3]+br@np.asarray(state['plan']['target_center']);rr=desired@np.asarray(belief['object_in_tcp_rotation']).T;zero=target-rr@np.asarray(belief['object_in_tcp_position']);cur=base.proprio(ctx,arm);width=float(cur['width_m']);ladder=[];height=None
    for h in np.arange(.003,.031,.001):
        point=zero+[0,0,h];centers,radii=base._target_centers(point,rr,belief);guards=[base._guard(point,rr,float(w),scene,board,target_centers=centers,target_radii=radii)for w in np.linspace(width,.09,25)];gap=min(x['clearance_m']for x in guards);ladder.append(dict(height_m=float(h),minimum_opening_clearance_m=float(gap)))
        if gap>=.0005:height=float(h);break
    if height is None:raise c.PlanningRefusal('Computed release no live-jaw opening height '+json.dumps(ladder))
    # Restore the actual true goal in two original corridor moves. Splitting
    # the small tilt keeps each TCP lateral change within the existing 1mm
    # descent gate; goal_R and the original opening-height ladder are retained.
    alignment=[];adjustment=state.get('transport_record',{}).get('ingress_pose_adjustment')
    if adjustment is not None:
        if 'board' not in state['observation'].get('poses',{}):raise StageRefusal('Current board RGB required before tilted release alignment')
        angle=float(adjustment['world_rotation_vector_rad'][0]);half_goal=Rotation.from_rotvec([angle/2.,0.,0.]).as_matrix()@desired
        half_R=half_goal@np.asarray(belief['object_in_tcp_rotation']).T
        half_point=target+[0,0,.026]-half_R@np.asarray(belief['object_in_tcp_position'])
        alignment.append(move(ctx,state,scene,board,belief,half_point,c.qdict(Rotation.from_matrix(half_R).as_quat()),width,descent=True))
        alignment.append(move(ctx,state,scene,board,belief,zero+[0,0,.024],c.qdict(Rotation.from_matrix(rr).as_quat()),width,descent=True))
    original=ca._exec
    def adapted(ctx2,state2,scene2,board2,belief2,endpoint,rotation,width2,descent=False):
        # Original release requested 3 mm; replace with live computed height.
        request=np.asarray(endpoint).copy()
        if descent:request=zero+[0,0,height]
        return move(ctx2,state2,scene2,board2,belief2,request,rotation,width2,descent=descent)
    ca._exec=adapted
    try:
        out=ca.release(ctx,state);out['state']['release_execution']['computed_release_height_m']=height;out['state']['release_execution']['computed_height_ladder']=ladder;out['state']['release_execution']['true_goal_alignment']=alignment;return out
    finally:ca._exec=original

# Uniform modular graph skill API; implementation and dataflow are unchanged.
execute_transport = carry
segment_conditional_release = release
