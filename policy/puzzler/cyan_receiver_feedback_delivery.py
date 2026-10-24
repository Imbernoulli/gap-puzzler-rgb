"""Receiver-arm Cyan delivery after independently qualified handoff.

Current RGB is preferred. An authenticated six-mm conditional close binding
may control while the target is occluded, without filling measured poses.
Final physical release and placement receipt remain independent checkpoints.
The receiver first changes its actual view while the existing scene is still
valid. Donor homing follows an actual renewed environment observation.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation

from . import cyan_geometry as g,cyan_original_transport as tr,cyan_inverse_tip as tip
from . import vision_controller as c,visual_memory,model
from .release_admission import check_bounded_release
from .modular_controller import save_observation,result
from .finite_release_scene import finite_clearance
from .cyan_bimanual_adapter import _qpath,_width_rows,_geometry
from .cyan_receiver_motion import move_held,_material_geometry,require_board_margin


class _ReceiverSceneContext:
    def __init__(self,ctx,state):
        self.ctx=ctx
        self.limit=min(30.,float(state['parameters'].get('historical_max_age_s',30.)))
        if not np.isfinite(self.limit) or self.limit<=0:
            raise c.PlanningRefusal('Receiver delivery needs the original finite scene age scope')
        scene=state['planning_scene']
        if 'reference' in scene:self.bind_delivery_scene(state,scene)
        else:
            self.reference=deepcopy(state['parameters']['historical_complete_rgb'])
            measurement=scene['measurement']
            if (measurement.get('timestamp')!=self.reference.get('timestamp')
                    or measurement.get('frame_sha256')!=self.reference.get('frame_sha256')):
                raise c.PlanningRefusal('Receiver initial scene and dated RGB epochs disagree')

    def __getattr__(self,name):return getattr(self.ctx,name)

    def bind_delivery_scene(self,state,scene):
        reference=scene['reference']
        packet=(scene['measurement'] if reference['mode']=='current_complete_reference'
                else state['parameters']['historical_complete_rgb'])
        if (reference['reference_timestamp']!=packet.get('timestamp')
                or reference['reference_frame_sha256']!=packet.get('frame_sha256')):
            raise c.PlanningRefusal('Receiver scene must retain its authentic RGB reference epoch')
        self.reference=deepcopy(packet)

    def tool(self,name,**kwargs):
        if name in ('robot.execute_trajectory','robot.set_grip'):
            now=self.ctx.tool('robot.execution_journal_cursor').get('timestamp')
            stamp=self.reference.get('timestamp')
            if (not isinstance(now,(int,float)) or not np.isfinite(now)
                    or not isinstance(stamp,(int,float)) or not np.isfinite(stamp)
                    or not 0<=now-stamp<=self.limit
                    or not self.reference.get('frame_sha256')
                    or self.reference.get('observation_source')!='rgb_shape_geometry'):
                raise c.PlanningRefusal('Receiver motor scene exceeded its original30s actual RGB scope; reobserve and rescreen')
        return self.ctx.tool(name,**kwargs)


def _selected_bead(state,belief):
    centers=model.local_beads('cyan')@np.asarray(belief['object_in_tcp_rotation']).T+np.asarray(belief['object_in_tcp_position'])
    # Match the original physical selected bead through a possible exact CAD
    # frame symmetry. Nearest to TCP is wrong for an oblique grasp: the elbow
    # bead can be nearer the TCP while sitting behind the finger contact zone.
    capture=state['transfer']['receiver_conditional_attachment']
    chosen=int(state['transfer']['candidate']['bead_index'])
    reference=np.asarray(capture['object_in_tcp_position'])+np.asarray(capture['object_in_tcp_rotation'])@model.local_beads('cyan')[chosen]
    distances=np.linalg.norm(centers-reference,axis=1)
    index=int(np.argmin(distances))
    if distances[index]>.006000001:
        raise c.PlanningRefusal('Current receiver selected-bead binding contradicts the original6mm transfer bound')
    return index


def _refresh(ctx,state):
    belief=tr._belief(ctx,state)
    observed=c.state(ctx,{'cyan':belief['predicted_body_pose']})
    admission=check_bounded_release('cyan',belief,observed,state['parameters'])
    state=save_observation(state,observed)
    if 'cyan' in observed.get('poses',{}):
        state['parameters']['owned_held_belief_source']=g._source(observed,'cyan')
        state['parameters'].pop('conditional_attachment_id',None)
        state['held_status']='CURRENT_RGB_VERIFIED'
        belief=tr._belief(ctx,state)
        # Refresh a complete real packet only; never paste an occluded target
        # prediction or a public goal into an RGB record to renew scene age.
        required={'board',*(set(model.CELLS)-set(state['parameters']['conditional_prefix']))}
        if required.issubset(observed['poses']):
            state['parameters']['historical_complete_rgb']=g.rgb_record(observed)
    else:
        state['held_status']='UNKNOWN_dated_RGB_or_conditional_held_belief'
    scene,board,margin=tip._release_scene(state,observed)
    state['planning_scene']=scene
    if hasattr(ctx,'bind_delivery_scene'):ctx.bind_delivery_scene(state,scene)
    state.setdefault('receiver_delivery',{})['selected_receiver_bead']=_selected_bead(state,belief)
    return state,observed,scene,belief,dict(admission=admission,current_target_measured='cyan' in observed.get('poses',{}),
        source=deepcopy(belief['source']),observation=g.rgb_record(observed))


def _board_goal(state,scene,relative):
    board=np.asarray(scene['board_for_planning']['pose']);Rb=Rotation.from_quat(board[3:]).as_matrix()
    center,_=g.goals('pyramid',state['parameters']['pyramid_plan'])['cyan']
    return board,Rb,board[:3]+Rb@center,Rb@np.asarray(relative)


def _hover(state,scene,belief,relative):
    board,Rb,center,body_R=_board_goal(state,scene,relative)
    hand_R=body_R@np.asarray(belief['object_in_tcp_rotation']).T
    body_p=center+Rb[:,2]*float(state['parameters']['transport'])
    tcp=body_p-hand_R@np.asarray(belief['object_in_tcp_position'])
    desired=np.r_[body_p,Rotation.from_matrix(body_R).as_quat()]
    return dict(hover=tcp.tolist(),rotation=c.qdict(Rotation.from_matrix(hand_R).as_quat()),
        relative_target_R=np.asarray(relative).tolist(),target_tcp=(tcp-Rb[:,2]*state['parameters']['transport']).tolist(),
        authored_bound_estimate_public_hover_error_m=float(visual_memory.shape_displacement('cyan',belief['predicted_body_pose'],desired)),
        current_target_measured='cyan' in state.get('observation',{}).get('poses',{}))


def _select_target(ctx,state,scene,belief,*,lock_relative=False):
    planned={**state,'parameters':{**state['parameters'],'current_board_for_inverse_planning':deepcopy(scene['board_for_planning'])}}
    _,choices,_=g.carry_target(ctx,planned)
    arm=int(state['arm']);off=0 if arm==0 else 8
    width=float(belief['current_robot_encoder_values'][off+6]-belief['current_robot_encoder_values'][off+7])
    feasible=[]
    for choice in choices:
        if lock_relative and not np.allclose(choice['relative_target_R'],state['plan']['target_R']):continue
        R=Rotation.from_quat(c.qarr(choice['rotation'])).as_matrix()
        for height in (.003,.006,.010):
            if height>float(state['parameters'].get('max_release_height',.010)):continue
            pos=np.asarray(choice['target_tcp'])+[0.,0.,height]
            sweep=[finite_clearance(pos,R,float(w),scene,exclude=('cyan',)) for w in np.linspace(width,.055,12)]
            if min(row['clearance_m'] for row in sweep)<.0005:continue
            ik=ctx.tool('puzzle.plan_pose',position=pos.tolist(),rotation=choice['rotation'],arm_id=arm,min_joint_limit_margin_rad=.035)
            if ik['position_error_m']>.002 or ik['rotation_error_rad']>.03 or ik['joint_limit_margin_rad']<.035:continue
            feasible.append({**choice,'height':height,'release_tcp':pos.tolist(),
                'release_ik':ik,'opening_sweep':sweep,'opening_width_range_m':[width,.055]})
            break
    if not feasible:raise c.PlanningRefusal('Actual receiver attachment has no original-margin 3–10mm release')
    return min(feasible,key=lambda row:(row['height'],-row['release_ik']['joint_limit_margin_rad']))


def _stream(ctx,state,arm,full):
    off=0 if arm==0 else 8
    records=[]
    for start in range(1,len(full),8):
        tr._belief(ctx,state)
        output=ctx.tool('robot.execute_trajectory',trajectory=dict(waypoints=[dict(positions=row[off:off+6].tolist())
            for row in full[start:start+8]]),arm_id=arm,max_steps_per_waypoint=2)
        if output.get('status')!='ok':raise RuntimeError('Receiver delivery motor refused')
        records.append(output)
    return records


def _home_donor_while_held(ctx,state,scene,belief):
    arm=int(state['transfer']['sender']);owner=int(state['arm'])
    geometry=_geometry();position,quaternion=geometry.robot.iks[geometry._arm(arm)].home_pose
    rotation=c.qdict(quaternion)
    plan=ctx.tool('puzzle.plan_joint_path',position=np.asarray(position).tolist(),rotation=rotation,arm_id=arm,min_joint_limit_margin_rad=.035)
    if not plan.get('feasible'):raise c.PlanningRefusal('Empty donor public-home IK refused')
    full=_qpath(plan,belief['current_robot_encoder_values'],arm)
    proof=geometry.screen_q16_path(full,scene,moving_arm=arm,held_arm=owner,
        held_body_pose=belief['predicted_body_pose'],object_in_tcp_position=belief['object_in_tcp_position'],
        object_in_tcp_rotation=belief['object_in_tcp_rotation'],target='cyan',
        finger_target_contract={owner:[_selected_bead(state,belief)]},
        held_envelope_m=max(.006,float(belief['uncertainty_m'])))
    if not proof.get('safe'):raise c.PlanningRefusal('Donor public-home full geometry refused: '+str(proof))
    require_board_margin(proof,scene,held=True)
    motor=_stream(ctx,state,arm,full)
    c.await_motion(ctx,position,rotation,arm,accuracy=.0025)
    return dict(native_ik=plan,complete_geometry=proof,motor=motor)


def _execute_transport(ctx,state):
    state=deepcopy(state)
    if state['piece']!='cyan' or state['arm']!=state['transfer']['receiver']:
        raise c.PlanningRefusal('Receiver delivery requires the qualified receiver owner')
    state,observed,scene,belief,source=_refresh(ctx,state)
    # The original donor home consumed15s while the receiver hid board/lime.
    # Move the held receiver first, using BOTH actual arms and the unchanged
    # old-scene age/4mm persistence envelope; then take a genuine new packet.
    arm=int(state['arm']);off=0 if arm==0 else 8;view_steps=[]
    scene_current=scene['reference']['reference_timestamp']==observed['timestamp']
    for delta in ([0.,-.030,.020],[0.,-.030,0.],[0.,-.030,0.]):
        if scene_current:break
        width=float(belief['current_robot_encoder_values'][off+6]-belief['current_robot_encoder_values'][off+7])
        tcp,rotation=c.hand(ctx,arm)
        motion=move_held(ctx,state,tcp+delta,rotation,scene,width,belief=belief,joint=False)
        state,observed,scene,belief,view_source=_refresh(ctx,state)
        # A successful refresh can still be honest bounded persistence. It
        # cannot license the long donor home unless this REAL packet renewed
        # all required occupancy, with every missing datum still dated.
        scene_current=scene['reference']['reference_timestamp']==observed['timestamp']
        view_steps.append(dict(relative_world_proposal_m=delta,motion=motion,
            actual_observation=view_source,actual_environment_reference_renewed=scene_current))
    if not scene_current:
        raise c.PlanningRefusal('Actual bounded receiver views did not renew required environment RGB; donor remains withdrawn, do not home on old occupancy')
    donor_home=_home_donor_while_held(ctx,state,scene,belief)
    state,observed,scene,belief,source2=_refresh(ctx,state)
    target=_select_target(ctx,state,scene,belief);relative=target['relative_target_R']
    state['plan']['target_R']=relative
    arm=int(state['arm']);off=0 if arm==0 else 8
    width=float(belief['current_robot_encoder_values'][off+6]-belief['current_robot_encoder_values'][off+7])
    first=move_held(ctx,state,target['hover'],target['rotation'],scene,width,belief=belief,joint=True,path_fraction=.5)
    state,middle,scene,belief,middle_source=_refresh(ctx,state)
    target=_hover(state,scene,belief,relative)
    width=float(belief['current_robot_encoder_values'][off+6]-belief['current_robot_encoder_values'][off+7])
    second=move_held(ctx,state,target['hover'],target['rotation'],scene,width,belief=belief,joint=True)
    feedback=[];corrections=[]
    for attempt in range(3):
        ctx.tool('robot.wait_steps',steps=6)
        state,observed,scene,belief,entry=_refresh(ctx,state)
        target=_hover(state,scene,belief,relative)
        feedback.append(dict(source=entry,target=target))
        if target['authored_bound_estimate_public_hover_error_m']<=.003:break
        if attempt==2:raise c.PlanningRefusal('Receiver bounded hover corrections exhausted')
        tcp=np.asarray(belief['current_tcp'])
        distance=float(np.linalg.norm(np.asarray(target['hover'])-tcp[:3]))
        angle=float((Rotation.from_quat(tcp[3:]).inv()*Rotation.from_quat(c.qarr(target['rotation']))).magnitude())
        if distance>.012 or angle>.25:raise c.PlanningRefusal('Receiver correction exceeds original12mm/.25rad bounds')
        width=float(belief['current_robot_encoder_values'][off+6]-belief['current_robot_encoder_values'][off+7])
        corrections.append(move_held(ctx,state,target['hover'],target['rotation'],scene,width,belief=belief,joint=True))
    carry=dict(contract='receiver_owned_RGB_or_honest_conditional_segmented_carry_v1',
        receiver_first_actual_view_steps=view_steps,
        donor_screened_public_home=donor_home,segments=[first,second],sources=[source,source2,middle_source],
        endpoint_feedback=feedback,corrections=corrections,held_body_envelope_m=.006)
    state.update(cyan_selected_goal=target,transport_record=carry)
    return result('carried',state,observed,'transport',transport_record=carry)


def _material_empty(ctx,state,scene,arm,position,rotation,body,q16,*,released_body_envelope_m):
    plan=ctx.tool('puzzle.plan_joint_path',position=np.asarray(position).tolist(),rotation=rotation,
                  arm_id=arm,min_joint_limit_margin_rad=.035)
    if not plan.get('feasible'):raise c.PlanningRefusal('Released empty-hand IK refused')
    geometry=_material_geometry()
    full=geometry.densify_q16_path(_qpath(plan,q16,arm))
    proof=geometry.screen_robot_material_path(full,scene,target='cyan',target_body_poses=body,finger_target_contract={})
    if not proof.get('safe'):raise c.PlanningRefusal('Released empty-hand material path refused: '+str(proof))
    require_board_margin(proof,scene)
    envelope=float(released_body_envelope_m)
    if not np.isfinite(envelope) or envelope<.006:
        raise c.PlanningRefusal('Released-body persistence must retain original6mm bound and release drop')
    for row in proof['rows']:
        if row['groups']['robot_target_nominal']['clearance_m']<.0005+envelope:
            raise c.PlanningRefusal('Empty-hand path intersects bounded released-body/drop envelope: '+str(row))
    proof['caller_released_body_guard']=dict(radial_inflation_m=envelope,required_clearance_m=.0005,
        source='Actual dated preopen RGB/FK6mm plus bounded nominal release drop; new postopen RGB remains actual when present',
        measured_current_pose_filled=False)
    off=0 if arm==0 else 8
    motor=[]
    for start in range(1,len(full),8):
        output=ctx.tool('robot.execute_trajectory',trajectory=dict(waypoints=[dict(positions=row[off:off+6].tolist())
            for row in full[start:start+8]]),arm_id=arm,max_steps_per_waypoint=2)
        if output.get('status')!='ok':raise RuntimeError('Empty receiver motor refused')
        motor.append(output)
    c.await_motion(ctx,position,rotation,arm,accuracy=.0025)
    return dict(native_ik=plan,robot_material=proof,motor=motor,
        released_body_semantics='Explicit short stationary/bounded persistence of actual preopen RGB/FK estimate; no held propagation after opening')


def _segment_conditional_release(ctx,state):
    state=deepcopy(state);arm=int(state['arm']);off=0 if arm==0 else 8
    mark=ctx.tool('robot.execution_journal_cursor')
    state,preopen,scene,belief,admission=_refresh(ctx,state)
    target=_select_target(ctx,state,scene,belief,lock_relative=True)
    state['plan']['target_R']=target['relative_target_R']
    width=float(belief['current_robot_encoder_values'][off+6]-belief['current_robot_encoder_values'][off+7])
    descent=move_held(ctx,state,target['release_tcp'],target['rotation'],scene,width,belief=belief,terminal=True)
    ctx.tool('robot.wait_steps',steps=30)
    state,preopen,scene,belief,preopen_source=_refresh(ctx,state)
    corrections=[]
    for attempt in range(3):
        target=_select_target(ctx,state,scene,belief,lock_relative=True)
        actual,actual_q=c.hand(ctx,arm)
        delta=float(np.linalg.norm(actual-np.asarray(target['release_tcp'])))
        angle=float((Rotation.from_quat(c.qarr(actual_q)).inv()*Rotation.from_quat(c.qarr(target['rotation']))).magnitude())
        if delta<=.0025 and angle<=.04:break
        if attempt==2 or delta>.012 or angle>.25:
            raise c.PlanningRefusal('Receiver actual preopen alignment exceeds bounded correction contract')
        width=float(belief['current_robot_encoder_values'][off+6]-belief['current_robot_encoder_values'][off+7])
        corrections.append(move_held(ctx,state,target['release_tcp'],target['rotation'],scene,width,belief=belief,terminal=True))
        state,preopen,scene,belief,preopen_source=_refresh(ctx,state)
    actual,actual_q=c.hand(ctx,arm)
    width=float(belief['current_robot_encoder_values'][off+6]-belief['current_robot_encoder_values'][off+7])
    # Recheck the complete actual loaded-to-open sweep in both robot arms.
    geometry=_material_geometry();rows=np.asarray(_width_rows(belief['current_robot_encoder_values'],arm,.055))
    body=np.asarray(belief['predicted_body_pose'])
    opening_guard=geometry.screen_robot_material_path(rows,scene,target='cyan',target_body_poses=body,
        finger_target_contract={arm:[_selected_bead(state,belief)]})
    if not opening_guard.get('safe'):raise c.PlanningRefusal('Receiver actual opening material sweep refused: '+str(opening_guard))
    require_board_margin(opening_guard,scene)
    source=deepcopy(belief['source'])
    opening=ctx.tool('robot.set_grip',arm_id=arm,width_m=.055,ramp_steps=18,settle_steps=15)
    if opening.get('clamped') is not False:raise RuntimeError('Receiver final opening clamped')
    # The old held lineage is now expired. Read actual empty q16 through a
    # fresh real RGB packet's synchronous encoder provenance, without using
    # target visibility as a physical success gate or reusing expired belief.
    after_open=c.state(ctx)
    q16=after_open['robot_geometry_provenance']['robot_encoder_values']
    if 'cyan' in after_open.get('poses',{}):body=np.asarray(after_open['poses']['cyan'])
    released_bound=max(.006,float(belief['uncertainty_m']))+float(target['height'])
    retreat=_material_empty(ctx,state,scene,arm,actual+[0.,0.,.055],actual_q,body,q16,released_body_envelope_m=released_bound)
    current=c.state(ctx)
    q16=current['robot_geometry_provenance']['robot_encoder_values']
    if 'cyan' in current.get('poses',{}):body=np.asarray(current['poses']['cyan'])
    home_p,home_q=geometry.robot.iks[geometry._arm(arm)].home_pose
    home=_material_empty(ctx,state,scene,arm,home_p,c.qdict(home_q),body,q16,released_body_envelope_m=released_bound)
    ctx.tool('robot.wait_steps',steps=60)
    before=c.state(ctx)
    ctx.tool('robot.wait_steps',steps=60)
    after=c.state(ctx)
    receipt=None;reason=None;memory=state['memory'];tcp,_=c.hand(ctx,arm)
    try:
        receipt=visual_memory.release_receipt('cyan',before,after,tcp,state['parameters'])
        memory=visual_memory.mint_placement(visual_memory.observe(memory,after,state['parameters']),receipt,after,state['parameters'])
    except visual_memory.MemoryRefusal as error:reason=str(error)
    record=dict(contract='receiver_conditional_or_owned_low_release_v1',execution_mark=mark,
        source=source,held_belief=belief,planning_scene=scene,preopen_rgb=g.rgb_record(preopen),
        bounded_release_admission=admission,preopen_source=preopen_source,
        current_target_measured_at_preopen='cyan' in preopen.get('poses',{}),
        requested_release_tcp=target['release_tcp'],requested_rotation=target['rotation'],
        actual_preopen_tcp=actual.tolist(),actual_preopen_rotation=actual_q,
        release_height_m=target['height'],actual_preopen_width_m=width,open_width_m=.055,
        open_result=opening,descent=descent,preopen_corrections=corrections,retreat=retreat,home_result=home,
        full_open_sweep=opening_guard,rigid_held_belief_after_open='EXPIRED',
        new_visual_release_receipt=receipt,receipt_refusal=reason,
        support_cleanup=dict(arm=state['transfer']['sender'],already_open_and_screened_home=True),
        release_rgb_pair=[g.rgb_record(before),g.rgb_record(after)],carry=state['transport_record'])
    return g._plain(dict(route='executed',state={**state,'memory':memory,'release_execution':record,
        'observation':g.rgb_record(after),'last_stage':'conditional_segment_release_execution','last_outcome':'executed'},
        evidence=dict(kind='conditional_segment_release_execution',timestamp=after['timestamp'],frame_sha256=after['frame_sha256'])))


def execute_transport(ctx,state):
    return _execute_transport(_ReceiverSceneContext(ctx,state),state)


def segment_conditional_release(ctx,state):
    return _segment_conditional_release(_ReceiverSceneContext(ctx,state),state)
