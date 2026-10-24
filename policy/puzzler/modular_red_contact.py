"""RGB/proprio/public-CAD RED source contact primitive; no object-GT input.

The sole learned prior is the static PieceProfile. All robot poses and paths
are freshly computed from the observed body/CAD point, native IK and encoders.
A compliant endpoint is not advertised as a precise reached TCP pose.
"""
from copy import deepcopy
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
from . import model
from .red_source_geometry import source_contact_clearance,calibrated_source_rgb_supported
from .red_contact_region import selected_bead_region

class ContactSourceRefusal(RuntimeError):pass

def packed_observation(value):
    return {key:deepcopy(value[key])for key in ('poses','timestamp','frame_sha256','assessment','observation_source','robot_geometry_provenance')if key in value}

def encoder_epoch(observed):
    raw=observed.get('robot_geometry_provenance',{});q=np.asarray(raw.get('robot_encoder_values',[]),np.float32)
    source=observed.get('observation_source')
    if (source not in ('rgb_shape_geometry','calibrated_rgb_cad_seeded')
        or (source=='calibrated_rgb_cad_seeded' and not calibrated_source_rgb_supported(observed))
        or not observed.get('frame_sha256')
        or raw.get('coordinate_indices')!=list(range(16))or q.shape!=(16,)or not np.isfinite(q).all()
        or hashlib.sha256(q.tobytes()).hexdigest()!=raw.get('robot_encoder_sha256')
        or raw.get('timestamp')!=observed.get('timestamp')):
        raise ContactSourceRefusal('Actual RGB and synchronous public16 encoder epoch required')
    return dict(step=raw['step'],timestamp=raw['timestamp'],values=q.tolist(),sha256=raw['robot_encoder_sha256'])

def profile_parameters(state):
    p=dict(state['parameters']);p.update(p.get('pieces',{}).get('red',{}));p.update(state.get('profile',{}))
    p['open_width']=p.get('source_open_width',p.get('open_width',.055))
    p['release_width']=p.get('release_open_width',p.get('release_width',.09))
    return p

# The lift ends where the sender starts the hand-to-hand transfer, whose full
# public guard needs 0.5mm same-arm clearance at the actual encoders. A
# straight vertical lift can fold link_3 onto link_5 when Red's yaw changes the
# wrist; keep 0.5mm extra so loaded tracking cannot consume the floor.
LIFT_SELF_CLEARANCE_M=.001
LIFT_PATH_FLOOR_M=.0005


def _lift_proposals(arm,lift):
    """Public Cartesian lift endpoints relative to the actual closed TCP."""
    up=np.array([0.,0.,1.]);inboard=np.array([0.,1. if arm==1 else -1.,0.]);forward=np.array([1.,0.,0.])
    rows=[('vertical',up*lift)]
    rows+=[('inboard_%dmm'%round(d*1000),up*lift+inboard*d) for d in (.03,.06)]
    rows+=[('vertical_%dmm'%round((lift-d)*1000),up*(lift-d)) for d in (.015,.03)]
    rows+=[('inboard_30mm_forward_%+dmm'%round(d*1000),up*lift+inboard*.03+forward*d) for d in (.03,-.03)]
    return rows


def screened_lift(ctx,arm,start,rotation,lift,reserve):
    """Choose the first lift whose joint segment keeps robot material clear.

    Every proposal uses fresh native IK from current encoders with the
    profile joint reserve, then public robot colliders on the dense joint
    segment. If none passes, the original vertical lift is returned unchanged
    (the later full transfer guard still decides).
    """
    from .red_bimanual_handoff import geometry
    from . import modular_red_placement as base
    q16=np.asarray(base.proprio(ctx,arm)['robot_encoder_values'],float);off=8*arm
    g=geometry();rows=[]
    for name,offset in _lift_proposals(arm,lift):
        point=np.asarray(start,float)+offset
        ik=ctx.tool('puzzle.plan_pose',position=point.tolist(),rotation=rotation,arm_id=arm,min_joint_limit_margin_rad=reserve)
        row=dict(name=name,offset_m=offset.tolist(),position_error_m=float(ik['position_error_m']),
                 rotation_error_rad=float(ik['rotation_error_rad']),joint_limit_margin_rad=float(ik['joint_limit_margin_rad']))
        rows.append(row)
        if ik['position_error_m']>.002 or ik['rotation_error_rad']>.03 or ik['joint_limit_margin_rad']<reserve:
            row['status']='native_ik_or_reserve_refused';continue
        a=np.asarray(ik['current'],float);b=np.asarray(ik['joints'],float)
        if np.max(np.abs(a-q16[off:off+6]))>.002:
            row['status']='planner_seed_differs_from_actual_encoders';continue
        count=max(8,int(np.ceil(np.max(np.abs(b-a))/.012)))
        worst_path=np.inf;cross=np.inf
        for u in np.linspace(0.,1.,count+1):
            qq=q16.copy();qq[off:off+6]=a+(b-a)*u
            groups=g.robot_material(qq)
            worst_path=min(worst_path,groups['same_arm_nonadjacent']['clearance_m']);cross=min(cross,groups['cross_arm']['clearance_m'])
        endpoint=groups['same_arm_nonadjacent']['clearance_m'];table=groups['moving_robot_table']['clearance_m']
        row.update(path_same_arm_clearance_m=float(worst_path),endpoint_same_arm_clearance_m=float(endpoint),
                   path_cross_arm_clearance_m=float(cross),endpoint_table_clearance_m=float(table),samples=count+1)
        if worst_path<LIFT_PATH_FLOOR_M or cross<LIFT_PATH_FLOOR_M or table<LIFT_PATH_FLOOR_M or endpoint<LIFT_SELF_CLEARANCE_M:
            row['status']='robot_material_refused';continue
        row['status']='selected'
        print(json.dumps(dict(event='red_screened_lift',arm=arm,selected=name,endpoint_same_arm_clearance_m=float(endpoint),
            refused=[r['name'] for r in rows[:-1]])),flush=True)
        return point,dict(selected=name,proposals=rows,plan=ik,required_endpoint_same_arm_clearance_m=LIFT_SELF_CLEARANCE_M,
                          required_path_floor_m=LIFT_PATH_FLOOR_M)
    print(json.dumps(dict(event='red_screened_lift',arm=arm,selected=None,refused=[r['name'] for r in rows])),flush=True)
    return np.asarray(start,float)+np.array([0.,0.,lift]),dict(selected=None,proposals=rows,plan=None,
        fallback='original vertical lift; later full transfer guard decides')


def execute_contact_source(ctx,state,observed,plan):
    """Run the first-feasible RED contact fan with strict fresh RGB gates.

    A finger-occluded endpoint is a candidate refusal. The bounded recovery
    opens/withdraws, acquires a fresh board+RED RGB/q16 epoch, rechecks every
    alternate candidate's native IK/CAD corridor, then reapproaches. No RGB
    pose is filled from the original frame.
    """
    from . import vision_controller as c,conditional_scene,visual_memory
    from puzzler.modular_controller import scene_belief,save_observation
    if state.get('piece')!='red':raise ContactSourceRefusal('RED public contact profile only')
    p=profile_parameters(state);spec=p.get('source_contact_grasp')
    if not isinstance(spec,dict) or int(spec.get('selected_bead_index',4)) not in range(len(model.local_beads('red'))):
        raise ContactSourceRefusal('Named public opposed-blade source contact contract required')
    selected_bead_index=int(spec.get('selected_bead_index',4));arm=int(state.get('arm',1))
    if arm not in (0,1):raise ContactSourceRefusal('RED public source contact requires a declared arm')
    if not {'board','red'}.issubset(observed.get('poses',{})):
        raise ContactSourceRefusal('Fresh current source RED and board RGB required')
    if 'robot_geometry_provenance'not in observed:
        packet=next((r for r in reversed(state.get('rgb_history',[]))if r.get('timestamp')==observed.get('timestamp')and r.get('frame_sha256')==observed.get('frame_sha256')and r.get('poses')==observed.get('poses')and 'robot_geometry_provenance'in r),None)
        if packet is None:raise ContactSourceRefusal('Serialized source RGB lost its actual synchronous encoder epoch')
        observed={**observed,'robot_geometry_provenance':deepcopy(packet['robot_geometry_provenance'])}
    epoch0=encoder_epoch(observed);source=np.asarray(observed['poses']['red'])
    if plan.get('s',{}).get('frame_sha256')!=observed['frame_sha256']:
        raise ContactSourceRefusal('Source plan RGB provenance mismatch')
    family=list(plan.get('candidate_family',[plan]));attempts=[];chosen=None
    # All family members were statically screened by the online planner. Each
    # retry gets a fresh endpoint RGB/q16/FK and rechecks current native IK.
    for family_index,candidate in enumerate(family):
        # The planner has already screened each member with its declared arm.
        # Bind execution to that same member arm; never fall back to the
        # source y-side heuristic after a cross-arm candidate is selected.
        candidate_arm=int(candidate.get('arm',candidate.get('selected_contact_candidate',{}).get('arm',arm)))
        if candidate_arm not in (0,1):
            attempts.append(dict(candidate=deepcopy(candidate.get('selected_contact_candidate',{})),status='refused',reason='invalid candidate arm'))
            continue
        arm=candidate_arm
        g=np.asarray(candidate['grasp'],float);q=candidate['rotation'];idx=int(candidate['index'])
        if idx!=selected_bead_index:
            attempts.append(dict(candidate=deepcopy(candidate.get('selected_contact_candidate',{})),status='refused',reason='source RGB body/index binding'))
            continue
        if not np.array_equal(np.asarray(candidate['body']),source):
            # Recovery acquired a new RGB body epoch. Rebind the candidate's
            # body-relative TCP offset to that measured pose before fresh IK.
            old=np.asarray(candidate['body']);oldR=Rotation.from_quat(old[3:]).as_matrix();newR=Rotation.from_quat(source[3:]).as_matrix()
            rel=oldR.T@(np.asarray(candidate['grasp'])-old[:3]);candidate=deepcopy(candidate);candidate['body']=source.copy();candidate['grasp']=source[:3]+newR@rel;g=np.asarray(candidate['grasp'],float)
        # The first attempt starts from the open idle hand. Later candidates
        # use bounded open/withdraw/reobserve recovery before reapproach.
        if family_index==0:
            c.grip(ctx,arm,p['open_width'])
        else:
            tcp_now,q_now=c.hand(ctx,arm)
            c.move(ctx,tcp_now+[0,0,float(p['source_hover'])],q_now,arm)
            recovery=ctx.tool('vision.observe',hints={'red':observed['poses']['red']})
            if not {'board','red'}.issubset(recovery.get('poses',{})):
                attempts.append(dict(candidate=deepcopy(candidate.get('selected_contact_candidate',{})),status='refused',reason='recovery RGB RED still occluded'))
                continue
            recovery_epoch=encoder_epoch(recovery);source=np.asarray(recovery['poses']['red'])
            # Revalidate candidate pose and static guards at the fresh FK/RGB
            # epoch before any new motor command.
            current_state=save_observation(state,recovery)
            from .red_computed_carry import _repair_prefix_rgb
            current_state,prefix_reacquisition=_repair_prefix_rgb(ctx,current_state,label='contact_recovery_before_scene')
            scene=scene_belief(current_state,current_state['observation'])
            state=current_state
            hand_R=Rotation.from_quat(c.qarr(q)).as_matrix()
            for point in (g+[0,0,float(p['source_hover'])],g):
                ik=ctx.tool('puzzle.plan_pose',position=point.tolist(),rotation=q,arm_id=arm)
                if ik['position_error_m']>.002 or ik['rotation_error_rad']>.03 or ik.get('joint_limit_margin_rad',-1.)<p.get('min_ik_joint_margin_rad',0.):
                    attempts.append(dict(candidate=deepcopy(candidate.get('selected_contact_candidate',{})),status='refused',reason='fresh native IK/joint guard',ik=ik));break
            else:
                guard=source_contact_clearance(g,hand_R,float(p['open_width']),scene,conditional_scene,target='red')
                if guard['clearance_m']<spec['retained_clearance_margin_m']:
                    attempts.append(dict(candidate=deepcopy(candidate.get('selected_contact_candidate',{})),status='refused',reason='fresh CAD guard',guard=guard));continue
            if attempts and attempts[-1].get('candidate')==candidate.get('selected_contact_candidate',{}):
                continue
        c.move(ctx,g+[0,0,float(p['source_hover'])],q,arm)
        accuracy_error=None
        try:c.move(ctx,g,q,arm,cartesian=p.get('source_cartesian',False),contact='grasp')
        except c.MotorAccuracyError as error:
            if error.pe>spec['max_tcp_residual_m']or error.re>spec['max_rotation_residual_rad']:raise
            accuracy_error=dict(type=type(error).__name__,message=str(error),position_error_m=float(error.pe),rotation_error_rad=float(error.re))
        tcp0,qr0=c.hand(ctx,arm)
        before=ctx.tool('vision.observe',hints={'red':source.tolist()})
        if not {'board','red'}.issubset(before.get('poses',{})):
            attempts.append(dict(candidate=deepcopy(candidate.get('selected_contact_candidate',{})),status='refused',reason='endpoint RGB RED occluded'))
            # Bounded recovery is performed by the next candidate iteration.
            continue
        ep1=encoder_epoch(before);ctx.tool('robot.wait_steps',steps=int(spec['stable_control_steps']))
        current=ctx.tool('vision.observe',hints={'red':before['poses']['red']})
        if not {'board','red'}.issubset(current.get('poses',{})):
            attempts.append(dict(candidate=deepcopy(candidate.get('selected_contact_candidate',{})),status='refused',reason='stable endpoint RGB RED occluded'))
            continue
        ep2=encoder_epoch(current);tcp,qr=c.hand(ctx,arm)
        drift=float(np.linalg.norm(tcp-tcp0));angular=float((Rotation.from_quat(c.qarr(qr0)).inv()*Rotation.from_quat(c.qarr(qr))).magnitude())
        residual=float(np.linalg.norm(tcp-g));rotation_error=float((Rotation.from_quat(c.qarr(q)).inv()*Rotation.from_quat(c.qarr(qr))).magnitude())
        if (ep2['step']-ep1['step']<spec['stable_control_steps']or drift>=spec['max_tcp_drift_m']or angular>=spec['max_angular_drift_rad']or residual>spec['max_tcp_residual_m']or rotation_error>spec['max_rotation_residual_rad']):
            attempts.append(dict(candidate=deepcopy(candidate.get('selected_contact_candidate',{})),status='refused',reason='endpoint FK stability bound'));continue
        width=float(ep2['values'][14]-ep2['values'][15]);RR=Rotation.from_quat(c.qarr(qr)).as_matrix()
        region=selected_bead_region(tcp,RR,width,current['poses']['red'],selected_bead_index=selected_bead_index,max_target_finger_penetration_m=float(spec.get('max_target_finger_penetration_m',.003)))
        if not region['passes']:
            attempts.append(dict(candidate=deepcopy(candidate.get('selected_contact_candidate',{})),status='refused',reason='opposed-blade corridor'));continue
        chosen=dict(candidate=candidate,before=before,current=current,ep1=ep1,ep2=ep2,tcp0=tcp0,qr0=qr0,tcp=tcp,qr=qr,drift=drift,angular=angular,residual=residual,rotation_error=rotation_error,width=width,RR=RR,region=region,accuracy_error=accuracy_error)
        attempts.append(dict(candidate=deepcopy(candidate.get('selected_contact_candidate',{})),status='accepted_rgb_endpoint'))
        break
    if chosen is None:raise ContactSourceRefusal('All RED fan candidates refused before close: '+json.dumps(attempts,separators=(',',':')))
    candidate=chosen['candidate'];before=chosen['before'];current=chosen['current'];ep1=chosen['ep1'];ep2=chosen['ep2'];tcp=chosen['tcp'];qr=chosen['qr'];RR=chosen['RR'];width=chosen['width'];region=chosen['region'];g=np.asarray(candidate['grasp']);q=candidate['rotation'];residual=chosen['residual'];rotation_error=chosen['rotation_error'];drift=chosen['drift'];angular=chosen['angular'];accuracy_error=chosen['accuracy_error']
    current_state=save_observation(state,current)
    from .red_computed_carry import _repair_prefix_rgb
    current_state,prefix_reacquisition=_repair_prefix_rgb(ctx,current_state,label='contact_preclose_before_scene')
    # Prefix occupancy belongs to the independently reacquired current packet.
    # Keep the selected RED contact RGB and synchronous q16 epochs above intact.
    scene=scene_belief(current_state,current_state['observation']);memory=deepcopy(current_state['memory']);closing=[]
    for w in np.linspace(width,float(p.get('close_width',0.)),10):
        guard=source_contact_clearance(tcp,RR,float(w),scene,conditional_scene,target='red')
        if guard['clearance_m']<spec['retained_clearance_margin_m']:raise ContactSourceRefusal('Actual-width closing path intersects a retained CAD guard')
        closing.append(dict(width_m=float(w),guard=guard))
    prefix_packets=[deepcopy(packet) for packet in current_state.get('rgb_history',[]) if any(packet.get('timestamp')==attempt.get('timestamp') and packet.get('frame_sha256')==attempt.get('frame_sha256') for attempt in (prefix_reacquisition or {}).get('attempts',[]))]
    opening_reference=dict(actual_tcp=tcp.tolist(),actual_rotation=qr,requested_tcp=g.tolist(),requested_rotation=q,requested_endpoint_precisely_reached=residual<.0025 and rotation_error<.04,actual_tcp_residual_m=residual,actual_rotation_residual_rad=rotation_error,original_motor_accuracy_error=accuracy_error,contact_supported_label=True,default_free_space_accuracy_contract_unchanged=True,source_RGB_epochs=[packed_observation(before),packed_observation(current)],robot_encoder_epochs=[ep1,ep2],max_tcp_drift_m=drift,max_angular_drift_rad=angular,public_selected_bead_region=region,actual_width_closing_CAD_checks=closing,candidate_attempts=attempts,prefix_reacquisition=prefix_reacquisition,prefix_reacquisition_observation=packed_observation(current_state['observation']),prefix_reacquisition_rgb_packets=prefix_packets)
    close=ctx.tool('robot.set_grip',arm_id=arm,width_m=float(p.get('close_width',0.)),ramp_steps=18,settle_steps=15)
    if not isinstance(close,dict)or close.get('clamped')is not False:raise ContactSourceRefusal('Source close command refused or clamped')
    actual_lift_start,actual_lift_q=c.hand(ctx,arm)
    requested_lift,lift_screen=screened_lift(ctx,arm,actual_lift_start,actual_lift_q,float(p['lift']),float(p.get('min_ik_joint_margin_rad',.035)))
    if lift_screen['plan'] is not None:
        # Execute exactly the joint segment that was screened (as c.move would).
        c.feed_joint_plan(ctx,lift_screen['plan'],arm);c.settle_motion(ctx,requested_lift,actual_lift_q,arm,False,.006)
    else:c.move(ctx,requested_lift,actual_lift_q,arm,cartesian=False)
    actual_end,actual_end_q=c.hand(ctx,arm)
    prediction=np.asarray(current['poses']['red']).copy();prediction[:3]+=requested_lift-actual_lift_start;postlift=ctx.tool('vision.observe',hints={'red':prediction.tolist()});post_epoch=encoder_epoch(postlift);held='UNKNOWN';rise=None;gap=None
    if 'red'in postlift.get('poses',{}):
        rise=float(np.asarray(postlift['poses']['red'])[2]-np.asarray(current['poses']['red'])[2]);gap=c.part_hand_gap('red',postlift['poses']['red'],actual_end)
        if rise>=spec['min_lift_rgb_rise_m']and gap<=spec['max_held_bead_tcp_gap_m']:held='rgb_verified'
    return dict(contract='red_source_RGB_CAD_proprio_contact_v1',piece='red',arm=arm,original_source_rgb=packed_observation(observed),original_source_encoder_epoch=epoch0,source_plan={k:(candidate[k].tolist()if isinstance(candidate[k],np.ndarray)else deepcopy(candidate[k]))for k in ('index','heading','lean','grasp_height_m','target_center','target_R')},selected_contact_candidate=deepcopy(candidate.get('selected_contact_candidate',{})),candidate_policy='gap_first_feasible_online_rgb_cad_fk_ik',updated_memory=memory,updated_belief_ledger=scene.get('belief_ledger',{}),preclose_planning_scene=scene,contact_endpoint=opening_reference,actual_close_result=close,lift_reference='Actual measured TCP after closing, never failed requested grasp TCP',lift_screen={k:v for k,v in lift_screen.items() if k!='plan'},actual_lift_start_tcp=actual_lift_start.tolist(),actual_lift_start_rotation=actual_lift_q,requested_lift_tcp=requested_lift.tolist(),actual_postlift_tcp=actual_end.tolist(),actual_postlift_rotation=actual_end_q,postlift_rgb=packed_observation(postlift),postlift_encoder_epoch=post_epoch,actual_postlift_jaw_width_m=float(post_epoch['values'][14]-post_epoch['values'][15]),held_status=held,RGB_measured_rise_m=rise,RGB_measured_bead_tcp_gap_m=gap,current_pose_never_filled=True,object_GT_contact_evaluator_read=False,scope='Strict fresh RGB/q16/FK/CAD source primitive; occluded candidates are refused and never pose-filled.')
