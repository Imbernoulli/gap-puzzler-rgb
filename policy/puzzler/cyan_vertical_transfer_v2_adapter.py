"""Transfer a qualified vertical Cyan source with its stronger jaw contract.

The measured vertical source has a different attachment from the original tip
pinch. Existing runtime planning binds that actual new source; no old path or
certificate is imported. This adapter adds the source-specific >=20mm actual
loaded-jaw gate to every donor propagation before opening.
"""
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation

from . import cyan_bimanual_feedback_adapter as transfer
from . import vision_controller as c,cyan_geometry as g,model
from .release_admission import check_bounded_release
from .modular_controller import save_observation
from .cyan_bimanual_adapter import _geometry


class _VerticalSourceContext:
    def __init__(self,ctx,state):
        self.ctx=ctx
        self.donor=int(state.get('transfer',{}).get('sender',state['arm']))
        self.minimum=float(state['parameters'].get('source_minimum_loaded_jaw_width_m',.020))
        if not np.isfinite(self.minimum) or self.minimum<.020:
            raise c.PlanningRefusal('Vertical source transfer cannot weaken the actual20mm loaded-jaw contract')
        self.bind_transfer_scene(state)

    def bind_transfer_scene(self,state):
        # An explicit snapshot is updated only by actual RGB scene rebuilds.
        # Do not retain the caller's state: transfer leaves deep-copy it.
        self.scene_measurement=deepcopy(state['planning_scene']['measurement'])
        prior=state['parameters']['historical_complete_rgb']
        measured=self.scene_measurement
        if (measured.get('timestamp')!=prior.get('timestamp')
                or measured.get('frame_sha256')!=prior.get('frame_sha256')
                or 'board' not in measured.get('poses',{})
                or not np.array_equal(measured['poses']['board'],prior.get('poses',{}).get('board'))):
            raise c.PlanningRefusal('Planning scene and historical board must retain the same actual RGB epoch')
        self.scene_limit=min(30.,float(state['parameters'].get('historical_max_age_s',30.)))
        if not np.isfinite(self.scene_limit) or self.scene_limit<=0:
            raise c.PlanningRefusal('Actual handoff scene requires the original bounded age scope')

    def __getattr__(self,name):
        return getattr(self.ctx,name)

    def tool(self,name,**kwargs):
        if name in ('robot.execute_trajectory','robot.set_grip'):
            epoch=self.ctx.tool('robot.execution_journal_cursor')
            now=epoch.get('timestamp');stamp=self.scene_measurement.get('timestamp')
            if (not isinstance(now,(int,float)) or not np.isfinite(now)
                    or not isinstance(stamp,(int,float)) or not np.isfinite(stamp)
                    or not 0<=now-stamp<=self.scene_limit
                    or not self.scene_measurement.get('frame_sha256')
                    or self.scene_measurement.get('observation_source')!='rgb_shape_geometry'):
                raise c.PlanningRefusal('Motor requires an authentic actual RGB scene within the original30s scope; reobserve and rescreen')
        result=self.ctx.tool(name,**kwargs)
        if name in ('puzzle.propagate_pyramid_held_belief','puzzle.propagate_conditional_attachment') and int(kwargs.get('arm_id',0))==self.donor:
            enc=np.asarray(result['current_robot_encoder_values'],dtype=float)
            off=0 if self.donor==0 else 8
            if enc.shape!=(16,) or not np.isfinite(enc).all():
                raise c.PlanningRefusal('Vertical source propagation needs actual finite q16')
            width=float(enc[off+6]-enc[off+7])
            if width<self.minimum:
                raise c.PlanningRefusal('Actual vertical donor jaw violates its20mm source-contact certificate')
        return result


def approach_receiver(ctx,state):
    state=deepcopy(state)
    scoped=_VerticalSourceContext(ctx,state)
    previous=transfer._belief(scoped,state)
    observed=state.get('observation',{})
    if ('cyan' not in observed.get('poses',{})
            or observed.get('timestamp',-1)<previous['current_timestamp']-.5):
        observed=c.state(scoped,{'cyan':previous['predicted_body_pose']})
    if 'cyan' not in observed.get('poses',{}):
        raise c.PlanningRefusal('New vertical transfer needs the actual postqualification RGB source before planning')
    admission=check_bounded_release('cyan',previous,observed,state['parameters'])
    state['parameters']['owned_held_belief_source']=g._source(observed,'cyan')
    state['parameters'].pop('conditional_attachment_id',None)
    state=save_observation(state,observed)
    state['held_status']='CURRENT_RGB_VERIFIED'
    # Replace persistence only with an actual complete current packet. The
    # diagnostic hold must not consume the later scene's original30s budget.
    prefix=set(state['parameters'].get('conditional_prefix',()))|set(state.get('executed_placements',()))
    if {'board',*(set(model.CELLS)-prefix)}.issubset(observed.get('poses',{})):
        state['planning_scene']=g.planning_scene(observed,state)
        state['parameters']['historical_complete_rgb']=g.rgb_record(observed)
    scoped.bind_transfer_scene(state)
    current=transfer._belief(scoped,state)
    old_indices=state['parameters']['source_contact_bead_indices']
    if len(old_indices)!=1:
        raise c.PlanningRefusal('Qualified vertical source must retain its single measured source contact region')
    beads=model.local_beads('cyan')
    reference=(np.asarray(previous['object_in_tcp_position'])
               +np.asarray(previous['object_in_tcp_rotation'])@beads[int(old_indices[0])])
    measured=beads@np.asarray(current['object_in_tcp_rotation']).T+np.asarray(current['object_in_tcp_position'])
    distances=np.linalg.norm(measured-reference,axis=1)
    selected=int(np.argmin(distances))
    if distances[selected]>.006:
        raise c.PlanningRefusal('Postqualification RGB changed the named vertical contact region beyond original6mm')
    state['parameters']['source_contact_bead_indices']=[selected]
    state['vertical_transfer_source_refresh']=dict(
        actual_postqualification_source=g._source(observed,'cyan'),admission=admission,
        previous_source_timestamp=previous['source'].get('timestamp'),
        measured_frame_selected_contact_bead=selected,
        contact_region_frame_mapping_displacement_m=float(distances[selected]),
        current_target_measured=True,source_validity_not_extended=True)
    return transfer.approach_receiver(scoped,state)


def _refresh_close(ctx,state):
    """Rebind a real packet and its named physical contact regions together."""
    previous=transfer._belief(ctx,state)
    observed=c.state(ctx,{'cyan':previous['predicted_body_pose']})
    state=save_observation(state,observed)
    evidence=dict(actual_rgb_timestamp=observed['timestamp'],
                  current_target_measured='cyan' in observed.get('poses',{}))
    if 'cyan' in observed.get('poses',{}):
        beads=model.local_beads('cyan');candidate=state['transfer']['candidate']
        old_index=int(candidate['bead_index'])
        old_center=(np.asarray(previous['object_in_tcp_position'])
                    +np.asarray(previous['object_in_tcp_rotation'])@beads[old_index])
        source,admission,current=transfer.refresh_source_scene(ctx,state,observed,previous)
        centers=beads@np.asarray(current['object_in_tcp_rotation']).T+np.asarray(current['object_in_tcp_position'])
        distances=np.linalg.norm(centers-old_center,axis=1);index=int(np.argmin(distances))
        if distances[index]>.006:
            raise c.PlanningRefusal('Fresh close RGB moved the named receiver region beyond original6mm')
        candidate['bead_index']=index
        candidate['permitted_distal_bead_indices']=[index]
        state['transfer']['preclose_source']=deepcopy(source)
        evidence.update(admission=admission,actual_selected_bead=index,
                        physical_region_mapping_displacement_m=float(distances[index]))
    else:
        # Occlusion does not create a new measured target or extend its scope.
        # Retain only the actual dated source and its original propagation gate.
        current=transfer._belief(ctx,state)
        evidence['dated_actual_source_timestamp']=state['transfer']['preclose_source']['timestamp']
    ctx.bind_transfer_scene(state)
    return state,observed,current,evidence


def _center_open_receiver(ctx,state,belief):
    arm=int(state['transfer']['receiver']);off=0 if arm==0 else 8
    enc=np.asarray(belief['current_robot_encoder_values'])
    if enc[off+6]-enc[off+7]<.045:
        raise c.PlanningRefusal('Measured receiver centering requires the actual hand still open')
    candidate=state['transfer']['candidate'];body=np.asarray(belief['predicted_body_pose'])
    selected=model.local_beads('cyan')[int(candidate['bead_index'])]
    point=body[:3]+Rotation.from_quat(body[3:]).as_matrix()@selected
    tcp,q=c.hand(ctx,arm);H=Rotation.from_quat(c.qarr(q)).as_matrix()
    actual=H.T@(point-tcp)
    depth=float(candidate['bead_in_tcp_m'][2])
    if not .010<=depth<=.0165:
        raise c.PlanningRefusal('Receiving public grasp depth exceeds the distal contact region')
    delta=H@(actual-np.array([0.,0.,depth]))
    if not np.isfinite(delta).all() or np.linalg.norm(delta)>.006:
        raise c.PlanningRefusal('Actual receiver RGB/FK correction exceeds the original6mm local scope')
    evidence=dict(actual_selected_bead_in_tcp_m=actual.tolist(),
        desired_selected_bead_in_tcp_m=[0.,0.,depth],correction_world_m=delta.tolist(),
        desired_transform_is_only_a_proposal=True,actual_rotation_preserved=True)
    if np.linalg.norm(delta)>.00015:
        evidence['motion']=transfer._execute(ctx,state,arm,[(tcp+delta,q)],
            'actual_RGB_open_receiver_center',receiver_contact=False)
        # _execute normally permits2.5mm transit arrival error. Centering a
        #22mm bead needs its own measured submillimeter endpoint gate.
        c.await_motion(ctx,tcp+delta,q,arm,accuracy=.0007)
        latest=transfer._belief(ctx,state)
        transfer._guard(state,[latest['current_robot_encoder_values']],arm,latest,
                        receiver_contact=False)
    return evidence


def close_receiver(ctx,state):
    state=deepcopy(state);scoped=_VerticalSourceContext(ctx,state)
    state,observed,belief,refresh=_refresh_close(scoped,state)
    centered=_center_open_receiver(scoped,state,belief)
    state['transfer'].setdefault('receiver_close_feedback',[]).append(dict(
        stage='actual_open_center',refresh=refresh,centering=centered))
    receiver=int(state['transfer']['receiver'])
    # Each sweep is generated from the next actual q16 and actual bounded
    # source. No previous swept-volume certificate licenses a changed source.
    for command,screened in ((.030,.030),(.026,.026),(.0242,.0242),(0.,.022)):
        state,observed,belief,refresh=_refresh_close(scoped,state)
        result=transfer._grip(scoped,state,receiver,command,
            loaded_screen_width=screened,receiver_contact=True)
        state['transfer']['execution'].append(result)
        state['transfer']['receiver_close_feedback'].append(dict(
            stage='actual_full_close' if command==0. else 'actual_width_stage',
            refresh=refresh,command_width_m=command,geometry_screened_width_m=screened))
    transfer_state=state['transfer']
    assumption=dict(source_stationary_or_bounded_until_close=True,
        bead_displacement_bound_m=.006,
        reason='Donor remains commanded closed and stationary during actual receiver staged close; '
               'independent dualhold physical checkpoint must pass before donor opens.')
    attachment=scoped.tool('puzzle.capture_conditional_attachment',
        source_rgb=transfer_state['preclose_source'],assumption=assumption,arm_id=receiver)
    transfer_state['receiver_conditional_attachment']=attachment
    transfer_state['actual_receiver_close_proprio']=attachment['close_proprio_epoch']
    scoped.tool('robot.wait_steps',steps=8)
    observed=c.state(scoped,{'cyan':transfer._belief(scoped,state)['predicted_body_pose']})
    return transfer._out('receiver_closed',state,observed,
        actual_receiver_close=attachment['close_proprio_epoch'],current_target_measured=False,
        scope='Actual RGB/FK open centering and feedback after each real jaw stage; full close command0; independent dualhold CP required before donor opens.')


def release_donor(ctx,state):
    state=deepcopy(state);scoped=_VerticalSourceContext(ctx,state)
    handoff=state['transfer'];donor=int(handoff['sender']);receiver=int(handoff['receiver'])
    p=state['parameters']
    handoff['retired_sender_source']=p.pop('owned_held_belief_source',None)
    p.pop('conditional_attachment_id',None)
    attachment=deepcopy(handoff['receiver_conditional_attachment'])
    p['conditional_attachment']=attachment;p['conditional_attachment_id']=attachment['attachment_id']
    state.update(arm=receiver,conditional_attachment=attachment,
                 held_status='UNKNOWN_conditional_receiver_attachment')
    handoff['active_owner']=receiver;handoff['old_sender_source_active']=False
    # Only a successful independent dualhold graph checkpoint can reach this
    # leaf. Opening physically and permanently retires the donor lineage.
    handoff['execution'].append(transfer._grip(scoped,state,donor,.030,receiver_contact=True))
    handoff['donor_contact_revoked']=True
    _release_guard(state,[transfer._belief(scoped,state)['current_robot_encoder_values']],donor,
                   transfer._belief(scoped,state))
    tcp,q=c.hand(scoped,donor);backward=-Rotation.from_quat(c.qarr(q)).as_matrix()[:,2]
    handoff['execution'].append(_release_execute(scoped,state,donor,tcp+.015*backward,q,
                                               'donor_short_axial_exit'))
    handoff['execution'].append(_release_grip(scoped,state,donor,.055))
    tcp,q=c.hand(scoped,donor)
    # Board-independent inward translation follows the real short axial exit;
    # it avoids the vertical source arm's upper joint boundary. This is a
    # relative public workspace proposal, rescreened at actual runtime q16.
    handoff['execution'].append(_release_execute(scoped,state,donor,tcp+[-.060,0.,.040],q,
                                               'donor_inward_clear_exit'))
    state['anchor_evidence']={**state.get('anchor_evidence',{}),'arm':donor,
        'previous_board_support_arm':receiver,'support_cleanup_already_completed':True,
        'donor_open_and_withdrawn_after_independent_dualhold':True}
    scoped.tool('robot.wait_steps',steps=8)
    observed=c.state(scoped,{'cyan':transfer._belief(scoped,state)['predicted_body_pose']})
    return transfer._out('donor_released',state,observed,current_target_measured=False,
        active_receiver=receiver,scope='Actual dualhold precedes donor opening; source retired; donor target permission revoked before both withdrawal segments; independent receiver-only CP required.')


def _release_guard(state,rows,arm,belief):
    handoff=state['transfer']
    if not handoff.get('donor_contact_revoked'):
        raise c.PlanningRefusal('Donor withdrawal requires explicit contact permission revocation')
    scene=deepcopy(state['planning_scene'])
    for name,entry in scene['occupancy'].items():
        if name!='cyan':entry['inflation_m']=max(.004,float(entry['inflation_m']))
    checked=_geometry().screen_q16_path(q16_path=np.asarray(rows).tolist(),scene=scene,
        moving_arm=int(arm),held_arm=int(state['arm']),held_body_pose=belief['predicted_body_pose'],
        object_in_tcp_position=belief['object_in_tcp_position'],
        object_in_tcp_rotation=belief['object_in_tcp_rotation'],target='cyan',held_envelope_m=.006,
        finger_target_contract={int(handoff['receiver']):[int(handoff['candidate']['bead_index'])]})
    if not checked.get('safe',checked.get('passed',False)):
        raise c.PlanningRefusal('Actual donor withdrawal full public geometry refused: '+str(checked))
    return checked


def _release_execute(ctx,state,arm,position,rotation,phase):
    belief=transfer._belief(ctx,state);plan=transfer._plan(ctx,arm,[(position,rotation)])
    full=transfer._qpath(plan,belief['current_robot_encoder_values'],arm)
    proof=_release_guard(state,full,arm,belief);off=0 if arm==0 else 8;executed=[]
    for begin in range(1,len(full),16):
        transfer._belief(ctx,state)
        output=ctx.tool('robot.execute_trajectory',trajectory=dict(waypoints=[dict(positions=row.tolist())
            for row in full[begin:begin+16,off:off+6]]),arm_id=arm,max_steps_per_waypoint=2)
        if output.get('status')!='ok':raise RuntimeError('Actual donor retreat motor refused')
        executed.append(output)
    c.await_motion(ctx,position,rotation,arm,accuracy=.0025)
    actual=transfer._belief(ctx,state)
    endpoint=_release_guard(state,[actual['current_robot_encoder_values']],arm,actual)
    return dict(phase=phase,native_ik=plan,complete_geometry=proof,motor=executed,
                actual_endpoint_geometry=endpoint,donor_contact_permission_revoked=True)


def _release_grip(ctx,state,arm,width):
    belief=transfer._belief(ctx,state)
    proof=_release_guard(state,transfer._width_rows(belief['current_robot_encoder_values'],arm,width),arm,belief)
    output=ctx.tool('robot.set_grip',arm_id=arm,width_m=width,ramp_steps=18,settle_steps=15)
    actual=transfer._belief(ctx,state)
    endpoint=_release_guard(state,[actual['current_robot_encoder_values']],arm,actual)
    return dict(command_width_m=width,proposal_screen=proof,actual_endpoint_screen=endpoint,motor=output,
                donor_contact_permission_revoked=True)


def verify_receiver(ctx,state):
    return transfer.verify_receiver(_VerticalSourceContext(ctx,state),state)
