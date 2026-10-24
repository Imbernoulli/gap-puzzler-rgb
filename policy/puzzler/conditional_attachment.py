"""Authenticated conditional attachment from distinct RGB andclose-proprio epochs.

No past source pose is relabelled as a current held measurement. The stationary
source and bounded grasp-slip assumptions are explicit, and independent test
checkpoints must still verify actual physical grasp/transport.
"""
from copy import deepcopy
import hashlib,json,uuid
import numpy as np
from scipy.spatial.transform import Rotation
from . import model
from .vision_planning import robot_seed,finite_fk


def register(registry,env,bus,observer,*,initial_records=(),historical_owned_records=()):
    cache={};invalidated_sources=set()
    owned_history=deepcopy(tuple(historical_owned_records))
    def owned_matches(source):
        return [row for row in [*observer.records,*owned_history]if row.get('timestamp')==source.get('timestamp')and row.get('frame_sha256')==source.get('frame_sha256')and row.get('poses',{}).get(source['piece'])==source.get('pose')]
    for record in initial_records:
        # Private native initializer only: newly captured on the actual replay
        # close epoch, never imported from a caller-supplied former UUID/cache.
        provenance=record.get('test_initialization_provenance',{})
        if provenance.get('kind')!='new capture at actual reexecuted close epoch'or len(owned_matches(record['source_rgb']))!=1 or record.get('rgb_and_close_are_same_epoch')is not False or record.get('current_target_measured')is not False or record.get('invalidated'):
            raise ValueError('Invalid private test-initialized new close attachment')
        if record['close_proprio_epoch']['step']>int(env.step_count):raise ValueError('Future close initialization')
        cache[record['attachment_id']]=deepcopy(record)
        alias=provenance.get('initializer_alias')
        if alias:cache[alias]=cache[record['attachment_id']]
    def source_key(source,arm):
        return (source['piece'],arm,source['timestamp'],json.dumps(source['frame_sha256'],sort_keys=True))
    def revoke(record,reason):
        record.update(invalidated=True,invalidation_reason=reason)
        invalidated_sources.add(source_key(record['source_rgb'],record['arm_id']))
    def robot_epoch(arm):
        q=np.asarray([float(env.state.joint_q[i:i+1].numpy()[0])for i in range(16)],np.float32);seed=robot_seed(env);seed[:16]=q
        p,r=finite_fk(env.arm('left'if arm==0 else'right'),seed);offset=0 if arm==0 else 8
        action=np.asarray(env.controllers.get('joint_position').target_action(env.control),float)
        if action.shape!=(14,)or not np.isfinite(action).all():raise ValueError('Publicrobot commandinverse shape mismatch')
        return dict(gripper_close_command_fraction=float(action[6 if arm==0 else 13]),timestamp=float(env.step_count/env.pipeline.control_hz),step=int(env.step_count),robot_encoder_values=q.tolist(),robot_encoder_sha256=hashlib.sha256(q.tobytes()).hexdigest(),tcp=[*p.tolist(),*r.tolist()],actual_jaw_width_m=float(q[offset+6]-q[offset+7]))
    def capture_conditional_attachment(source_rgb:dict,assumption:dict,arm_id:int=0):
        def capture():
            name=source_rgb['piece'];pose=np.asarray(source_rgb['pose'],float)
            if arm_id not in(0,1)or name not in model.CELLS or pose.shape!=(7,)or not np.isfinite(pose).all():raise ValueError('Invalid RGB source attachment input')
            matches=owned_matches(source_rgb)
            if len(matches)!=1:raise ValueError('Attachment origin is not an owned actual acceptedRGB record')
            if source_key(source_rgb,arm_id)in invalidated_sources:raise ValueError('This old RGB attachment lineage was permanentlyinvalidated; a fresh RGB epoch is required')
            close=robot_epoch(arm_id);age=close['timestamp']-source_rgb['timestamp'];bound=float(assumption.get('bead_displacement_bound_m',.008))
            if not 0<=age<=20.0 or not .004<=bound<=.012:raise ValueError('Conditional source/grasp bound orage outside declared scope')
            if not assumption.get('source_stationary_or_bounded_until_close'):raise ValueError('An explicit stationary/bounded source assumption is required')
            if close['gripper_close_command_fraction']<.95:raise ValueError('Actual robotcommand doesnot requestclose; cannotbind attachment')
            if not .004<close['actual_jaw_width_m']<.094:raise ValueError('Actual close proprio does not support an obstructed gripper')
            sourcepoints=model.local_beads(name)@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3];gap=float(np.linalg.norm(sourcepoints-np.asarray(close['tcp'][:3]),axis=1).min())
            if gap>.045+bound:raise ValueError('Actualclose TCP is not near the dated source CAD under thedeclared bound')
            r=Rotation.from_quat(close['tcp'][3:]).as_matrix();relative_p=r.T@(pose[:3]-np.asarray(close['tcp'][:3]));relative_r=r.T@Rotation.from_quat(pose[3:]).as_matrix()
            learned=assumption.get('learned_object_in_tcp_se3')
            learned_source_displacement=None
            if learned is not None:
                relative_p=np.asarray(learned['position_m'],float);relative_q=np.asarray(learned['quat_xyzw'],float)
                if relative_p.shape!=(3,)or relative_q.shape!=(4,)or not np.isfinite(relative_p).all()or not np.isfinite(relative_q).all()or abs(np.linalg.norm(relative_q)-1.)>1.e-4:
                    raise ValueError('Learned relative attachment has invalid finite SE3 parameters')
                relative_r=Rotation.from_quat(relative_q).as_matrix()
                nominal=model.local_beads(name)@(r@relative_r).T+np.asarray(close['tcp'][:3])+r@relative_p
                distances=np.linalg.norm(nominal[:,None]-sourcepoints[None],axis=2)
                learned_source_displacement=float(max(np.min(distances,axis=0).max(),np.min(distances,axis=1).max()))
                if learned_source_displacement>bound:
                    raise ValueError('Learned close attachment contradicts dated source CAD beyond declared displacement bound')
            record=dict(contract='conditional_pregrasp_rgb_close_proprio_attachment_v1',attachment_id=uuid.uuid4().hex,piece=name,arm_id=arm_id,
                source_rgb=deepcopy(source_rgb),source_rgb_role='Dated truepregrasp RGB; NOT a measuredpostlift held pose',close_proprio_epoch=close,
                rgb_and_close_are_same_epoch=False,source_age_at_close_s=age,source_hand_gap_under_stationarity_m=gap,
                invalidated=False,learned_nominal_close_vs_dated_source_CAD_displacement_m=learned_source_displacement,source_rgb_robot_proprio_epoch=deepcopy(matches[0].get('robot_geometry_provenance')),assumption=deepcopy(assumption),object_in_tcp_position=relative_p.tolist(),object_in_tcp_rotation=relative_r.tolist(),uncertainty_m=bound,
                current_target_measured=False,verified_pose_error_m=None,scope='Conditionalattachment controlproposal; stationarysource/grasp-slip bound is not a measured accuracy guarantee. Independentphysical grasp verification stillrequired.')
            cache[record['attachment_id']]=deepcopy(record);return record
        return bus.call(capture)
    def propagate_conditional_attachment(attachment_id:str,arm_id:int=0):
        def propagate():
            if attachment_id not in cache:raise ValueError('Unknown or forged attachment token')
            record=cache[attachment_id]
            if record.get('invalidated')or source_key(record['source_rgb'],record['arm_id'])in invalidated_sources:raise ValueError('Attachment was permanentlyinvalidated by opening/contradiction')
            if record['arm_id']!=arm_id:raise ValueError('Attachment belongs to anotherarm')
            current=robot_epoch(arm_id);age=current['timestamp']-record['close_proprio_epoch']['timestamp']
            if not 0<=age<=30.:
                revoke(record,'Conditional attachment expired');raise ValueError('Conditional attachment expired')
            if current['gripper_close_command_fraction']<.95:
                revoke(record,'Actualclose command no longer retainsgrasp');raise ValueError('Currentrobotcommand contradicts retainedgrasp')
            if not .004<current['actual_jaw_width_m']<.094:
                revoke(record,'Actualjaw proprio contradicts retainedgrasp');raise ValueError('Current jawproprio contradicts retained conditionalgrasp')
            from .held_proprioception import retained_jaw_evidence
            retained=retained_jaw_evidence(record['close_proprio_epoch']['robot_encoder_values'],current['robot_encoder_values'],arm_id)
            if retained['contradicted']:
                revoke(record,'Loaded close jaw closed below4mm; retained attachment contradicted')
                raise ValueError('Robot jaw collapse permanently invalidated conditional attachment')
            p=np.asarray(current['tcp'][:3]);r=Rotation.from_quat(current['tcp'][3:]).as_matrix();relative_p=np.asarray(record['object_in_tcp_position']);relative_r=np.asarray(record['object_in_tcp_rotation'])
            predicted=[*list(p+r@relative_p),*Rotation.from_matrix(r@relative_r).as_quat().tolist()]
            return dict(contract='conditional_pregrasp_close_attachment_control_v1',piece=record['piece'],attachment_id=attachment_id,source=deepcopy(record),current_timestamp=current['timestamp'],current_robot_encoder_values=current['robot_encoder_values'],current_robot_encoder_sha256=current['robot_encoder_sha256'],current_tcp=current['tcp'],object_in_tcp_position=relative_p.tolist(),object_in_tcp_rotation=relative_r.tolist(),predicted_body_pose=predicted,uncertainty_m=record['uncertainty_m'],current_target_measured=False,verified_pose_error_m=None,source_age_s=age,scope='Distinctepochs preserved: pastRGB plus actualclose-proprio andexplicit bounded contactassumptions; no fresh3D poseclaim.')
        return bus.call(propagate)
    def invalidate(arm,reason):
        for record in cache.values():
            if record['arm_id']==arm:revoke(record,reason)
    registry._invalidate_conditional_attachments=invalidate
    registry.register_callable('puzzle.capture_conditional_attachment',capture_conditional_attachment,summary='Bindowned pregraspRGB plus distinct actualcloseproprio under explicit geometryassumptions; never measuredheld pose',tags=('planning',))
    registry.register_callable('puzzle.propagate_conditional_attachment',propagate_conditional_attachment,summary='PublicFK propagates authenticated conditionalattachment; predictionnot currentRGB',tags=('planning',))
