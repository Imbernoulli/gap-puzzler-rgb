"""Conditional control proposals from a past RGB grasp and robot encoders.

Predictions stay outside current RGB poses. Rigid grasp persistence and a6mm
planning envelope are assumptions; only released RGB can verify placement.
"""
from copy import deepcopy
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
from . import model,visual_memory,conditional_scene
from .vision_observer import goals
from .vision_planning import robot_seed,finite_fk
from .held_proprioception import retained_jaw_evidence

CONTRACT='pyramid_rgb_owned_held_control_v1'

def sha(value):return hashlib.sha256(json.dumps(value,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def accepted_rgb_source(source,records=()):
    """Preserve the original source type and authenticate calibrated priors."""
    source_type=source.get('observation_source')
    if source_type not in ('rgb_shape_geometry','calibrated_rgb_cad_seeded'):return False
    from .red_source_geometry import calibrated_source_rgb_supported
    matches=[]
    for row in records:
        q16=row.get('robot_geometry_provenance',{})
        if (row.get('observation_source')==source_type
            and row.get('timestamp')==source.get('timestamp')
            and row.get('frame_sha256')==source.get('frame_sha256')
            and row.get('poses',{}).get(source.get('piece'))==source.get('pose')
            and q16.get('robot_encoder_values')==source.get('robot_encoder_values')
            and q16.get('robot_encoder_sha256')==source.get('robot_encoder_sha256')
            and q16.get('step')==source.get('source_step')):
            matches.append(row)
    # Two actual observe calls can capture the identical unchanged epoch.
    # Matching that same complete source twice does not forge a new lineage.
    return bool(matches) and (source_type=='rgb_shape_geometry'
        or all(calibrated_source_rgb_supported(row) for row in matches))

def register(registry,env,bus,*,source_provider):
    if not callable(source_provider):raise ValueError('Actual RGB source provider is required')
    if hasattr(registry,'_invalidate_owned_held_sources'):raise ValueError('Owned held belief already registered; opening cutoffs cannot be reset')
    invalidated_sources={}
    opening_cutoffs={}
    coordinates=tuple(sorted({int(i) for a in env.handles.arms.values() for i in (*a.arm_coords,*a.gripper_coords)}|{int(i) for name in env.handles.arms for i in (env.gripper(name).width_coords or ())}))
    if coordinates!=tuple(range(16)):raise ValueError('Held belief requires the declared pinned YAM robot q16 slices')
    def invalidate_owned_held_sources(arm,reason):
        if arm not in (0,1):raise ValueError('Unknown opening robot arm')
        def revoke():
            step=int(env.step_count)
            previous=opening_cutoffs.get(arm,{}).get('step',-1)
            opening_cutoffs[arm]=dict(step=max(step,previous),reason=str(reason))
        # Opening attempts arrive on the graph thread. The native step belongs
        # to the owner thread, as do propagation and the actual RGB archive.
        return bus.call(revoke)
    registry._invalidate_owned_held_sources=invalidate_owned_held_sources
    def propagate_pyramid_held_belief(source:dict,arm_id:int=0):
        """Public robot FK propagates an immutable previously accepted RGB prior."""
        def solve():
            if arm_id not in (0,1):raise ValueError('Unknown robot arm')
            if source.get('mode')!='pyramid':raise ValueError('Explicit pyramid RGB source contract required')
            name=source['piece'];pose=np.asarray(source['pose'],float);q16=np.asarray(source['robot_encoder_values'],np.float32)
            if (name not in model.CELLS or pose.shape!=(7,) or q16.shape!=(16,)
                    or not np.isfinite(pose).all() or not np.isfinite(q16).all()
                    or abs(np.linalg.norm(pose[3:])-1.)>1.e-4
                    or not isinstance(source.get('source_step'),int) or isinstance(source.get('source_step'),bool)
                    or not accepted_rgb_source(source,source_provider())
                    or not source.get('frame_sha256')
                    or source.get('robot_encoder_sha256')!=hashlib.sha256(q16.tobytes()).hexdigest()):
                raise ValueError('Held belief needs an exact accepted RGB/robot prior')
            current_time=float(env.step_count/env.pipeline.control_hz)
            if not isinstance(source.get('timestamp'),(int,float)) or not np.isfinite(source['timestamp']) or source['timestamp']>current_time or current_time-source['timestamp']>30.:raise ValueError('Grasp prior is future, nonfinite or outside bounded30s control scope')
            cutoff=opening_cutoffs.get(arm_id)
            if cutoff is not None and source['source_step']<=cutoff['step']:
                raise ValueError('Owned held RGB epoch permanently invalidated by opening: '+cutoff['reason'])
            ik=env.arm('left' if arm_id==0 else 'right');seed=robot_seed(env)
            current=np.asarray([float(env.state.joint_q[i:i+1].numpy()[0]) for i in range(16)],np.float32);seed[:16]=current
            lineage=(name,arm_id,sha(source))
            if lineage in invalidated_sources:
                raise ValueError('Owned held RGB lineage permanently invalidated: '+invalidated_sources[lineage])
            retained=retained_jaw_evidence(q16,current,arm_id)
            if retained['contradicted']:
                reason='Previously loaded jaw >10mm has closed below4mm; robot proprioception contradicts retained grasp'
                invalidated_sources[lineage]=reason
                raise ValueError(reason)
            old=seed.copy();old[:16]=q16
            pp,pq=finite_fk(ik,old);cp,cq=finite_fk(ik,seed)
            source_centers=model.local_beads(name)@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]
            source_gap=float(np.linalg.norm(source_centers-pp,axis=1).min())
            if source_gap>.045:raise ValueError('Past RGB target was not near the same-image robot TCP; cannot bind a grasp prior')
            pr=Rotation.from_quat(pq).as_matrix();cr=Rotation.from_quat(cq).as_matrix()
            relative_p=pr.T@(pose[:3]-pp);relative_r=pr.T@Rotation.from_quat(pose[3:]).as_matrix()
            predicted=np.r_[cp+cr@relative_p,Rotation.from_matrix(cr@relative_r).as_quat()]
            return dict(contract=CONTRACT,piece=name,source=deepcopy(source),source_sha256=sha(source),
                source_hand_bead_gap_m=source_gap,current_timestamp=current_time,source_age_s=current_time-source['timestamp'],maximum_source_age_s=30.,current_robot_encoder_values=current.tolist(),
                current_robot_encoder_sha256=hashlib.sha256(current.tobytes()).hexdigest(),
                current_tcp=[*cp.tolist(),*cq.tolist()],object_in_tcp_position=relative_p.tolist(),
                object_in_tcp_rotation=relative_r.tolist(),predicted_body_pose=predicted.tolist(),uncertainty_m=.006,
                current_target_measured=False,verified_pose_error_m=None,
                retained_jaw_proprioception=retained,
                assumption='Rigid sourceRGB target-to-TCP transform persists through the bounded declared control; slip is unobserved and not certified. 6mm is a declared planning envelope, not covariance or a measured error bound.',
                scope='Conditional control proposal only, never current RGB poses or a release receipt.')
        return bus.call(solve)
    registry.register_callable('puzzle.propagate_pyramid_held_belief',propagate_pyramid_held_belief,summary='Robot-only FK propagates a prior accepted RGB grasp transform as explicitly conditional control proposal.',tags=('planning',))


def build_scene(observed,memory,p,name,belief,conditional_api=conditional_scene):
    """Fresh scene plus separately labelled conditional held CAD, no fake poses."""
    if p.get('mode')!='pyramid' or not p.get('static_public_prefix_precondition'):
        raise ValueError('This bounded release probe requires the named pyramid local-static contract')
    if belief.get('contract')!=CONTRACT or belief.get('piece')!=name or belief.get('current_target_measured') is not False:
        raise ValueError('Invalid conditional held proposal')
    raw=observed.get('poses',{});prefix=set(p['conditional_prefix']);required={'board',*(set(model.CELLS)-prefix-{name})}
    if not required.issubset(raw):raise ValueError('Fresh board and all other unplaced RGB poses required: '+str(sorted(required-set(raw))))
    if observed.get('observation_source')!='rgb_shape_geometry' or not observed.get('frame_sha256'):raise ValueError('Fresh recorded RGB required')
    bound=conditional_api.bind_local_prefix(memory,observed,p);board=visual_memory._pose(raw['board']);rb=Rotation.from_quat(board[3:]).as_matrix();public=goals(p['mode'],p['pyramid_plan']);occupancy={}
    for item in model.CELLS:
        if item==name:
            options=[visual_memory._pose(belief['predicted_body_pose']).tolist()]
            if item in raw:options.append(visual_memory._pose(raw[item]).tolist())
            source='conditional_robot_propagated_RGB_grasp';margin=.006
        elif item in prefix:
            center,r=public[item];options=[[*list(board[:3]+rb@center),*Rotation.from_matrix(rb@r).as_quat().tolist()]]
            if item in raw:options.append(visual_memory._pose(raw[item]).tolist())
            source='public_prefix_assumed_local_only';margin=.004
        else:options=[visual_memory._pose(raw[item]).tolist()];source='current_rgb_measurement';margin=0.
        occupancy[item]=dict(source=source,pose_hypotheses=options,inflation_m=margin,shape='public_beads_and_4.5mm_connector_sphere_cover')
    keys=('poses','timestamp','frame_sha256','observation_source','prefix_compatibility')
    return dict(contract='conditional_pyramid_held_control_scene_v1',target=name,measurement={k:deepcopy(observed[k]) for k in keys if k in observed},
        measured_names=sorted(raw),conditional_names=sorted(prefix),predicted_names=[name],occupancy=occupancy,
        prefix_reference_board=bound['local_prefix_reference_board'],held_belief=deepcopy(belief),
        margin_scope='Explicit local-static prefix and rigid-grasp planning assumptions; predicted held pose is not a current measurement. All exit/ALLstep target+prefix physics scored externally.')
