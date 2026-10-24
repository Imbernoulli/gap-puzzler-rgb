"""A fixed public-goal image witness for an explicitly conditional prior.

The witness is never returned as a measured pose. Outside preferred estimates
stay unchanged. A failed finite proposal means unknown, not physical motion.
"""
from copy import deepcopy
import hashlib

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from . import model,shape_vision as sv,vision
from .vision_observer import goals


CONTRACT='fixed_public_goal_image_compatibility_v1'
LIMITS=dict(median_residual_m=.0018,rms_residual_m=.004,center_support_px=4.,
            minimum_supported_views_per_bead=2,shape_equivalence_m=.003)


def cloud(name,pose):
    pose=np.asarray(pose,dtype=float)
    return model.local_beads(name)@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]


def public_goal(name,observed,targets):
    board=np.asarray(observed['poses']['board'],dtype=float)
    rb=Rotation.from_quat(board[3:]).as_matrix();center,rotation=targets[name]
    return np.r_[board[:3]+rb@center,Rotation.from_matrix(rb@rotation).as_quat()]


def fixed_silhouette(name,pose,reference,images,cameras):
    masks={key:sv.foreground(image) for key,image in images.items()}
    regions=[sv.object_masks(name,seed,masks,cameras)[0] for seed in (pose,reference)]
    union={key:regions[0][key]|regions[1][key] for key in cameras}
    raw=sv.raw_image_evidence(masks,cameras)
    centers=cloud(name,pose);residuals=[];supports=[];views=[]
    for key,camera in cameras.items():
        points=raw[key]['points'];pixels=np.rint(points).astype(int)
        points=points[union[key][pixels[:,1],pixels[:,0]]>0]
        if len(points)<35 or union[key].sum()<100:continue
        views.append(key)
        residuals.extend(vision.ray_shape_distance(name,centers,camera.rays(points),camera.camera_to_world[:3,3]))
        uv=camera.project(centers);distance=cv2.distanceTransform(1-union[key],cv2.DIST_L2,5)
        supports.append(float(np.mean(map_coordinates(distance,[uv[:,1],uv[:,0]],order=1,mode='constant',cval=1000.)<4.)))
    values=np.asarray(residuals)
    median=float(np.median(np.abs(values))) if len(values) else None
    rms=float(np.sqrt(np.mean(values**2))) if len(values) else None
    passing=bool(len(views)>=2 and median<.0018 and rms<.004 and sum(value>=.6 for value in supports)>=2
        and .1<pose[0]<1.1 and abs(pose[1])<.6 and .73<pose[2]<1.3)
    return dict(passes=passing,median_residual_m=median,rms_residual_m=rms,aggregate_support=supports,views=views),raw


def evaluate(name,observed,images,cameras,targets,robot_mesh,robot_provenance):
    """Test exactly one fixed public proposal using the reviewed construction."""
    from .robot_cad import robot_withheld_support
    result=dict(contract=CONTRACT,status='unknown',compatible=False,thresholds=dict(LIMITS),
        frame_sha256={key:hashlib.sha256(image.tobytes()).hexdigest() for key,image in images.items()},
        timestamp=robot_provenance['timestamp'],robot_geometry_provenance=deepcopy(robot_provenance),
        uniquely_measured_pose=False,measurement_precision_repaired=False,
        physical_precondition_verified_by_image=False,
        meaning='Existential compatibility of a conditional public prior; not a measurement or physical verification.')
    if observed.get('observation_source')!='rgb_shape_geometry':
        return {**result,'reason':'The source is not current shape-RGB perception'}
    if set(observed.get('poses',{}))!={'board',*model.CELLS}:
        return {**result,'reason':'A complete current tracked interpretation is unavailable; no physical contradiction asserted'}
    if robot_mesh is None:
        return {**result,'reason':'Synchronous robot geometry is unavailable'}
    witness=public_goal(name,observed,targets);preferred=list(observed['poses'][name])
    result.update(witness_pose=witness.tolist(),outside_preferred_fit_retained=preferred,
                  planning_union_pose_alternatives=[witness.tolist(),preferred])
    score,raw=fixed_silhouette(name,witness,preferred,images,cameras)
    result['silhouette']=score
    if not score['passes']:return {**result,'reason':'Fixed public hypothesis lacks unchanged silhouette evidence'}
    detections=deepcopy({key:value for key,value in observed['detections'].items() if key in model.CELLS})
    detections[name]={**detections[name],'visible':True,'pose':witness.tolist()}
    sv.refuse_shape_aliases(detections,raw,cameras)
    structural={key:item.get('supported_pose_hypotheses',0) for key,item in detections.items()
                if item.get('supported_pose_hypotheses',0)>1}
    result['retained_structural_alias_counts']=structural
    if structural:return {**result,'reason':'Distinct structural partial-CAD hypotheses remain; minimal rule refuses unknown'}
    if set(detections)!=set(model.CELLS) or any(not item.get('visible') for item in detections.values()):
        return {**result,'reason':'No complete attributed interpretation containing the fixed witness'}
    poses={key:list(item['pose']) for key,item in detections.items()}
    shapes={key:cloud(key,pose) for key,pose in poses.items()}
    if any(np.linalg.norm(shapes[aa][:,None]-shapes[bb][None,:],axis=2).min()<.020
           for index,aa in enumerate(shapes) for bb in list(shapes)[index+1:]):
        return {**result,'reason':'The interpretation is geometrically incompatible'}
    attribution={}
    for key,centers in shapes.items():
        foreign={other:[points] for other,points in shapes.items() if other!=key}
        attribution[key]=robot_withheld_support(centers,raw,cameras,robot_mesh,foreign=foreign)
    result.update(interpretation_poses=poses,all_track_attribution=attribution)
    if not all(item['passes_original_gate_after_withholding'] for item in attribution.values()):
        return {**result,'reason':'A participating track lacks genuine robot/foreign-withheld evidence'}
    return {**result,'status':'compatible_prior_uncertain_point','compatible':True,
            'reason':'Fixed in-prior hypothesis participates in one complete attributed interpretation'}


def validate_record(name,record,observed,parameters):
    """Recompute wire/provenance/geometric guards; pixels remain sensor work."""
    try:
        if (not parameters.get('prefix_image_compatibility') or not parameters.get('conditional_planning')
                or observed.get('observation_source')!='rgb_shape_geometry'
                or name not in parameters['conditional_prefix'] or parameters.get('conditional_prefix_margin_m',.004)!=.004
                or record.get('contract')!=CONTRACT or record.get('status')!='compatible_prior_uncertain_point'
                or record.get('thresholds')!=LIMITS or record.get('compatible') is not True
                or record.get('uniquely_measured_pose') is not False
                or record.get('measurement_precision_repaired') is not False
                or record.get('physical_precondition_verified_by_image') is not False
                or record.get('retained_structural_alias_counts')!={}
                or record.get('timestamp')!=observed.get('timestamp')
                or record.get('frame_sha256')!=observed.get('frame_sha256')
                or record.get('outside_preferred_fit_retained')!=list(observed['poses'][name])):
            return False
        target_map=goals(parameters.get('mode','flat'),parameters.get('pyramid_plan','pyramid_robot6_plan.json'))
        witness=np.asarray(record['witness_pose'],dtype=float);expected=public_goal(name,observed,target_map)
        if (witness.shape!=(7,) or not np.isfinite(witness).all() or abs(np.linalg.norm(witness[3:])-1.)>1.e-4
                or sv.shape_distance(name,witness,expected)>1.e-9):return False
        if record.get('planning_union_pose_alternatives')!=[record['witness_pose'],list(observed['poses'][name])]:return False
        score=record['silhouette']
        if (not np.isfinite([score['median_residual_m'],score['rms_residual_m'],*score['aggregate_support']]).all()
                or score['passes'] is not True or len(set(score['views']))<2
                or not set(score['views']).issubset(observed['frame_sha256'])
                or len(score['aggregate_support'])!=len(score['views'])
                or any(not 0.<=value<=1. for value in score['aggregate_support']) or score['median_residual_m']>=.0018
                or score['rms_residual_m']>=.004 or sum(value>=.6 for value in score['aggregate_support'])<2):return False
        provenance=record['robot_geometry_provenance'];encoders=np.asarray(provenance['robot_encoder_values'],dtype=np.float32)
        indices=provenance['coordinate_indices']
        if (indices!=list(range(16)) or encoders.shape!=(16,) or not np.isfinite(encoders).all()
                or provenance['timestamp']!=observed['timestamp'] or provenance['visible_robot_shapes']<=0
                or hashlib.sha256(encoders.tobytes()).hexdigest()!=provenance['robot_encoder_sha256']):return False
        poses=record['interpretation_poses'];attribution=record['all_track_attribution']
        if set(poses)!=set(model.CELLS) or set(attribution)!=set(model.CELLS):return False
        for key in model.CELLS:
            expected_pose=record['witness_pose'] if key==name else list(observed['poses'][key])
            if poses[key]!=expected_pose:return False
            rows=attribution[key]['by_view'];counts=np.asarray(attribution[key]['bead_supported_views'])
            if not rows or not set(rows).issubset(observed['frame_sha256']):return False
            actual=np.sum([item['support'] for item in rows.values()],axis=0)
            if counts.shape!=(len(model.CELLS[key]),) or not np.array_equal(counts,actual) or np.any(counts<2):return False
        shapes={key:cloud(key,pose) for key,pose in poses.items()}
        if any(np.linalg.norm(shapes[aa][:,None]-shapes[bb][None,:],axis=2).min()<.020
               for index,aa in enumerate(shapes) for bb in list(shapes)[index+1:]):return False
        return True
    except (KeyError,ValueError,TypeError,IndexError,OverflowError):
        return False
