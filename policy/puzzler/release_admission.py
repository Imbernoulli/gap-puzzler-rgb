"""Release geometry and bounded ownership admission; no simulator state.

RGB observability is separate from physical nonfinger hand/camera CAD clearance.
An unobserved held body remains an explicit dated-RGB/FK conditional estimate.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from . import model, visual_memory
from .gripper_geometry import hand_primitives, clearance
from .vision_stages import StageRefusal


def check_bounded_release(name, belief, observed, parameters):
    source=belief.get('source',{})
    bound=float(belief.get('uncertainty_m',float('nan')))
    if belief.get('piece')!=name or not np.isfinite(bound) or not 0.<=bound<=.008:
        raise StageRefusal('Release needs a finite bounded owned held estimate')
    if parameters.get('conditional_attachment_id'):
        owned=(belief.get('contract')=='conditional_pregrasp_close_attachment_control_v1'
               and belief.get('attachment_id')==parameters['conditional_attachment_id']
               and source.get('attachment_id')==parameters['conditional_attachment_id']
               and source.get('piece')==name and not source.get('invalidated'))
    else:
        owned=(belief.get('contract')=='pyramid_rgb_owned_held_control_v1'
               and source==parameters.get('owned_held_belief_source')
               and source.get('piece')==name)
    if not owned:
        raise StageRefusal('Release estimate is not bound to the owned RGB/close token')
    source_rgb=source.get('source_rgb',source)
    if (not source_rgb.get('frame_sha256') or not isinstance(source_rgb.get('timestamp'),(int,float))
            or source_rgb.get('piece')!=name):
        raise StageRefusal('Release ownership has no dated actual RGB source')
    measured=name in observed.get('poses',{})
    contradiction=None
    if measured:
        contradiction=visual_memory.shape_displacement(name,observed['poses'][name],belief['predicted_body_pose'])
        if contradiction>bound:
            raise StageRefusal('Current accepted RGB contradicts the bounded held estimate')
    return dict(kind='owned_bounded_release_admission_v1',piece=name,owned_source_authenticated=True,
                current_target_3d_status='measured_current_RGB' if measured else 'UNKNOWN',
                conditional_body_bound_m=bound,physical_guard_body_inflation_m=0. if measured else bound,
                current_RGB_vs_FK_displacement_m=contradiction,
                rgb_observability='supported' if measured else 'UNKNOWN',
                source_timestamp=source_rgb['timestamp'],current_timestamp=belief.get('current_timestamp'),
                final_receipt_required=True,
                scope='Conditional execution permission only; no pose, placement or receipt is filled')


def target_wrist_camera_guard(position, rotation, width, body, name, parameters):
    """Check same-candidate hand/camera and body CAD, with declared inflation."""
    local=model.local_beads(name);pose=np.asarray(body,dtype=float)
    centers=local@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]
    distal={'left_lf_rot','left_lf_down','left_rf_rot','left_rf_down'}
    primitives=tuple(shape for shape in hand_primitives(float(width))
                     if shape.name.split(':')[0] not in distal)
    found=clearance(position,rotation,width,centers,radius=model.RADIUS,primitives=primitives)
    inflation=float(parameters.get('target_wrist_camera_guard_body_uncertainty_m',0.))
    margin=float(parameters.get('target_wrist_camera_guard_margin_m',.0005))
    if not np.isfinite(inflation) or not 0.<=inflation<=.008:
        raise StageRefusal('Invalid held body inflation for the physical release guard')
    raw=float(found['clearance_m']);physical=raw-inflation
    rgb=parameters.get('target_wrist_camera_guard_evidence')
    return dict(contract='target_to_wrist_camera_guard_v2',status='passed' if physical>=margin else 'rejected',
                wrist_nonfinger_clearance_m=physical,physical_wrist_camera_CAD_clearance_m=physical,
                raw_nonfinger_CAD_clearance_m=raw,body_inflation_m=inflation,
                minimum_clearance_m=physical,required_margin_m=margin,witness=found.get('witness'),
                rgb_observability='supported' if isinstance(rgb,dict) else 'UNKNOWN',
                rgb_evidence=deepcopy(rgb),rgb_fit_margin_is_physical_clearance=False,
                body_reference='Candidate release body pose in the same frame and epoch as candidate hand',
                admission=deepcopy(parameters.get('bounded_release_admission')),
                scope='Physical wrist and mounted-camera nonfinger CAD only; RGB observability is separate')
