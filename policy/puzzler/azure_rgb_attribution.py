"""Conservative appearance attribution for the explicit original task.

The canonical task uses the public original material palette. A generic
foreground CAD fit is not an Azure measurement when its positive evidence
belongs to other materials. This adds an attribution gate; it never creates a
pose, changes a geometric threshold, clears a ledger, or refreshes a source.
"""
from copy import deepcopy

from .vision import detect
from .shape_vision import raw_image_evidence, bead_image_support


def refuse_unattributed_original_azure(observed, images, cameras):
    """Keep original residual/bead gates and require own-material attribution.

    Caller scope is UnifiedPyramidGraph, whose public task Variant explicitly
    declares appearance='original'. Full palette competition is retained in
    detect(); another material cannot be reassigned to Azure by this request.
    The archived generic candidate remains diagnostic, never a measured pose.
    """
    if 'azure' not in observed.get('poses', {}):
        return observed
    masks = {key: detect(rgb, only_names=('azure',))['azure']
             for key, rgb in images.items()}
    evidence = raw_image_evidence(masks, cameras)
    supported, proof = bead_image_support(
        'azure', observed['poses']['azure'], evidence, cameras)
    result = deepcopy(observed)
    candidate = result['detections']['azure']
    attribution = dict(
        contract='original_Azure_current_RGB_attribution_v1',
        accepted=bool(supported),
        palette_source='Public original task assets model.COLORS; full-palette '
                       'nearest-neighbour classification of current RGB only',
        semantic_scope='Explicit original-appearance task; no fixture or earned-state dependency',
        per_bead_rgb_support=proof,
        thresholds='Unchanged bead_image_support: every bead needs centre and '
                   'independent tangent arc support in at least two current views',
        pose_created=False, ledger_modified=False,
        simulator_segmentation_used=False, privileged_body_state_used=False)
    if supported:
        candidate['original_appearance_attribution'] = attribution
        return result
    result['poses'].pop('azure')
    result['detections']['azure'] = dict(
        visible=False,
        reason='generic Azure CAD candidate lacks independent current RGB '
               'attribution to its public original material',
        raw_shape_candidate=candidate,
        original_appearance_attribution=attribution,
        raw_candidate_retained=True, current_pose_supplied=False)
    return result
