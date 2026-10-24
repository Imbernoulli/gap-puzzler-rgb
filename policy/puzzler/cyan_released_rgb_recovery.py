"""Explicit postopen Cyan RGB/CAD recovery; public goal is a seed only."""
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from . import model
from .vision import detect, fit_piece
from .shape_vision import (foreground, raw_image_evidence, bead_image_support,
                          object_masks, refuse_shape_aliases, shape_distance)


def recover_released_cyan(strict, images, cameras, public_target):
    """Revalidate one released target with unchanged image/alias gates.

    Called only by the explicit released-Cyan observer mode. Neither the
    desired pose nor the old optimizer seed supplies a measured pose. Missing
    evidence and inequivalent accepted fits remain UNKNOWN. Other detections,
    sources, timestamps, ledgers and receipts are not changed here.
    """
    result = deepcopy(strict)
    original = deepcopy(result.get('detections', {}).get('cyan', {}))
    result.setdefault('poses', {}).pop('cyan', None)
    result.setdefault('bounded_rgb_source', {}).pop('cyan', None)
    diagnostic = dict(contract='released_Cyan_current_RGB_recovery_v1',
                      accepted=False, public_goal_role='optimizer proposal only',
                      pose_copied_from_goal=False, trials=[])

    def refuse(reason):
        result.setdefault('detections', {})['cyan'] = dict(
            visible=False, reason=reason, raw_shape_candidate=original,
            released_rgb_recovery=diagnostic, current_pose_supplied=False)
        return result

    if (strict.get('observation_source') != 'rgb_shape_geometry'
            or strict.get('is_current_rgb') is False
            or 'board' not in strict.get('poses', {})):
        return refuse('Released Cyan recovery needs current RGB board evidence')
    board = np.asarray(strict['poses']['board'], dtype=float)
    center, rotation = map(lambda x: np.asarray(x, dtype=float), public_target)
    if (board.shape != (7,) or center.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(board).all() or not np.isfinite(center).all()
            or not np.isfinite(rotation).all()
            or abs(np.linalg.norm(board[3:])-1.) > 1.e-4
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1.e-6)
            or abs(np.linalg.det(rotation)-1.) > 1.e-6):
        return refuse('Released Cyan recovery has invalid public CAD proposal')
    rb = Rotation.from_quat(board[3:]).as_matrix()
    proposal = np.r_[board[:3]+rb@center,
                     Rotation.from_matrix(rb@rotation).as_quat()]
    seeds = [('current_board_plus_public_goal', proposal)]
    if 'cyan' in strict.get('poses', {}):
        seeds.insert(0, ('original_current_candidate', strict['poses']['cyan']))
    foregrounds = {k: foreground(v) for k, v in images.items()}
    raw = raw_image_evidence(foregrounds, cameras)
    masks = {k: detect(v, only_names=('cyan',))['cyan']
             for k, v in images.items()}
    own = raw_image_evidence(masks, cameras)
    cache = {}

    def fit(shape, seed):
        key = (shape, tuple(seed))
        if key in cache:
            return cache[key]
        if shape == 'cyan':
            fitted = fit_piece(shape, masks, cameras, previous=seed)
        else:
            local_masks, contours = object_masks(shape, seed, foregrounds, cameras)
            fitted = fit_piece(shape, local_masks, cameras, previous=seed,
                               contour_points=contours)
        if fitted.get('visible'):
            supported, proof = bead_image_support(shape, fitted['pose'], raw, cameras)
            fitted['independent_rgb_support'] = proof
            if shape == 'cyan':
                attributed, ownproof = bead_image_support(shape, fitted['pose'], own, cameras)
                fitted['original_appearance_attribution'] = ownproof
                support = np.asarray(fitted.get('center_support', []), dtype=float)
                supported = bool(supported and attributed
                    and len(fitted.get('views', [])) >= 4 and support.shape == (4,)
                    and np.all(support >= .6)
                    and fitted['median_residual_m'] < .0018
                    and fitted['rms_residual_m'] < .004)
            if not supported:
                fitted.update(visible=False,
                              reason='Released RGB candidate lacks original independent bead/appearance support')
        cache[key] = fitted
        return fitted

    accepted = []
    for label, seed in seeds:
        fitted = fit('cyan', seed)
        proof = dict(optimizer_proposal=label, candidate=deepcopy(fitted))
        diagnostic['trials'].append(proof)
        if fitted.get('visible') is not True:
            continue
        trial = deepcopy(strict.get('detections', {}))
        trial['cyan'] = deepcopy(fitted)
        # Existing joint hypotheses and possible foreign evidence can only
        # withhold support. Never restore this candidate after alias refusal.
        refuse_shape_aliases(trial, raw, cameras, fit_candidate=fit)
        proof['after_original_alias_foreign_gate'] = deepcopy(trial['cyan'])
        if trial['cyan'].get('visible') is True:
            accepted.append(trial['cyan'])
    if not accepted:
        return refuse('No unique independently attributed released Cyan RGB fit')
    if any(shape_distance('cyan', accepted[0]['pose'], other['pose']) > .003
           for other in accepted[1:]):
        return refuse('Distinct released Cyan fits remain supported by current RGB')
    selected = deepcopy(min(accepted, key=lambda x: x['rms_residual_m']))
    diagnostic.update(accepted=True, supported_equivalent_fits=len(accepted))
    selected.update(identity_source='current released RGB/CAD fit with original alias and attribution gates',
                    released_rgb_recovery=diagnostic,
                    bounded_rgb_source=dict(source='fresh_rgb_calibrated_cad_fit',
                        current_board_measured=True, pose_fitted_from_current_rgb=True,
                        pose_not_copied_from_public_cad=True, q16_fk_required=True))
    result['detections']['cyan'] = selected
    result['poses']['cyan'] = selected['pose']
    return result
