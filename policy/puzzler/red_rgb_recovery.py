"""Reviewed Red-only RGB recovery applied to an existing strict observation.

The inner estimator is identical to the frozen modular observer. Its strict
input is supplied unchanged; only a missing Red can enter recovery.
"""
from copy import deepcopy
from . import vision as plain_vision


def recover_missing_red(strict,images,cameras,previous=None):
    if 'red' in strict.get('poses',{}) or 'board' not in strict.get('poses',{}):
        return strict
    known=('red',)
    def estimate_shapes(*args,**kwargs):
        return deepcopy(strict)
    def guarded_estimator(images,cameras,previous=None):
                    """Shape tracker plus bounded plain RGB/CAD recovery.

                    Packed lower-layer pieces can fail the shape tracker's
                    all-bead occlusion rule after a neighbouring release.  For a
                    missing target, run the independent calibrated RGB/CAD fit
                    as a second observer.  A pose is merged only when the fresh
                    plain fit itself passes its four-view support and residual
                    gates and the same packet contains a board fit; no CAD pose
                    is copied into policy state.
                    """
                    strict=estimate_shapes(images,cameras,previous,
                                            known_unplaced_names=known)
                    missing=[name for name in known
                             if name not in strict.get('poses',{})]
                    if not missing or 'board' not in strict.get('poses',{}):
                        return strict
                    plain=plain_vision.estimate(images,cameras,previous={})
                    board_ok='board' in plain.get('poses',{})
                    for name in missing:
                        detection=plain.get('detections',{}).get(name,{})
                        if name=='red':
                            from puzzler.shape_vision import foreground,raw_image_evidence,bead_image_support,refuse_shape_aliases,object_masks
                            # A held RGB/FK prediction is an optimizer proposal,
                            # never the fresh pose. Preserve full-palette pixel
                            # competition when refitting this missing shape.
                            masks={key:plain_vision.detect(rgb,only_names=(name,))[name]
                                   for key,rgb in images.items()}
                            if previous and name in previous:
                                detection=plain_vision.fit_piece(name,masks,cameras,previous=previous[name])
                            foregrounds={key:foreground(rgb) for key,rgb in images.items()}
                            evidence=raw_image_evidence(foregrounds,cameras)
                            supported,diagnostic=bead_image_support(name,detection['pose'],evidence,cameras) if detection.get('visible') else (False,{})
                            if supported:
                                trial=deepcopy(strict['detections'])
                                trial[name]=dict(detection,independent_rgb_support=diagnostic)
                                cache={}
                                def fit_hypothesis(shape,seed):
                                    key=(shape,tuple(seed))
                                    if key not in cache:
                                        if shape==name:
                                            fitted=plain_vision.fit_piece(shape,masks,cameras,previous=seed)
                                        else:
                                            local_masks,contours=object_masks(shape,seed,foregrounds,cameras)
                                            fitted=plain_vision.fit_piece(shape,local_masks,cameras,previous=seed,contour_points=contours)
                                        if fitted.get('visible'):
                                            supported_beads,proof=bead_image_support(shape,fitted['pose'],evidence,cameras)
                                            fitted['independent_rgb_support']=proof
                                            if not supported_beads:fitted.update(visible=False,reason=proof['reason'])
                                        cache[key]=fitted
                                    return cache[key]
                                # Reuse the shape observer's joint alias and
                                # foreign-evidence withholding gates unchanged.
                                refuse_shape_aliases(trial,evidence,cameras,fit_candidate=fit_hypothesis)
                                supported=trial[name].get('visible') is True
                                if supported:detection=trial[name]
                            if not supported:
                                strict['detections'][name]['appearance_recovery_diagnostic']=dict(
                                    accepted=False,independent_rgb_support=diagnostic,
                                    reason='current RGB recovery lacks unique independently attributed bead support')
                                continue
                        support=detection.get('center_support',[])
                        residual=float(detection.get('median_residual_m',float('inf')))
                        rms=float(detection.get('rms_residual_m',float('inf')))
                        accepted=bool(board_ok and detection.get('visible')
                                      and len(support)>=4 and all(float(v)>=.6 for v in support)
                                      and residual<.003 and rms<.004
                                      and (name=='red' or name in plain.get('poses',{})))
                        if not accepted:
                            continue
                        merged=dict(detection)
                        merged['identity_source']='plain RGB/CAD seeded fallback'
                        if name=='red':
                            merged['optimizer_proposal_source']="previous['red']; optimizer only, never a measurement" if previous and name in previous else 'current RGB initialization only'
                            merged['shape_hypothesis_consistency_checked']=True
                        merged['bounded_rgb_source']=dict(
                            source='fresh_rgb_calibrated_cad_fit',
                            current_board_measured=True,
                            support_views=int(sum(float(v)>=.6 for v in support)),
                            support_by_view=[float(v) for v in support],
                            median_ray_fit_residual_m=residual,
                            rms_ray_fit_residual_m=rms,
                            pose_fitted_from_current_rgb=True,
                            pose_not_copied_from_public_cad=True,
                            q16_fk_required=True)
                        strict.setdefault('poses',{})[name]=detection['pose'] if name=='red' else plain['poses'][name]
                        strict.setdefault('detections',{})[name]=merged
                        strict.setdefault('bounded_rgb_source',{})[name]=dict(merged['bounded_rgb_source'])
                    if strict.get('bounded_rgb_source'):
                        strict['bounded_rgb_source']['board']=dict(
                            source='fresh_rgb_board_fit', current_board_measured=True,
                            frame_bound=True)
                    return strict
    return guarded_estimator(images,cameras,previous)
