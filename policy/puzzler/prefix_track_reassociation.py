"""Current RGB compatibility for an already released, conditional prefix track.

This never creates a globally unique measured pose or a new placement receipt.
The static prefix promise supplies a 4-mm board-relative public-goal prior.
Board world estimation adds 4 mm and current object RGB adds 6 mm, for a
14-mm innovation range. A conditional MAP choice preserves other supported
hypotheses; it never means that those alternatives were physically disproved.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation

from . import model, visual_memory
from .vision import detect, fit_piece
from .shape_vision import (partial_shape_aliases, raw_image_evidence,
                           bead_image_support)

CONTRACT = 'conditional_prefix_current_rgb_reassociation_v1'
RGB_BOUND_M = .006
PREFIX_BOUND_M = .004
BOARD_BOUND_M = .004
EQUIVALENCE_M = .003
SOURCE = 'conditional_executed_goal_MAP_persistence_current_rgb'


def select_cluster(name, candidates, goal_pose):
    """Choose the nearest prior-compatible fit; preserve every alternative."""
    compatible = []
    for index, row in enumerate(candidates):
        if not row.get('pixel_supported'):
            continue
        pose = row['fit']['pose']
        innovation = visual_memory.shape_displacement(name, goal_pose, pose)
        if innovation <= RGB_BOUND_M + PREFIX_BOUND_M + BOARD_BOUND_M:
            compatible.append(index)
    selected = min(compatible, key=lambda index: visual_memory.shape_displacement(
        name, goal_pose, candidates[index]['fit']['pose'])) if compatible else None
    return selected, compatible


def earned_lineage(state, cursor):
    """Require this Env's own executed release and the original earned receipt."""
    try:
        name = 'maroon'
        if (state.get('piece') != 'red' or name not in state.get('executed_placements', [])
                or name in state.get('initialized_prefix', [])
                or not state['parameters'].get('allow_maroon_conditional_MAP_reassociation', False)):
            return None
        token = state['execution_tokens'][name]
        receipt = token['new_visual_release_receipt']
        track = state['memory']['placements'][name]
        checked = visual_memory.release_receipt(name, receipt['before'], receipt['after'],
            receipt['released_tcp_position'], state['parameters'])
        mark = token.get('mark', token.get('execution_mark', {}))
        pair = token['release_rgb_pair']
        order = state['parameters']['assembly_order']
        if (checked != receipt or track.get('receipt') != checked
                or order.index(name) >= order.index('red')
                or not mark.get('session_nonce') or mark.get('session_nonce') != cursor.get('session_nonce')
                or not isinstance(mark.get('call_id'), int) or mark['call_id'] >= cursor.get('call_id', -1)
                or len(pair) != 2 or pair[0]['timestamp'] >= pair[1]['timestamp']
                or pair[1]['timestamp'] >= state['observation']['timestamp']):
            return None
        for bound, actual in zip((receipt['before'], receipt['after']), pair):
            if visual_memory._observation(actual, name) != bound:
                return None
        history = state.get('rgb_history', [])
        if any(not any(row.get('timestamp') == bound['timestamp']
                       and row.get('frame_sha256') == bound['frame_sha256']
                       for row in history) for bound in (receipt['before'], receipt['after'])):
            return None
        return dict(piece=name, actual_release_mark=deepcopy(mark),
            original_receipt=deepcopy(receipt), same_env_session=cursor['session_nonce'],
            receipt_not_reminted=True, given_fixture_not_used=True)
    except (KeyError, ValueError, TypeError, IndexError, OverflowError, visual_memory.MemoryRefusal):
        return None


def evaluate(name, images, cameras, goal_pose, receipt_pose, conflicting_poses=(),
             current_poses=None):
    """Fit a finite CAD alias family to actual pixels, then associate the track.

    Goal/receipt/old estimates are optimizer proposals. Every accepted fit must
    independently pass the original RGB residuals and all-bead two-view arc
    support. A conditional association says nothing about physical success.
    """
    result = dict(contract=CONTRACT, status='unknown', compatible=False,
        unique_current_measurement=False, physical_precondition_verified_by_image=False,
        bounds_m=dict(rgb=RGB_BOUND_M, prefix_board_relative=PREFIX_BOUND_M,
                      board_world=BOARD_BOUND_M,
                      innovation=RGB_BOUND_M+PREFIX_BOUND_M+BOARD_BOUND_M, equivalence=EQUIVALENCE_M),
        goal_prior_pose=visual_memory._pose(goal_pose).tolist(),
        receipt_proposal_pose=visual_memory._pose(receipt_pose).tolist(),
        scope='Static released-prefix promise plus fresh RGB compatibility; '
              'raw aliases retained, no pose/receipt mint and no simulator truth.')
    if name != 'maroon' or set(images) != {'top', 'front', 'rear', 'side'}:
        return {**result, 'reason': 'Only the explicit Maroon prefix and complete raw RGB rig are supported'}
    masks = {key: detect(image)[name] for key, image in images.items()}
    evidence = raw_image_evidence(masks, cameras)
    proposals = [np.asarray(goal_pose), np.asarray(receipt_pose)]
    proposals.extend(visual_memory._pose(pose) for pose in conflicting_poses)
    # Enumerate the same finite shared-bead CAD aliases as shape perception.
    seeds = []
    for proposal in proposals:
        rr = Rotation.from_quat(proposal[3:]).as_matrix()
        seeds.append(proposal)
        for alias_R, alias_p in partial_shape_aliases(name):
            seeds.append(np.r_[proposal[:3]+rr@alias_p,
                               Rotation.from_matrix(rr@alias_R).as_quat()])
    unique_seeds = []
    for seed in seeds:
        if not any(visual_memory.shape_displacement(name, seed, prior)<1.e-7
                   for prior in unique_seeds):
            unique_seeds.append(seed)
    rows = []
    for seed in unique_seeds:
        fitted = fit_piece(name, masks, cameras, previous=seed)
        foreign = {}
        for other, raw in (current_poses or {}).items():
            if other != name and other in model.CELLS:
                raw = np.asarray(raw)
                foreign[other] = [model.local_beads(other) @ Rotation.from_quat(raw[3:]).as_matrix().T+raw[:3]]
        bare_supported, bare_diagnostic = (bead_image_support(name, fitted['pose'], evidence, cameras)
                                          if fitted.get('visible') else (False, {}))
        supported, diagnostic = (bead_image_support(name, fitted['pose'], evidence, cameras, foreign=foreign)
                                 if fitted.get('visible') else (False, {}))
        disjoint = True
        if supported:
            pose = np.asarray(fitted['pose'])
            cloud = model.local_beads(name) @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3]
            for other, raw in (current_poses or {}).items():
                if other == name or other not in model.CELLS:
                    continue
                raw = np.asarray(raw)
                foreign = model.local_beads(other) @ Rotation.from_quat(raw[3:]).as_matrix().T + raw[:3]
                if float(np.linalg.norm(cloud[:,None]-foreign[None,:], axis=2).min()) < .020:
                    disjoint = False
                    break
        rows.append(dict(seed_proposal=seed.tolist(), fit=fitted,
            unwithheld_pixel_supported=bool(bare_supported),
            unwithheld_per_bead_rgb_support=bare_diagnostic,
            per_bead_rgb_support=diagnostic, current_tracks_disjoint=disjoint,
            pixel_supported=bool(supported and disjoint)))
    selected, compatible = select_cluster(name, rows, goal_pose)
    for row in rows:
        if row['fit'].get('pose') is not None:
            row['innovation_m'] = visual_memory.shape_displacement(name, goal_pose, row['fit']['pose'])
    result.update(candidates=rows, compatible_candidate_indices=compatible,
                  independently_fitted_seed_count=len(rows))
    if selected is None:
        return {**result, 'reason': 'No independently RGB-supported track within the fixed 14-mm innovation range'}
    return {**result, 'status': 'conditional_MAP_track_persistence', 'compatible': True,
            'selected_candidate_index': selected,
            'fitted_pose_diagnostic': deepcopy(rows[selected]['fit']['pose']),
            'selection_rule': 'Nearest public-goal static prior among supported candidates; no residual ranking or claim that alternatives are impossible',
            'reason': 'Conditional MAP track persistence only; global RGB and hidden-prefix ambiguity remain explicit'}


def validate_record(name, record, observed, goal_pose):
    """Validate the current packet binding; this is not a physical checkpoint."""
    try:
        if (name != 'maroon' or record.get('contract') != CONTRACT
                or record.get('status') != 'conditional_MAP_track_persistence'
                or record.get('compatible') is not True
                or record.get('unique_current_measurement') is not False
                or record.get('physical_precondition_verified_by_image') is not False
                or record.get('timestamp') != observed.get('timestamp')
                or record.get('frame_sha256') != observed.get('frame_sha256')
                or record.get('board_pose_at_reassociation') != observed.get('poses',{}).get('board')
                or set(observed.get('frame_sha256', {})) != {'top', 'front', 'rear', 'side'}
                or record.get('bounds_m') != dict(rgb=RGB_BOUND_M, prefix_board_relative=PREFIX_BOUND_M,
                    board_world=BOARD_BOUND_M, innovation=RGB_BOUND_M+PREFIX_BOUND_M+BOARD_BOUND_M,
                    equivalence=EQUIVALENCE_M)
                or visual_memory.shape_displacement(name, record['goal_prior_pose'], goal_pose)>1.e-9):
            return False
        selected, compatible = select_cluster(name, record['candidates'], goal_pose)
        if selected is None or record.get('selected_candidate_index') != selected or record.get('compatible_candidate_indices') != compatible:
            return False
        return record.get('fitted_pose_diagnostic') == record['candidates'][selected]['fit']['pose']
    except (KeyError, ValueError, TypeError, IndexError, OverflowError, visual_memory.MemoryRefusal):
        return False


def scene_track(state, name, public_targets):
    """Revalidate a dated conditional witness without making it fresh RGB."""
    try:
        ledger=state['belief_ledger'][name]
        if (name!='maroon' or ledger.get('status')!='conditional_MAP_track_persistence'
                or not state['parameters'].get('allow_maroon_conditional_MAP_reassociation',False)
                or name not in state.get('executed_placements',[])
                or name in state.get('initialized_prefix',[])):
            return None
        lineage=ledger['conditional_MAP_lineage'];record=ledger['conditional_MAP_reassociation']
        token=state['execution_tokens'][name];receipt=token['new_visual_release_receipt']
        if (lineage['original_receipt']!=receipt
                or state['memory']['placements'][name]['receipt']!=receipt
                or lineage['actual_release_mark']!=token.get('mark',token.get('execution_mark',{}))
                or lineage['same_env_session']!=lineage['actual_release_mark'].get('session_nonce')):
            return None
        packet=next((row for row in reversed(state.get('rgb_history',[])) if row.get('timestamp')==record['timestamp']
                     and row.get('frame_sha256')==record['frame_sha256']
                     and row.get('poses',{}).get('board')==record.get('board_pose_at_reassociation')),None)
        if packet is None or record['timestamp']>state['observation']['timestamp']:
            return None
        board=np.asarray(packet['poses']['board']);rb=Rotation.from_quat(board[3:]).as_matrix()
        center,rotation=public_targets[name]
        goal=np.r_[board[:3]+rb@center,Rotation.from_matrix(rb@rotation).as_quat()]
        return record if validate_record(name,record,packet,goal) else None
    except (KeyError,ValueError,TypeError,IndexError,OverflowError,visual_memory.MemoryRefusal):
        return None
