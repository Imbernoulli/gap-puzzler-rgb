"""Choose the scene reference from accepted RGB without editing its packet."""
import numpy as np
from . import model, visual_memory


# Board-local case outline (visual_memory._cloud corners) and the margins for
# trusting table observations after the case itself has shifted.
CASE_X, CASE_Y = (.0275 - .110, .0275 + .110), (-.1575, .1575)
CASE_SHIFT_ENVELOPE_M = .004  # the goal envelope used for seated pieces
LOOSE_CLEARANCE_M = .020


def loose_clearance(board, poses, names):
    """Smallest gap between the case outline and any listed loose piece's beads."""
    from scipy.spatial.transform import Rotation
    board = np.asarray(board, dtype=float)
    to_local = Rotation.from_quat(board[3:]).as_matrix().T
    gaps = []
    for name in names:
        pose = np.asarray(poses[name], dtype=float)
        beads = model.local_beads(name) @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3]
        local = (beads - board[:3]) @ to_local.T
        dx = np.maximum(np.maximum(CASE_X[0] - local[:, 0], local[:, 0] - CASE_X[1]), 0.)
        dy = np.maximum(np.maximum(CASE_Y[0] - local[:, 1], local[:, 1] - CASE_Y[1]), 0.)
        gaps.append(float(np.min(np.hypot(dx, dy)) - model.RADIUS))
    return min(gaps) if gaps else float('inf')


def select_reference(observed, prior, *, now, max_age_s, board_limit_m, refusal, loose=None):
    """A complete current packet supersedes persistence, not safety envelopes.

    The historical reference and its age remain diagnostics. Partial current
    RGB still requires the original historical age and board displacement gate.
    `now` can precede the RGB call; it never dates the complete current packet.
    """
    raw = observed.get('poses', {})
    timestamp = observed.get('timestamp')
    complete = (observed.get('observation_source') == 'rgb_shape_geometry'
                and observed.get('is_current_rgb') is not False
                and isinstance(timestamp, (int, float)) and np.isfinite(timestamp)
                and bool(observed.get('frame_sha256'))
                and {'board', *model.CELLS}.issubset(raw))
    if complete:
        for name in ('board', *model.CELLS):
            visual_memory._pose(raw[name])
    board_measured = 'board' in raw
    board = np.asarray(raw['board'] if board_measured else prior['poses']['board'])
    old_age = float(now-prior['timestamp'])
    displacement = visual_memory.shape_displacement('board', prior['poses']['board'], board)
    if not complete:
        if not 0 <= old_age <= max_age_s:
            raise refusal('Historical unplaced occupancy exceeds bounded age scope')
        case_shift = None
        if displacement >= board_limit_m:
            # The old pose of a loose piece on the table stays valid when only
            # the case has moved, by less than the goal envelope, and the
            # case is nowhere near that piece. Seated pieces are planned
            # relative to the fresh case pose. Anything else still refuses.
            names = [n for n in (loose or ()) if n in prior['poses']]
            clearance = loose_clearance(board, prior['poses'], names) if loose is not None else None
            if (loose is None or displacement >= CASE_SHIFT_ENVELOPE_M or len(names) != len(loose)
                    or clearance < LOOSE_CLEARANCE_M):
                raise refusal('Fresh board contradicts historical occupancy reference')
            case_shift = dict(displacement_m=float(displacement), loose_clearance_m=float(clearance),
                              envelope_m=CASE_SHIFT_ENVELOPE_M, required_clearance_m=LOOSE_CLEARANCE_M)
    else:
        case_shift = None
    return dict(mode='current_complete_reference' if complete else 'historical_partial_reference',
                case_shift_accepted=case_shift,
                reference_timestamp=timestamp if complete else prior['timestamp'],
                reference_frame_sha256=observed['frame_sha256'] if complete else prior['frame_sha256'],
                historical_prior_timestamp=prior['timestamp'], historical_prior_age_s=old_age,
                historical_board_displacement_m=float(displacement),
                historical_board_limit_m=float(board_limit_m),
                historical_persistence_required=not complete)
