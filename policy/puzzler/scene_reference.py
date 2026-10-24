"""Choose the scene reference from accepted RGB without editing its packet."""
import numpy as np
from . import model, visual_memory


def select_reference(observed, prior, *, now, max_age_s, board_limit_m, refusal):
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
        if displacement >= board_limit_m:
            raise refusal('Fresh board contradicts historical occupancy reference')
    return dict(mode='current_complete_reference' if complete else 'historical_partial_reference',
                reference_timestamp=timestamp if complete else prior['timestamp'],
                reference_frame_sha256=observed['frame_sha256'] if complete else prior['frame_sha256'],
                historical_prior_timestamp=prior['timestamp'], historical_prior_age_s=old_age,
                historical_board_displacement_m=float(displacement),
                historical_board_limit_m=float(board_limit_m),
                historical_persistence_required=not complete)
