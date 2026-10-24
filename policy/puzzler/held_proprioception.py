"""Robot encoder contradictions invalidate held assumptions without contact GT."""
import numpy as np


def retained_jaw_evidence(source_q16, current_q16, arm_id):
    source = np.asarray(source_q16, dtype=float)
    current = np.asarray(current_q16, dtype=float)
    if arm_id not in (0, 1) or source.shape != (16,) or current.shape != (16,) or not np.isfinite([source, current]).all():
        raise ValueError('Held contradiction check requires finite synchronous robot q16 packets')
    offset = 0 if arm_id == 0 else 8
    source_width = float(source[offset + 6] - source[offset + 7])
    current_width = float(current[offset + 6] - current[offset + 7])
    contradicted = source_width > .010 and current_width < .004
    return dict(contract='robot_jaw_held_contradiction_v1', source_jaw_width_m=source_width,
        current_jaw_width_m=current_width, loaded_source_minimum_width_m=.010,
        closed_empty_contradiction_width_m=.004, contradicted=bool(contradicted),
        source='Declared robot encoder jaw coordinates only; no object poses or contact state',
        retained_grasp_proven_by_this_check=False)
