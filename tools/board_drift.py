"""Case-pose drift in one episode: the policy's RGB fit versus the recorded physics.

  python tools/board_drift.py RUN_DIR OUT.json

RUN_DIR is the inner run directory (contains episode/ and native_pool/). The
physical poses are read offline for this figure only; the policy never sees them.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

CORNERS = np.array([[.0275 + sx * .110, sy * .1575, .013] for sx in (-1, 1) for sy in (-1, 1)])


def corners(pose):
    pose = np.asarray(pose, float)
    return CORNERS @ Rotation.from_quat(pose[3:]).as_matrix().T + pose[:3]


def shift_mm(a, b):
    return round(1000 * float(np.linalg.norm(corners(a) - corners(b), axis=1).max()), 3)


run = Path(sys.argv[1])
observations = json.loads((run / 'episode/rgb/observations.json').read_text())
steps = np.load(next(run.glob('native_pool/*/shards/00/0000/steps.npz')))
times, board = steps['sim_time'], steps['obj_board']
first = observations[0]['poses']['board']
rows = []
for row in observations:
    if 'board' not in row['poses']:
        continue
    i = min(int(np.searchsorted(times, row['timestamp'])), len(times) - 1)
    rows.append(dict(t=round(row['timestamp'], 2), rgb_mm=shift_mm(first, row['poses']['board']),
                     physical_mm=shift_mm(board[0], board[i]), rgb_error_mm=shift_mm(board[i], row['poses']['board'])))
Path(sys.argv[2]).write_text(json.dumps(dict(source=str(run), receipt_threshold_mm=1.5, goal_envelope_mm=4.0,
                                             rows=rows)) + '\n')
print(len(rows), 'frames; max RGB error', max(r['rgb_error_mm'] for r in rows), 'mm')
