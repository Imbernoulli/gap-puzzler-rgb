"""Whole-pyramid completion from current RGB, public CAD and the plan.

The finish node asks: if the pyramid is complete, which bead sites must each
calibrated camera see, and does each of those pixels show the right part? The
site positions come from the RGB-fitted case pose and the public pyramid plan;
occlusion is computed from that same planned geometry. No simulator state,
receipt or evaluator output enters this check.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from . import model

# Shaded Red and Maroon beads overlap in chroma; either label confirms the
# other's site, but neither confirms any third part.
CONFUSABLE = {'red': {'maroon'}, 'maroon': {'red'}}
MIN_SHARE = .6  # fraction of a piece's decisive views that must match
MAX_MISMATCH = .15  # fraction of all decisive views that may disagree


def _labels(rgb):
    """Nearest palette part per pixel, -1 where the pixel is not a lit part colour."""
    a = rgb.astype(np.float32) / 255
    norm = np.linalg.norm(a, axis=2)
    chroma = a / np.maximum(norm[..., None], 1e-6)
    names = list(model.CELLS)
    palette = np.array([model.COLORS[n] for n in names], dtype=np.float32)
    palette /= np.linalg.norm(palette, axis=1, keepdims=True)
    scores = chroma @ palette.T
    labels = scores.argmax(axis=2)
    valid = (scores.max(axis=2) > .985) & ((a.max(axis=2) - a.min(axis=2)) > .10) & (norm > .12)
    return np.where(valid, labels, -1), names


def planned_sites(board_pose, targets):
    board = np.asarray(board_pose, dtype=float)
    rotation = Rotation.from_quat(board[3:]).as_matrix()
    rows = []
    for name, (center, local) in targets.items():
        for bead in model.local_beads(name) @ np.asarray(local).T + center:
            rows.append((name, rotation @ bead + board[:3]))
    return rows


def evaluate(images, cameras, board_pose, targets, patch=2, pieces=None):
    sites = planned_sites(board_pose, targets)
    points = np.array([p for _, p in sites])
    names_by_site = [n for n, _ in sites]
    per_piece = {name: dict(visible=0, match=0, mismatch=0, unclear=0) for name in targets
                 if pieces is None or name in pieces}
    views = []
    for key, rgb in images.items():
        camera = cameras[key]
        labels, palette_names = _labels(rgb)
        height, width = labels.shape
        origin = camera.camera_to_world[:3, 3]
        offsets = points - origin
        distance = np.linalg.norm(offsets, axis=1)
        direction = offsets / distance[:, None]
        pixels = camera.project(points)
        for i, (name, point) in enumerate(sites):
            if pieces is not None and name not in pieces:
                continue
            u, v = pixels[i]
            if not (patch <= u < width - patch and patch <= v < height - patch):
                continue
            along = offsets @ direction[i]
            perpendicular = np.linalg.norm(offsets - along[:, None] * direction[i], axis=1)
            # Another planned bead whose sphere crosses the sight line in
            # front of this centre hides it (1 mm margin: skip, don't guess).
            blocking = (np.arange(len(points)) != i) & (along < distance[i]) & (perpendicular < model.RADIUS + .001)
            if blocking.any():
                continue
            window = labels[int(v) - patch:int(v) + patch + 1, int(u) - patch:int(u) + patch + 1].ravel()
            seen = window[window >= 0]
            per_piece[name]['visible'] += 1
            if len(seen) < window.size // 2:
                outcome, label = 'unclear', None
            else:
                label = palette_names[int(np.bincount(seen).argmax())]
                outcome = 'match' if label == name or label in CONFUSABLE.get(name, ()) else 'mismatch'
            per_piece[name][outcome] += 1
            views.append(dict(camera=key, piece=name, pixel=[round(float(u), 1), round(float(v), 1)],
                              outcome=outcome, label=label))
    decisive = sum(r['match'] + r['mismatch'] for r in per_piece.values())
    mismatches = sum(r['mismatch'] for r in per_piece.values())
    unconfirmed = sorted(name for name, r in per_piece.items()
                         if r['match'] + r['mismatch'] > 0
                         and (r['match'] == 0 or r['match'] < MIN_SHARE * (r['match'] + r['mismatch'])))
    hidden = sorted(name for name, r in per_piece.items() if r['match'] + r['mismatch'] == 0)
    complete = bool(decisive and not unconfirmed and mismatches <= MAX_MISMATCH * decisive)
    return dict(complete=complete, per_piece=per_piece, decisive_views=decisive, mismatched_views=mismatches,
                unconfirmed_pieces=unconfirmed, pieces_without_decisive_view=hidden, views=views,
                rule=f'each visible piece: matches >= {MIN_SHARE:g} of decisive views and >= 1; all pieces: '
                     f'mismatches <= {MAX_MISMATCH:g} of decisive views',
                source='current RGB, calibrated cameras, RGB case pose, public CAD and pyramid plan')


def goal_support(images, cameras, board_pose, targets, names):
    """Do the current pixels at a piece's goal bead sites show that piece?

    A fitted pose that disagrees with the goal is only a contradiction if the
    goal pose fails to explain the image too. Occlusion uses the completed
    plan, so a site that later layers would hide is never tested.
    """
    rows = evaluate(images, cameras, board_pose, targets, pieces=set(names))['per_piece']
    return {name: dict(row, consistent=bool(row['match'] >= 1 and row['match'] >= MIN_SHARE * (row['match'] + row['mismatch'])))
            for name, row in rows.items()}
