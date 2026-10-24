"""Finite public board CAD, real table/base and observed rigid occupancy.

The old plane at board+13mm extended the board rim across the entire world.
This query retains the real support plane and every collision mesh of the
board, including its rim, sockets, hinge and lid. No environment is accessed.
"""
import numpy as np
from . import conditional_scene
from .gripper_geometry import clearance, scene_clearance

_query = None


def finite_clearance(position, rotation, width, scene, *, exclude=(), board_margin=0.):
    global _query
    board_info = scene.get('board_for_planning', {})
    board = np.asarray(board_info['pose'] if 'pose' in board_info else scene['measurement']['poses']['board'])
    margin = max(float(board_margin), float(board_info.get('uncertainty_m', 0.)))
    centers, radii = conditional_scene.spheres(scene, exclude=exclude)
    other = clearance(position, rotation, width, centers, radius=radii)
    table = scene_clearance(position, rotation, width, [],
        table_z=float(board[2]-.003+margin), plane_z=float(board[2]-.003+margin), board=None)
    cheap = min((other, table), key=lambda x: x['clearance_m'])
    if cheap['clearance_m'] < .0005:
        return {**cheap, 'query_kind':'real_table_base_and_rgb_occupancy', 'finite_board_not_needed':True}
    if _query is None:
        from .public_board_clearance import PublicBoardQuery
        _query = PublicBoardQuery()
    finite = _query.query(board, position, rotation, width)
    finite = {**finite, 'clearance_m':float(finite['clearance_m'])-margin,
              'query_kind':'all_79_finite_public_board_components'}
    return min((other, table, finite), key=lambda x: x['clearance_m'])
