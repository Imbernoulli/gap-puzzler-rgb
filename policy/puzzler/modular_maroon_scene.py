"""Shared actual-RGB attribution/scene functions; no initializer or replay."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from . import model
from .vision_stages import StageRefusal,_plain,_observation
from .vision_observer import goals

def _output(route,state,observed,kind,**extra):
    return _plain(dict(route=route,state={**state,'observation':_observation(observed),'latest_rgb':deepcopy(observed),'last_stage':kind,'last_outcome':route},evidence=dict(kind=kind,timestamp=observed['timestamp'],frame_sha256=observed['frame_sha256'],**extra)))

def _owned(seen):
    q=seen['robot_geometry_provenance']
    return dict(mode='pyramid',piece='maroon',pose=deepcopy(seen['poses']['maroon']),timestamp=seen['timestamp'],frame_sha256=deepcopy(seen['frame_sha256']),robot_encoder_values=q['robot_encoder_values'],robot_encoder_sha256=q['robot_encoder_sha256'],observation_source='rgb_shape_geometry',source_step=q['step'])

def _scene(seen,history,p,name,belief):
    if '_common_belief_ledger'in p:
        from .modular_controller import scene_belief
        common=dict(piece=name,parameters=p,rgb_history=history,belief_ledger=deepcopy(p['_common_belief_ledger']),initialized_prefix=p['_common_given_prefix'],executed_placements=p['_common_executed'],released_placements=p.get('_common_released',[]))
        scene=scene_belief(common,seen);p['_common_belief_ledger']=deepcopy(scene['belief_ledger'])
        if name not in seen['poses']:scene['occupancy'][name]=dict(source='conditional_owned_RGB_FK_held',pose_hypotheses=[belief['predicted_body_pose']],inflation_m=belief['uncertainty_m'])
        return scene,np.asarray(scene['board_for_planning']['pose'])
    board=seen['poses'].get('board');raw=seen['poses'];prior=next((r for r in reversed(history)if'board'in r['poses']),None)
    if prior is None:raise StageRefusal('No measured RGB board history')
    if board is None:
        if belief['current_timestamp']-prior['timestamp']>25.:raise StageRefusal('Historical board bound expired')
        board=prior['poses']['board'];margin=.004
    else:margin=0.
    board=np.asarray(board);br=Rotation.from_quat(board[3:]).as_matrix();public=goals('pyramid',p['pyramid_plan']);occupancy={}
    for item in model.CELLS:
        if item in p['conditional_prefix']:
            center,rr=public[item];options=[[*list(board[:3]+br@center),*Rotation.from_matrix(br@rr).as_quat().tolist()]]
            if item in raw:options.append(raw[item])
            source='conditional_declared_prefix_public4mm';inflation=.004+margin
        elif item in raw:options=[raw[item]];source='current_rgb_measurement';inflation=0.
        elif item==name:options=[belief['predicted_body_pose']];source='conditional_owned_RGB_FK_held6mm';inflation=.006
        else:
            old=next((r for r in reversed(history)if item in r['poses']),None)
            if old is None or belief['current_timestamp']-old['timestamp']>25.:raise StageRefusal('No bounded historical occupancy for '+item)
            options=[old['poses'][item]];source='dated_recent_RGB_unplaced_assumption4mm';inflation=.004
        occupancy[item]=dict(source=source,pose_hypotheses=options,inflation_m=inflation)
    return dict(contract='maroon_actualRGB_attributed_geometry_v1',measurement=_observation(seen),occupancy=occupancy,board_for_planning=dict(pose=board.tolist(),uncertainty_m=margin,source='currentRGB'if'board'in raw else'datedRGB_bound4mm')),board
