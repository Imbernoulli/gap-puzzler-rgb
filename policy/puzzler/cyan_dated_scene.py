"""Authentic per-entry RGB occupancy with the original bounded persistence.

Different actual packets can observe different entries. Their dates are never
merged: measurement stays the current raw packet, every occupied hypothesis
retains its source packet, and the scene clock is the oldest USED RGB epoch.
Conditional prefix goals remain explicitly conditional with their original
4mm envelope; the held Cyan still uses its separate original6mm contract.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation

from . import cyan_geometry as g,model,visual_memory,vision_controller as c


def _metadata(packet):
    return dict(timestamp=packet['timestamp'],frame_sha256=deepcopy(packet['frame_sha256']),
                observation_source=packet['observation_source'])


def build(state,observed):
    p=state['parameters'];now=float(observed['timestamp'])
    limit=min(30.,float(p.get('historical_max_age_s',30.)))
    if (not np.isfinite(now) or not np.isfinite(limit) or limit<=0
            or observed.get('observation_source')!='rgb_shape_geometry'
            or observed.get('is_current_rgb') is False or not observed.get('frame_sha256')):
        raise c.PlanningRefusal('Dated environment requires finite original30s scope')
    history=[*state.get('rgb_history',()),observed]
    # Only original actual accepted packets are eligible. In particular, a
    # propagated held pose and a public prefix goal cannot enter this list.
    packets=[]
    for packet in history:
        timestamp=packet.get('timestamp')
        if (packet.get('observation_source')!='rgb_shape_geometry'
                or packet.get('is_current_rgb') is False
                or not packet.get('frame_sha256')
                or not isinstance(timestamp,(int,float)) or not np.isfinite(timestamp)
                or not 0<=now-timestamp<=limit):continue
        packets.append(packet)
    packets.sort(key=lambda row:row['timestamp'],reverse=True)

    def latest(name,required=True):
        for packet in packets:
            if name in packet.get('poses',{}):
                visual_memory._pose(packet['poses'][name])
                return packet
        if required:raise c.PlanningRefusal('No actual bounded RGB occupancy for '+name)
        return None

    board_packet=latest('board');board=np.asarray(board_packet['poses']['board'])
    # Preserve the original absolute anchor. A succession of sub2mm board
    # updates cannot make an old occupancy baseline drift without a bound.
    previous=p['historical_complete_rgb']['poses']['board']
    displacement=float(visual_memory.shape_displacement('board',previous,board))
    if displacement>=.002:
        raise c.PlanningRefusal('Actual per-entry board contradicts original2mm persistence scope')
    margin=0. if board_packet['timestamp']==now else .004
    Rb=Rotation.from_quat(board[3:]).as_matrix()
    prefix=set(p.get('conditional_prefix',()))|set(state.get('executed_placements',()))
    public=g.goals('pyramid',p['pyramid_plan']);occupancy={};sources={'board':_metadata(board_packet)}
    for name in model.CELLS:
        if name=='cyan':
            # Every delivery/material query excludes the held target from
            # scene occupancy and receives its independently bounded body
            # attachment explicitly. Do not add an unused old target pose to
            # the environment clock or relabel it as a static measurement.
            continue
        # Prefix objects retain their independently initialized/earned4mm
        # condition. Union every ACTUAL CURRENT hypothesis as before; old
        # prefix pixels stay in the archive and are never dated as current.
        packet=(observed if name in observed.get('poses',{}) else None) if name in prefix else latest(name)
        if name in prefix:
            center,rotation=public[name]
            hypotheses=[np.r_[board[:3]+Rb@center,Rotation.from_matrix(Rb@rotation).as_quat()].tolist()]
            source='explicit_conditional_prefix_public_goal4mm'
            if packet is not None:
                visual_memory._pose(packet['poses'][name])
                hypotheses.append(deepcopy(packet['poses'][name]))
                sources[name]=_metadata(packet)
                source+='_union_actual_dated_RGB'
            inflation=.004+margin
            entry=dict(source=source,pose_hypotheses=hypotheses,inflation_m=inflation,
                conditional_goal_source='Original initialized/earned prefix witness; never a current measured pose')
        else:
            sources[name]=_metadata(packet)
            current=packet['timestamp']==now
            entry=dict(source='current_rgb_measurement' if current else 'explicit_dated_unplaced4mm',
                pose_hypotheses=[deepcopy(packet['poses'][name])],inflation_m=0. if current else .004)
        if packet is not None:entry['actual_rgb_source']=_metadata(packet)
        entry['shape']='public CAD sphere cover';occupancy[name]=entry
    oldest=min(sources.values(),key=lambda row:row['timestamp'])
    reference=dict(mode='per_entry_dated_actual_RGB',reference_timestamp=oldest['timestamp'],
        reference_frame_sha256=deepcopy(oldest['frame_sha256']),
        observation_source='rgb_shape_geometry',actual_entry_sources=deepcopy(sources),
        oldest_used_actual_RGB_age_s=now-oldest['timestamp'],maximum_age_s=limit,
        board_displacement_m=displacement,current_measurement_was_not_filled=True)
    scene=dict(target='cyan',reference=reference,measurement=g.rgb_record(observed),occupancy=occupancy,
        board_for_planning=dict(pose=board.tolist(),uncertainty_m=margin,
            source='current_RGB' if margin==0 else 'explicit_dated_RGB_persistence',
            actual_rgb_source=_metadata(board_packet)))
    return scene
