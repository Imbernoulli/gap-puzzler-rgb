"""Independent audit of recorded poses, without the environment's grade helper."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from .model import local_beads,ORIGIN


def audit(result,frames):
    targets={n:np.asarray(v) for n,v in result['target_beads_local'].items()}
    previous=None;records=[]
    for frame in frames:
        t=frame['step']/result['control_hz']
        board=np.asarray(frame['poses']['board']);rotation=Rotation.from_quat(board[3:]).as_matrix()
        world={};errors={}
        for name,target in targets.items():
            pose=np.asarray(frame['poses'][name])
            world[name]=local_beads(name)@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]
            local=(world[name]-board[:3])@rotation
            distance=np.linalg.norm(local[:,None]-target[None,:],axis=2)
            i,j=linear_sum_assignment(distance);errors[name]=float(distance[i,j].max())
        speeds=[]
        if previous is not None:
            dt=t-previous[0]
            speeds=[float(np.linalg.norm(world[n]-previous[1][n],axis=1).max()/dt) for n in targets]
            speeds.append(float(np.linalg.norm(board[:3]-previous[2][:3])/dt))
        bead_cloud=np.concatenate(list(world.values()))
        hands=all(h[7]>.045 and np.min(np.linalg.norm(bead_cloud-np.asarray(h[:3]),axis=1))>.055 for h in frame['hands'])
        ok=(bool(speeds) and max(speeds)<.015 and max(errors.values())<.004 and hands
            and abs(board[2]-ORIGIN[2]-.003)<.006 and rotation[2,2]>.995
            and np.isfinite(bead_cloud).all())
        records.append((t,bool(ok),max(errors.values()),max(speeds,default=float('inf'))))
        previous=(t,world,board)
    tail=[]
    for record in reversed(records):
        if not record[1]:break
        tail.append(record)
    duration=tail[0][0]-tail[-1][0] if tail else 0.
    return dict(success=bool(duration>=1.-1e-8),released_stable_duration_s=duration,
                final_max_bead_error_m=records[-1][2] if records else None,
                final_max_bead_speed_mps=records[-1][3] if records else None,
                method='Recorded rigid poses, Hungarian bead matching, finite-difference bead speeds and recorded hand clearance; no environment grader calls.')
