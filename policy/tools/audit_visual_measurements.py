"""Offline image-estimate audit against saved evaluator frames, never policy input."""
import argparse
import bisect
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation, Slerp

from puzzler import model


def beads(name,pose):
    return model.local_beads(name)@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]


def run(directory):
    root=Path(directory)
    frames=json.loads((root/'frames.json').read_text())
    observations=json.loads((root/'rgb/observations.json').read_text())
    result=json.loads((root/'result.json').read_text())
    steps=[f['step'] for f in frames];rows=[];moving=0
    for oi,obs in enumerate(observations):
        step=obs['timestamp']*result['control_hz']
        hi=bisect.bisect_left(steps,step)
        if hi==0 or hi>=len(frames):continue
        lo=hi-1
        fraction=(step-steps[lo])/(steps[hi]-steps[lo])
        for name,pose in obs['poses'].items():
            a=np.asarray(frames[lo]['poses'][name]);b=np.asarray(frames[hi]['poses'][name])
            if name=='board':
                motion=float(np.linalg.norm(a[:3]-b[:3]))
            else:
                motion=float(np.linalg.norm(beads(name,a)-beads(name,b),axis=1).max())
            if motion>.001:
                moving+=1;continue  # Avoid certifying interpolation at moving/contact events.
            true_p=(1-fraction)*a[:3]+fraction*b[:3]
            true_q=Slerp([0,1],Rotation.from_quat([a[3:],b[3:]]))([fraction]).as_quat()[0]
            estimate=np.asarray(pose)
            if name=='board':
                error=float(np.linalg.norm(estimate[:3]-true_p))
            else:
                aa=beads(name,estimate);bb=beads(name,np.r_[true_p,true_q])
                distances=np.linalg.norm(aa[:,None]-bb[None,:],axis=2)
                ii,jj=linear_sum_assignment(distances);error=float(distances[ii,jj].max())
            rows.append(dict(observation=oi,timestamp=obs['timestamp'],piece=name,error_m=error,bracket_motion_m=motion))
    rows.sort(key=lambda r:-r['error_m'])
    report=dict(scope='Offline accuracy of accepted image estimates, using only nearly stationary frame brackets; no policy feedback.',
                scored=len(rows),unscored_moving=moving,max_error_m=rows[0]['error_m'] if rows else None,
                accepted_over_3mm=sum(r['error_m']>.003 for r in rows),worst=rows[:30],measurements=rows)
    (root/'visual_measurement_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='measurements'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');run(parser.parse_args().directory)
