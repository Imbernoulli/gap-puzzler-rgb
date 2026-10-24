import json
import numpy as np
from scipy.spatial.transform import Rotation
from . import vision_controller as c
PlanningRefusal=c.PlanningRefusal

def clear_release_height(name,poses,target_tcp,q,p,*,planning_scene=None):
    from puzzler.finite_release_scene import finite_clearance
    if planning_scene is None:
        raise PlanningRefusal('Finite release screening requires attributed RGB occupancy')
    R=Rotation.from_quat(c.qarr(q)).as_matrix()
    candidates=p.get('finite_release_heights_m',[.002,.004,.006,.008,.010,.012,.016,.020,.024,.030,.040,.050,.060])
    rows=[]
    for height in candidates:
        if not p.get('release_height',.002)-1e-9<=height<=p.get('max_release_height',.060)+1e-9:continue
        position=np.asarray(target_tcp)+[0,0,float(height)]
        checks=[finite_clearance(position,R,float(width),planning_scene,exclude=(name,))
                for width in np.linspace(p.get('release_start_width',.024),p.get('release_open_width',.032),5)]
        worst=min(checks,key=lambda x:x['clearance_m'])
        rows.append(dict(height_m=float(height),clearance_m=worst['clearance_m'],witness=worst.get('witness')))
        if worst['clearance_m']>=p.get('hand_clearance_margin',.0005):return float(height)
    raise PlanningRefusal('All finite-board actual-jaw release candidates refused: '+json.dumps(rows))
