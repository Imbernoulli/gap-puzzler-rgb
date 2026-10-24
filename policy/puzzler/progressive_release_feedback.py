"""One feedback-dependent separated release step using only public geometry.

Caller supplies measured robot FK/jaw before EVERY invocation. The returned
width/position is a proposal, never evidence that it was achieved. This module
accepts no environment, evaluator, contacts, teacher commands or world seeds.
It deliberately refuses a negative initial separation; intentional support
contact needs a distinct reviewed contract, not a hidden margin waiver.
"""
import hashlib
import json
import numpy as np
from puzzler import conditional_scene
from puzzler.gripper_geometry import clearance, scene_clearance


def _plain(x):
    return json.loads(json.dumps(x, default=lambda a: a.tolist() if isinstance(a, np.ndarray) else float(a), allow_nan=False))


def _sha(x):
    return hashlib.sha256(json.dumps(_plain(x), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def next_release_step(*, actual_position, actual_rotation, actual_width,
                      release_origin, retreat_vector, final_width, min_release_width,
                      scene, piece, board, proprio_provenance, board_query,
                      margin=.0005, move_step=.003, width_step=.002,
                      sample_translation=.0005, sample_width=.0005):
    """Return one open OR withdraw step, then require another proprio read.

    ``min_release_width`` is an explicit public grasp parameter, not an
    observed attachment/contact conclusion. It prevents withdrawal with a
    still-closed gripper. The caller must state its physical release scope.
    Every proposed width interval and translation segment is sampled. This
    proves sampled hand-CAD separation only, never whole-arm motion or release.
    """
    p=np.asarray(actual_position,float);r=np.asarray(actual_rotation,float)
    origin=np.asarray(release_origin,float);retreat=np.asarray(retreat_vector,float)
    b=np.asarray(board,float);w=float(actual_width);end=float(final_width)
    if (p.shape!=(3,)or r.shape!=(3,3)or origin.shape!=(3,)or retreat.shape!=(3,)or b.shape!=(7,)
        or not np.isfinite(np.r_[p,r.ravel(),origin,retreat,b,w,end,min_release_width,margin,move_step,width_step,sample_translation,sample_width]).all()
        or not np.allclose(r.T@r,np.eye(3),rtol=0,atol=1e-5)or abs(np.linalg.det(r)-1)>1e-5
        or not 0<=w<=.095 or not 0<=end<=.095
        or not 0<=min_release_width<=end or min(margin,move_step,width_step,sample_translation,sample_width)<=0):
        raise ValueError('Invalid progressive-release measured inputs')
    if abs(np.linalg.norm(b[3:])-1)>1e-5 or not callable(getattr(board_query,'query',None)):
        raise ValueError('Unit board pose and complete public finite-board query are required')
    if not isinstance(proprio_provenance,dict)or not all(k in proprio_provenance for k in('timestamp','robot_encoder_sha256')):
        raise ValueError('A synchronous actual robot provenance is required')
    length=float(np.linalg.norm(retreat))
    if length<1e-6:raise ValueError('Retreat vector must have positive length')
    centers,radii=conditional_scene.spheres(scene,exclude=(piece,))
    source=dict(actual_position=p,actual_rotation=r,actual_width=w,release_origin=origin,retreat_vector=retreat,final_width=end,min_release_width=min_release_width,scene=scene,board=b,proprio_provenance=proprio_provenance)
    inputs_hash=_sha(source)
    def check(pos,width):
        values=[clearance(pos,r,width,centers,radius=radii),scene_clearance(pos,r,width,[],table_z=float(b[2]-.003),plane_z=float(b[2]-.003),board=b)]
        if board_query is not None:values.append(board_query.query(b,pos,r,width))
        if any(not np.isfinite(v['clearance_m'])for v in values):raise ValueError('Nonfinite public clearance cannot authorize motion')
        worst=min(values,key=lambda v:v['clearance_m'])
        return dict(clearance_m=float(worst['clearance_m']),witness=worst.get('witness'))
    def result(status,**extra):
        return _plain(dict(status=status,input_sha256=inputs_hash,actual_proprio=proprio_provenance,scope='Sampled public handCAD versus all retained non-target occupancy/table/base/board. No target-release, whole-arm or placement proof; no support-contact allowance.',**extra))
    initial=check(p,w)
    if initial['clearance_m']<margin:return result('refused',reason='Actual starting configuration is not separated',initial=initial)
    direction=retreat/length;progress=float(np.clip(np.dot(p-origin,direction),0,length))
    lateral=float(np.linalg.norm((p-origin)-np.dot(p-origin,direction)*direction))
    if lateral>.003:return result('refused',reason='Actual FK left the declared retreat corridor',lateral_error_m=lateral)
    # Open at the measured pose only through a fully sampled interval.
    target_w=min(end,w+width_step)
    if target_w>w+1e-5 and w<end-.0005:
        widths=np.linspace(w,target_w,max(2,int(np.ceil((target_w-w)/sample_width))+1))
        samples=[dict(width_m=float(ww),**check(p,float(ww)))for ww in widths]
        if all(v['clearance_m']>=margin for v in samples):
            return result('command',operation='open',command=dict(width_m=float(target_w)),samples=samples,needs_new_proprio=True)
    # A partial opening can precede withdrawal. Never infer it from the last
    # requested command: only the newly measured jaw controls this branch.
    if w+1e-6<min_release_width:
        return result('refused',reason='Opening sweep blocked before explicit minimum release width',initial=initial,actual_width_m=w,min_release_width_m=min_release_width)
    distance=min(move_step,max(0,length-progress))
    if distance<=.0005 and w>=end-.0005:return result('completed',progress_m=progress,actual_width_m=w,initial=initial)
    if distance<=.0005:return result('refused',reason='Withdrawal complete but measured final opening not achieved',actual_width_m=w,final_width_m=end)
    target=origin+direction*(progress+distance)
    points=np.linspace(p,target,max(2,int(np.ceil(np.linalg.norm(target-p)/sample_translation))+1))
    samples=[dict(position=pos.tolist(),**check(pos,w))for pos in points]
    if not all(v['clearance_m']>=margin for v in samples):return result('refused',reason='Next measured-width withdrawal segment blocked',samples=samples)
    return result('command',operation='withdraw',command=dict(position=target.tolist(),rotation=r.tolist()),samples=samples,needs_new_proprio=True)
