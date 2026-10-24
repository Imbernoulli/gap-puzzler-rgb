"""Public YAM-only CUDA model; no environment, object or state constructor."""
import numpy as np
import newton
import warp as wp
from robosimstudio.robots.embodiments.yam_bimanual import YAM_BIMANUAL
from robosimstudio.robots.entity import EntityEmbodiment

PUBLIC_MOUNT=(.2525,0.,.772,0.,0.,0.,1.)

def build_public_robot(device='cuda:0'):
    wp.init(); target=wp.get_device(device)
    if not target.is_cuda:raise ValueError('Robot-only query requires explicit CUDA')
    robot=EntityEmbodiment(YAM_BIMANUAL,pose=PUBLIC_MOUNT)
    builder=newton.ModelBuilder();robot.build_into(builder,None);robot.apply_actuation(builder)
    model=builder.finalize(device=target);robot.bind(model)
    if model.joint_coord_count!=16:raise ValueError('Expected public robot-only q16 model')
    return robot,model

def static_shape_metadata(model):
    """Geometry only. Retain model itself for live mesh-handle queries.

    Numeric mesh pointers are deliberately not exported across processes.
    """
    result={}
    for key in ('shape_type','shape_scale','shape_transform','shape_body','body_label','shape_label'):
        value=getattr(model,key,None)
        if hasattr(value,'numpy'):value=value.numpy()
        if isinstance(value,np.ndarray):value=value.tolist()
        elif value is not None:value=list(value)
        result[key]=value
    kinds=np.asarray(model.shape_type.numpy(),int)
    ids,counts=np.unique(kinds,return_counts=True)
    result['shape_type_counts']={str(int(i)):int(c)for i,c in zip(ids,counts)}
    result['public_robot_mount']=list(PUBLIC_MOUNT)
    result['shape_sources']=[type(value).__name__ for value in getattr(model,'shape_source',[])]
    return result
