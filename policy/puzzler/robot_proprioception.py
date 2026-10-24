"""Robot-only synchronous full encoder/FK packet, including passive jaw mirror."""
import hashlib
import numpy as np
from .vision_planning import robot_seed,finite_fk

def read_robot_proprioception(env,arm_id=1):
 if arm_id not in(0,1):raise ValueError('Unknown public arm')
 coordinates=tuple(sorted({int(i)for a in env.handles.arms.values()for i in(*a.arm_coords,*a.gripper_coords)}|{int(i)for name in env.handles.arms for i in(env.gripper(name).width_coords or())}))
 if coordinates!=tuple(range(16)):raise ValueError('This pinned public robot must declare complete16 encoder coordinates')
 # Slice BEFORE readback, exactly as the existing audited held-belief adapter.
 # robot_seed by itself omits passive finger-mirror coordinates.
 current=np.asarray([float(env.state.joint_q[i:i+1].numpy()[0])for i in coordinates],np.float32);seed=robot_seed(env);seed[:16]=current;name='left'if arm_id==0 else'right';pp,qq=finite_fk(env.arm(name),seed);wc=tuple(env.gripper(name).width_coords)
 if len(wc)!=2:raise ValueError('Pinned public YAM width requires both mirrored axes')
 return dict(timestamp=float(env.step_count/env.pipeline.control_hz),step=int(env.step_count),robot_encoder_values=current.tolist(),robot_encoder_sha256=hashlib.sha256(current.tobytes()).hexdigest(),tcp=[*map(float,pp),*map(float,qq)],width_m=float(current[wc[0]]-current[wc[1]]),width_coordinates=list(wc),source='Synchronous actual robot encoder slices and neutral public FK; no object truth')


def register(registry,env,bus):
 def read_proprioception(arm_id:int=1):
  return bus.call(lambda:read_robot_proprioception(env,arm_id=arm_id))
 registry.register_callable('robot.read_proprioception',read_proprioception,summary='Synchronous actualrobot-only16encoder/FK/jaw packet including bothpassive mirrors; noobjecttruth')
