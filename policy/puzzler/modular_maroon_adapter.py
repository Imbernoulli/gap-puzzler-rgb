"""Common ModularState adapter for the actual RGB/FK/CAD MAROON skill.

No recovery runner, tape, test initializer or teacher/world commands are used
here. The host must register robot.read_proprioception using the exported
actual-q16 function. Physical placement is independently checkpointed by host.
"""
from copy import deepcopy
import numpy as np
from . import vision_controller as c
from . import visual_memory
from .vision_stages import StageRefusal,_pause,_observation,_plain
from .modular_maroon_resume import resume_release

def segment_conditional_release(ctx,state):
 if state['piece']!='maroon':raise StageRefusal('This learned geometry profile is MAROON only')
 p=state['parameters'];source=p.get('owned_held_belief_source')
 if not source or source.get('observation_source')!='rgb_shape_geometry':raise StageRefusal('MAROON requires an accepted actual postliftRGB/q16 source, not an invented held pose')
 conditional=list(dict.fromkeys([*state.get('initialized_prefix',[]),*state['executed_placements'],*state.get('released_placements',[])]))
 local={**deepcopy(state),'parameters':{**p,'conditional_prefix':conditional,'conditional_prefix_margin_m':.004,'_common_belief_ledger':deepcopy(state['belief_ledger']),'_common_given_prefix':list(state.get('initialized_prefix',[])),'_common_executed':list(state['executed_placements']),'_common_released':list(state.get('released_placements',[]))},'held_belief_source':deepcopy(source),'latest_rgb':deepcopy(state['observation']),'historical_rgb_records':deepcopy(state['rgb_history'])}
 out=resume_release(ctx,local)
 # Acquire a genuinely separated RGB pair after both hands have withdrawn.
 # Public CAD is an optimizer seed only; every result is fit to new pixels.
 from .vision_observer import goals
 from scipy.spatial.transform import Rotation
 observed=out['state']['latest_rgb'];board=np.asarray(observed['poses']['board'],dtype=float)
 center,rr=goals(p.get('mode','pyramid'),p['pyramid_plan'])['maroon'];rb=Rotation.from_quat(board[3:]).as_matrix()
 hint={'maroon':[*list(board[:3]+rb@center),*Rotation.from_matrix(rb@rr).as_quat().tolist()], '__observer_mode__':'plain_rgb_cad'}
 first=c.state(ctx,hint);_pause(ctx,60)
 next_hint=dict(hint)
 if 'maroon' in first.get('poses',{}):next_hint['maroon']=list(first['poses']['maroon'])
 second=c.state(ctx,next_hint);record=deepcopy(out['state']['release_execution'])
 packets=[{k:deepcopy(r[k])for k in ['poses','timestamp','frame_sha256','observation_source','assessment','robot_geometry_provenance']if k in r}for r in[first,second]]
 receipt=None;receipt_refusal=None
 # The learned MAROON release module historically left receipt creation to
 # the host even when both fresh RGB packets contained the target.  Recompute
 # the ordinary visual-memory receipt here from that pair and the measured
 # release-start TCP.  Every existing timestamp, displacement, goal-error and
 # hand-gap gate remains in visual_memory.release_receipt; a refusal leaves
 # the route UNKNOWN/uncertain exactly as before.
 try:
  tcp,_=c.hand(ctx,state['arm'])
  receipt=visual_memory.release_receipt('maroon',first,second,tcp,p)
 except (visual_memory.MemoryRefusal,TypeError,ValueError) as error:
  receipt_refusal=str(error)
 record.update(release_rgb_pair=packets,new_visual_release_receipt=receipt,
               receipt_refusal=receipt_refusal,
               target_current_rgb_status=['measured_rgb'if'maroon'in r['poses']else'UNKNOWN'for r in[first,second]],
               receipt_scope='Fresh RGB release pair receipt recomputed with public CAD goals, current board, synchronous q16 and actual withdrawn FK TCP and unchanged strict gates; no teacher or GT.')
 memory=deepcopy(state['memory'])
 if receipt is not None:
  memory=visual_memory.mint_placement(visual_memory.observe(memory,second,p),receipt,second,p)
 merged={**deepcopy(state),'memory':memory,'belief_ledger':deepcopy(out['state']['parameters']['_common_belief_ledger']),'observation':_observation(second),'release_execution':record,'last_stage':'release_execution','last_outcome':'executed'}
 record['target_nonfinger_guard_retained']=False
 record['geometry_scope']='Actual proprioception feedback and sampled non-target/table/board hand CAD; target housing clearance is not certified. Independent physical placement/prefix postconditions remain unchanged.'
 evidence={**out['evidence'],'timestamp':second['timestamp'],'frame_sha256':second['frame_sha256'],'target_3d_status':record['target_current_rgb_status'][-1],'visual_release_receipt':receipt is not None,'placement_verified':receipt is not None}
 return _plain(dict(route='executed',state=merged,evidence=evidence))
