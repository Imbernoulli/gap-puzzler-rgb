"""Transfer a qualified vertical Cyan source with its stronger jaw contract.

The measured vertical source has a different attachment from the original tip
pinch. Existing runtime planning binds that actual new source; no old path or
certificate is imported. This adapter adds the source-specific >=20mm actual
loaded-jaw gate to every donor propagation before opening.
"""
import numpy as np
from copy import deepcopy

from . import cyan_bimanual_near_stream_adapter as transfer
from . import vision_controller as c,cyan_geometry as g,model
from .release_admission import check_bounded_release
from .modular_controller import save_observation


class _VerticalSourceContext:
    def __init__(self,ctx,state):
        self.ctx=ctx
        self.donor=int(state.get('transfer',{}).get('sender',state['arm']))
        self.minimum=float(state['parameters'].get('source_minimum_loaded_jaw_width_m',.020))
        if not np.isfinite(self.minimum) or self.minimum<.020:
            raise c.PlanningRefusal('Vertical source transfer cannot weaken the actual20mm loaded-jaw contract')

    def __getattr__(self,name):
        return getattr(self.ctx,name)

    def tool(self,name,**kwargs):
        result=self.ctx.tool(name,**kwargs)
        if name in ('puzzle.propagate_pyramid_held_belief','puzzle.propagate_conditional_attachment') and int(kwargs.get('arm_id',0))==self.donor:
            enc=np.asarray(result['current_robot_encoder_values'],dtype=float)
            off=0 if self.donor==0 else 8
            if enc.shape!=(16,) or not np.isfinite(enc).all():
                raise c.PlanningRefusal('Vertical source propagation needs actual finite q16')
            width=float(enc[off+6]-enc[off+7])
            if width<self.minimum:
                raise c.PlanningRefusal('Actual vertical donor jaw violates its20mm source-contact certificate')
        return result


def approach_receiver(ctx,state):
    state=deepcopy(state)
    scoped=_VerticalSourceContext(ctx,state)
    previous=transfer._belief(scoped,state)
    observed=state.get('observation',{})
    if ('cyan' not in observed.get('poses',{})
            or observed.get('timestamp',-1)<previous['current_timestamp']-.5):
        observed=c.state(scoped,{'cyan':previous['predicted_body_pose']})
    if 'cyan' not in observed.get('poses',{}):
        raise c.PlanningRefusal('New vertical transfer needs the actual postqualification RGB source before planning')
    admission=check_bounded_release('cyan',previous,observed,state['parameters'])
    state['parameters']['owned_held_belief_source']=g._source(observed,'cyan')
    state['parameters'].pop('conditional_attachment_id',None)
    state=save_observation(state,observed)
    state['held_status']='CURRENT_RGB_VERIFIED'
    current=transfer._belief(scoped,state)
    old_indices=state['parameters']['source_contact_bead_indices']
    if len(old_indices)!=1:
        raise c.PlanningRefusal('Qualified vertical source must retain its single measured source contact region')
    beads=model.local_beads('cyan')
    reference=(np.asarray(previous['object_in_tcp_position'])
               +np.asarray(previous['object_in_tcp_rotation'])@beads[int(old_indices[0])])
    measured=beads@np.asarray(current['object_in_tcp_rotation']).T+np.asarray(current['object_in_tcp_position'])
    distances=np.linalg.norm(measured-reference,axis=1)
    selected=int(np.argmin(distances))
    if distances[selected]>.006:
        raise c.PlanningRefusal('Postqualification RGB changed the named vertical contact region beyond original6mm')
    state['parameters']['source_contact_bead_indices']=[selected]
    state['vertical_transfer_source_refresh']=dict(
        actual_postqualification_source=g._source(observed,'cyan'),admission=admission,
        previous_source_timestamp=previous['source'].get('timestamp'),
        measured_frame_selected_contact_bead=selected,
        contact_region_frame_mapping_displacement_m=float(distances[selected]),
        current_target_measured=True,source_validity_not_extended=True)
    return transfer.approach_receiver(scoped,state)


def close_receiver(ctx,state):
    return transfer.close_receiver(_VerticalSourceContext(ctx,state),state)


def release_donor(ctx,state):
    return transfer.release_donor(_VerticalSourceContext(ctx,state),state)


def verify_receiver(ctx,state):
    return transfer.verify_receiver(_VerticalSourceContext(ctx,state),state)
