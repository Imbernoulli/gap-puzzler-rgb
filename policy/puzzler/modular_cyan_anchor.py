"""Stable shared-module interface for the current-RGB board-relative brace."""
from .cyan_geometry import prepare_support_anchor
from . import vision_controller as c
from .vision_stages import StageRefusal


def release_support_anchor(ctx,state):
    """Withdraw the narrow brace before opening beside the board edge."""
    evidence=state['support_anchor_evidence'];arm=evidence.get('arm',evidence.get('arm_id'))
    actual,q=c.hand(ctx,arm);target=actual+[0,0,.060]
    movement=c.move(ctx,target,q,arm,cartesian=True,accuracy=.0025)
    after,after_q=c.hand(ctx,arm)
    if after[2]-actual[2]<.0575:raise StageRefusal('Support withdrawal did not achieve declared vertical clearance')
    opened=ctx.tool('robot.set_grip',arm_id=arm,width_m=.075,ramp_steps=18,settle_steps=15)
    if not isinstance(opened,dict)or opened.get('clamped')is not False:raise StageRefusal('Support opening refused')
    home=ctx.tool('robot.go_home',arm_id=arm)
    return dict(contract='actual_support_withdraw_then_open_v1',arm=arm,initial_actual_tcp=actual.tolist(),
        requested_tcp=target.tolist(),actual_withdrawn_tcp=after.tolist(),actual_rotation=after_q,
        movement=movement,open_result=opened,home_result=home,
        scope='Measured vertical retreat before wide opening; no new board pose or whole-arm collision guarantee.')

__all__ = ['prepare_support_anchor','release_support_anchor']
