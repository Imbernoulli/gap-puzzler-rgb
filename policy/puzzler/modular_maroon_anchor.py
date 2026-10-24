"""Board-relative public-CAD support skill, no absolute teacher commands."""
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
from . import vision_controller as c,conditional_scene
from .gripper_geometry import clearance,scene_clearance

PUBLIC_PROFILE=dict(arm_id=0,offset_board_m=[-.078,.145,.020],hover_m=.090,
    close_heading_in_board_deg=-90.,width_m=.008,
    intentional_contact='support board body only',min_clearance_m=.0005,
    brace_vertical_compliance_m=.020)

def _plain(value):
    return json.loads(json.dumps(value,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else float(x),allow_nan=False))

def plan_support_anchor(ctx,state,observed):
    """Pure native IK and public gripper-CAD query on the current RGB board."""
    p=state['parameters'];profile={**PUBLIC_PROFILE,**p.get('support_anchor',{})}
    if 'board'not in observed.get('poses',{}):raise c.PlanningRefusal('Support anchor requires actual RGB board')
    board=np.asarray(observed['poses']['board'],float);rb=Rotation.from_quat(board[3:]).as_matrix()
    arm=int(profile['arm_id']);heading=float(np.degrees(np.arctan2(rb[1,0],rb[0,0]))+profile['close_heading_in_board_deg'])
    target=board[:3]+rb@np.asarray(profile['offset_board_m']);hover=target+rb[:,2]*profile['hover_m']
    q=None;orientation_screens=[]
    for twist in profile.get('heading_offsets_deg',[0.,-2.,2.,-4.,4.,180.]):
        candidate=c.frame(ctx,heading+twist,arm,-rb[:,2])
        candidate_checks=[ctx.tool('puzzle.plan_pose',position=position.tolist(),rotation=candidate,arm_id=arm)for position in(hover,target)]
        good=all(np.isfinite([x.get('position_error_m',np.nan),x.get('rotation_error_rad',np.nan),x.get('joint_limit_margin_rad',np.nan)]).all()
            and x['position_error_m']<=.002 and x['rotation_error_rad']<=.03 and x['joint_limit_margin_rad']>=.035 for x in candidate_checks)
        orientation_screens.append(dict(board_relative_heading_deg=profile['close_heading_in_board_deg']+twist,checks=candidate_checks,green=bool(good)))
        if good:q=candidate;checks=candidate_checks;profile['selected_heading_in_board_deg']=profile['close_heading_in_board_deg']+twist;break
    if q is None:raise c.PlanningRefusal('All RGB board-relative support wrist branches refused')
    rot=Rotation.from_quat(c.qarr(q)).as_matrix()
    scene=state.get('planning_scene')
    if scene is None:raise c.PlanningRefusal('Support anchor needs attributed public/current scene occupancy')
    centers,radii=conditional_scene.spheres(scene)
    samples=[]
    # Board contact is the purpose of this primitive. It does not waive table,
    # non-target piece or fixed-base clearance, and is never called collisionfree.
    for u in np.linspace(0,1,13):
        point=(1-u)*hover+u*target
        pieces=clearance(point,rot,profile['width_m'],centers,radius=radii)
        fixed=scene_clearance(point,rot,profile['width_m'],[],table_z=float(board[2]-.003),plane_z=float(board[2]-.003),board=None)
        gap=min(pieces['clearance_m'],fixed['clearance_m'])
        samples.append(dict(fraction=float(u),non_board_clearance_m=gap))
    if min(x['non_board_clearance_m']for x in samples)<profile['min_clearance_m']:
        raise c.PlanningRefusal('Board support hand sweep violates non-board/table CAD clearance')
    dependencies=dict(board_current_RGB_pose=board.tolist(),timestamp=observed['timestamp'],frame_sha256=observed['frame_sha256'],
        public_board_relative_profile=profile,actual_native_IK_results=checks,orientation_screens=orientation_screens,
        non_board_occupancy_sources={name:item['source']for name,item in scene['occupancy'].items()})
    return _plain(dict(contract='RGB_board_relative_support_anchor_v1',arm_id=arm,profile=profile,
        grasp=target,hover=hover,rotation=q,swept_non_board_CAD=samples,input_dependencies=dependencies,
        input_dependencies_sha256=hashlib.sha256(json.dumps(_plain(dependencies),sort_keys=True,separators=(',',':')).encode()).hexdigest()))

def prepare_support_anchor(ctx,state,observed):
    """Replan from the image, then use measured native robot feedback."""
    plan=plan_support_anchor(ctx,state,observed);arm=plan['arm_id'];q=plan['rotation']
    c.move(ctx,plan['hover'],q,arm)
    result=ctx.tool('robot.set_grip',arm_id=arm,width_m=plan['profile']['width_m'],ramp_steps=18,settle_steps=15)
    if not isinstance(result,dict)or result.get('clamped')is not False:raise c.PlanningRefusal('Support jaw command refused')
    c.move(ctx,plan['grasp'],q,arm,contact='brace')
    actual,rotation=c.hand(ctx,arm)
    return _plain(dict(**plan,executed=True,grip_command_result=result,actual_TCP=actual,actual_rotation=rotation,
        board_3D_status='Prior current RGB; support contact alone is not a new board pose measurement',
        scope='Board-relative geometry and actual robot feedback. Native contact/force caps unchanged; no object GT or absolute teacher command.'))


def release_support_anchor(ctx,state):
    from .modular_cyan_anchor import release_support_anchor as release
    return release(ctx,state)
