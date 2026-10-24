"""Generate and screen live native Cartesian IK paths using attributed CAD."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from . import vision_controller as c,conditional_scene


def move_held_screened(ctx,state,endpoint,rotation,belief,scene):
    name=state['piece'];arm=state['arm'];p=state['parameters'];scene=deepcopy(scene)
    board_info=scene['board_for_planning'];board=np.asarray(board_info['pose']);board_margin=board_info['uncertainty_m']
    age=belief['current_timestamp']-board_info['source_timestamp']
    if not 0<=age<=p.get('historical_max_age_s',30.):raise c.PlanningRefusal('Carry board source is stale')
    path=ctx.tool('puzzle.plan_path',position=np.asarray(endpoint).tolist(),rotation=rotation,
        arm_id=arm,min_joint_limit_margin_rad=p.get('min_ik_joint_margin_rad',.035))
    cartesian_refusal=None
    if not path.get('feasible'):
        cartesian_refusal=deepcopy(path)
        path=ctx.tool('puzzle.plan_joint_path',position=np.asarray(endpoint).tolist(),rotation=rotation,
            arm_id=arm,min_joint_limit_margin_rad=p.get('min_ik_joint_margin_rad',.035))
    if not path.get('feasible')or len(path.get('tcp_path',()))!=len(path.get('path',()))or len(path['path'])<2:
        raise c.PlanningRefusal('Live held carry Cartesian and current-joint paths refused or missing public FK samples')
    # Current image occupancy becomes a dated persistence assumption throughout
    # this bounded move; its source image and raw poses remain unchanged.
    for other,item in scene['occupancy'].items():
        if other!=name:
            item['inflation_m']=max(.004,item['inflation_m'])
            item['source']+='; dated RGB/static during bounded carry assumption'
    obstacles,sizes=conditional_scene.spheres(scene,exclude=(name,));local,radii=conditional_scene.local_sphere_cover(name)
    relative_p=np.asarray(belief['object_in_tcp_position']);relative_R=np.asarray(belief['object_in_tcp_rotation'])
    uncertainty=float(belief.get('uncertainty_m',.008 if p.get('conditional_attachment_id')else .006))
    q16=belief['current_robot_encoder_values'];offset=8*arm;actual_width=float(q16[offset+6]-q16[offset+7]);checks=[]
    for sample in path['tcp_path']:
        point=np.asarray(sample['position']);rr=Rotation.from_quat(c.qarr(sample['rotation'])).as_matrix()
        hand=conditional_scene.hand_clearance(point,rr,actual_width,scene,exclude=(name,),plane_z=float(board[2]-.003+board_margin))
        centers=local@(rr@relative_R).T+point+rr@relative_p
        pair=np.linalg.norm(centers[:,None]-obstacles[None],axis=2)-radii[:,None]-sizes[None]-uncertainty
        held=min(float(pair.min()),float((centers[:,2]-radii-uncertainty-(board[2]-.003+board_margin)).min()))
        checks.append(dict(hand_clearance_m=hand['clearance_m'],held_clearance_m=held))
    if min(min(row.values())for row in checks)<p.get('carry_clearance_margin_m',.002):
        raise c.PlanningRefusal('Actual held-transform Cartesian carry violates attributed hand/held CAD envelope')
    waypoints=[]
    for aa,bb in zip(np.asarray(path['path'])[:-1],np.asarray(path['path'])[1:]):
        count=max(1,int(np.ceil(np.max(np.abs(bb-aa))/.018)))
        waypoints.extend(dict(positions=(aa+(bb-aa)*u).tolist())for u in np.linspace(0,1,count+1)[1:])
    moved=ctx.tool('robot.execute_trajectory',trajectory=dict(waypoints=waypoints),arm_id=arm,max_steps_per_waypoint=2)
    if not isinstance(moved,dict)or moved.get('status')!='ok':raise RuntimeError('Live carry trajectory failed')
    c.settle_motion(ctx,endpoint,rotation,arm,False,.0025)
    return dict(contract='live_RGB_FK_CAD_screened_carry_v1',native_ik=path,cartesian_refusal=cartesian_refusal,actual_width_m=actual_width,
        held_uncertainty_m=uncertainty,planning_scene=scene,cad_sample_checks=checks,motor_result=moved,
        scope='Sampled public hand/held CAD and native Cartesian FK path; dated occupancy persistence assumptions. No whole-arm or between-sample collision proof.')
