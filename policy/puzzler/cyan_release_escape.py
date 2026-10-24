"""Compute a fully disengaged jaw gap and an outward release pose from CAD."""
import numpy as np
from scipy.spatial.transform import Rotation
from . import model,conditional_scene,vision_controller as c
from .finite_release_scene import finite_clearance
from .gripper_geometry import clearance


def _path(ctx,scene,arm,point,rotation,width,*,target_cloud=None,target_radii=None,min_target=-np.inf):
    plan=ctx.tool('puzzle.plan_path',position=np.asarray(point).tolist(),rotation=rotation,arm_id=arm,min_joint_limit_margin_rad=.035)
    if not plan.get('feasible') or len(plan.get('path',[]))!=len(plan.get('tcp_path',[])):raise c.PlanningRefusal('Release escape Cartesian IK path refused')
    gaps=[]
    for sample in plan['tcp_path']:
        p=np.asarray(sample['position']);rr=Rotation.from_quat([sample['rotation'][k] for k in 'xyzw']).as_matrix()
        other=finite_clearance(p,rr,width,scene,exclude=('cyan',))
        gap=other['clearance_m']
        if gap<.0005:raise c.PlanningRefusal('Release escape path intersects prefix/finite board '+str(other))
        if target_cloud is not None:
            target=clearance(p,rr,width,target_cloud,radius=target_radii)
            if target['clearance_m']<min_target:raise c.PlanningRefusal('Release escape target clearance refused '+str(target))
        gaps.append(gap)
    return plan,min(gaps)


def _execute(ctx,plan,point,rotation,arm):
    waypoints=[]
    for a,b in zip(np.asarray(plan['path'])[:-1],np.asarray(plan['path'])[1:]):
        n=max(1,int(np.ceil(np.max(np.abs(b-a))/.018)))
        waypoints.extend(dict(positions=(a+(b-a)*u).tolist()) for u in np.linspace(0,1,n+1)[1:])
    moved=ctx.tool('robot.execute_trajectory',trajectory=dict(waypoints=waypoints),arm_id=arm,max_steps_per_waypoint=2)
    if moved.get('status')!='ok':raise RuntimeError('Release escape native motor execution failed')
    c.settle_motion(ctx,point,rotation,arm,False,.0025)
    return moved


def choose(ctx,scene,arm,point,rotation,actual_width,body_center,body_rotation):
    """Body follows the closed hand during the outward move, then detaches."""
    point=np.asarray(point);rr=Rotation.from_quat(c.qarr(rotation)).as_matrix();body_center=np.asarray(body_center);body_rotation=np.asarray(body_rotation)
    local,radii=conditional_scene.local_sphere_cover('cyan');cloud=local@body_rotation.T+body_center
    direction=-rr[:,2]
    if direction[2]<.05:raise c.PlanningRefusal('Public grasp approach does not permit upward outward release')
    rows=[]
    for length in [.0,.004,.006,.008,.010,.012]:
        delta=direction*length;position=point+delta;shifted=cloud+delta
        for width in [.040,.042,.043,.044,.045,.048,.050]:
            samples=[finite_clearance(position,rr,float(w),scene,exclude=('cyan',)) for w in np.linspace(actual_width,width,10)]
            other=min(samples,key=lambda r:r['clearance_m']);target=clearance(position,rr,width,shifted,radius=radii)
            row=dict(outward_distance_m=length,body_shift_m=delta.tolist(),position=position.tolist(),open_width_m=width,minimum_other_clearance_m=other['clearance_m'],fully_open_target_clearance_m=target['clearance_m'],body_follows_closed_hand=True)
            rows.append(row)
            if min(other['clearance_m'],target['clearance_m'])<.0005:continue
            try:
                path,pathgap=_path(ctx,scene,arm,position,rotation,actual_width)
                # Entire empty-hand retreat is checked at the selected full gap
                # against both the release geometry and the desired seat.
                retreat=position+direction*.035
                end=finite_clearance(retreat,rr,width,scene,exclude=('cyan',))
                if end['clearance_m']<.0005:continue
                for u in np.linspace(0,1,15):
                    pp=position+(retreat-position)*u
                    gap=finite_clearance(pp,rr,width,scene,exclude=('cyan',))['clearance_m']
                    tg=clearance(pp,rr,width,shifted,radius=radii)['clearance_m']
                    if min(gap,tg)<.0005:raise c.PlanningRefusal('Outward fully open retreat CAD refused')
            except c.PlanningRefusal as error:
                row['path_refusal']=str(error);continue
            row.update(selected=True,preopen_path_minimum_clearance_m=pathgap,withdraw_position=retreat.tolist(),release_geometry_scope='Current RGB-held transform aligned to public goal; body translated with closed hand; each opening width screened against prefix/finite board and fully open hand separated from target sphere+connector CAD.')
            motor=_execute(ctx,path,position,rotation,arm)
            return position,width,retreat,dict(candidates=rows,selected=row,preopen_native_motor=motor)
    raise c.PlanningRefusal('No outward fully disengaged release candidate passed finite CAD/native IK: '+str(rows))


def withdraw(ctx,scene,arm,position,rotation,width,body_center,body_rotation):
    local,radii=conditional_scene.local_sphere_cover('cyan');cloud=local@np.asarray(body_rotation).T+np.asarray(body_center)
    plan,gap=_path(ctx,scene,arm,position,rotation,width,target_cloud=cloud,target_radii=radii,min_target=.0005)
    motor=_execute(ctx,plan,position,rotation,arm)
    # The outward path has cleared the public part. Existing measured-FK lift
    # now changes to the safe home orientation away from the assembly.
    c.withdraw(ctx,arm)
    return dict(path_minimum_other_clearance_m=gap,motor=motor)
