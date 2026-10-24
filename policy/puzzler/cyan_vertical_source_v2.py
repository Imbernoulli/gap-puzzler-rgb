"""Optional open-jaw source feedback from fresh RGB, encoders and public CAD.

Call after the original source approach and before set_grip(close). All
correction paths are solved anew and screened with the original source guard.
This module has no evaluator, object-state, contact, or attachment access.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from . import vision_controller as c, cyan_geometry as g, model, visual_memory
from .cyan_inverse_tip import source_contact_clearance


def refine_open_endpoint(ctx, state):
    p=state['parameters'];spec=p['inverse_grasp_profile'];arm=state['arm']
    if arm!=0 or abs(float(p['source_open_width'])-.095)>1.e-8:
        raise c.PlanningRefusal('Vertical source feedback requires left/open95 profile')
    records=[];command_offset=np.zeros(3);previous=float('inf')
    for attempt in range(3):
        fresh=c.state(ctx)
        if ('cyan'not in fresh.get('poses',{}) or fresh.get('observation_source')!='rgb_shape_geometry'
                or fresh.get('is_current_rgb')is False or not fresh.get('frame_sha256')):
            raise c.PlanningRefusal('Open source correction requires accepted current Cyan RGB')
        scene=g.planning_scene(fresh,state)
        body=np.asarray(fresh['poses']['cyan']);BR=Rotation.from_quat(body[3:]).as_matrix()
        target=body[:3]+BR@np.asarray(spec['hand_in_body_position'])
        R=BR@np.asarray(spec['hand_in_body_rotation']);q=c.qdict(Rotation.from_matrix(R).as_quat())
        actual,aq=c.hand(ctx,arm);AR=Rotation.from_quat(c.qarr(aq)).as_matrix()
        bead=body[:3]+BR@model.local_beads('cyan')[0];local=AR.T@(bead-actual)
        delta=target-actual;axial=float(abs(delta@R[:,2]));error=float(np.linalg.norm(delta))
        rotation_error=float((Rotation.from_matrix(R).inv()*Rotation.from_matrix(AR)).magnitude())
        guard=source_contact_clearance(actual,AR,.095,scene,p)
        row=dict(timestamp=fresh['timestamp'],frame_sha256=deepcopy(fresh['frame_sha256']),
            actual_bead0_in_tcp_m=local.tolist(),axial_residual_m=axial,position_residual_m=error,
            rotation_residual_rad=rotation_error,actual_open_source_guard=guard)
        records.append(row)
        if (axial<=.0007 and error<=.0015 and rotation_error<.03 and local[2]<=.0145
                and abs(local[0])<=.003 and abs(local[1])<=.002 and guard['passed']):
            # The new RGB body and actual encoder FK, rather than the nominal
            # profile certificate, bind every width of this actual closure.
            closure=[]
            for width in np.linspace(.095,.020,31):
                check=source_contact_clearance(actual,AR,float(width),scene,p)
                closure.append(dict(width_m=float(width),guard=check))
            row['actual_RGB_FK_loaded_closure_sweep']=closure
            if not all(item['guard']['passed']for item in closure):
                raise c.PlanningRefusal('Actual RGB/FK95-to20mm source closure CAD refused')
            windows=[]
            for window in range(2):
                now=ctx.tool('robot.execution_journal_cursor').get('timestamp')
                stamp=fresh.get('timestamp');configured_limit=float(p.get('historical_max_age_s',30.))
                if not np.isfinite(configured_limit) or configured_limit<=0.:
                    raise c.PlanningRefusal('Source endpoint scene requires an authentic finite bounded age scope')
                limit=min(30.,configured_limit)
                if (not isinstance(now,(int,float)) or not np.isfinite(now)
                        or not isinstance(stamp,(int,float)) or not np.isfinite(stamp)
                        or not np.isfinite(limit) or not 0.<limit<=30.
                        or not 0.<=now-stamp or now-stamp+.1>limit):
                    raise c.PlanningRefusal('Source endpoint wait must fit the original actual30s RGB scene scope')
                ctx.tool('robot.wait_steps',steps=5)
                settled,sq=c.hand(ctx,arm)
                drift=float(np.linalg.norm(settled-actual))
                angular=float((Rotation.from_quat(c.qarr(aq)).inv()*Rotation.from_quat(c.qarr(sq))).magnitude())
                windows.append(dict(window=window+1,position_drift_m=drift,
                    angular_drift_rad=angular,actual_RGB_timestamp=fresh['timestamp']))
                if drift<.0004 and angular<.015:
                    break
                if window==1:
                    raise c.PlanningRefusal('Vertical source endpoint did not remain encoder-stable in two bounded windows')
                # A settling retry is read-only: it cannot send a correction
                # path or retain the earlier scene/body under a new timestamp.
                fresh=c.state(ctx)
                if ('cyan'not in fresh.get('poses',{}) or fresh.get('observation_source')!='rgb_shape_geometry'
                        or fresh.get('is_current_rgb')is False or not fresh.get('frame_sha256')):
                    raise c.PlanningRefusal('Source settling retry requires accepted actual Cyan RGB')
                scene=g.planning_scene(fresh,state)
                body=np.asarray(fresh['poses']['cyan']);BR=Rotation.from_quat(body[3:]).as_matrix()
                target=body[:3]+BR@np.asarray(spec['hand_in_body_position'])
                R=BR@np.asarray(spec['hand_in_body_rotation']);q=c.qdict(Rotation.from_matrix(R).as_quat())
                actual,aq=c.hand(ctx,arm);AR=Rotation.from_quat(c.qarr(aq)).as_matrix()
                bead=body[:3]+BR@model.local_beads('cyan')[0];local=AR.T@(bead-actual)
                delta=target-actual;axial=float(abs(delta@R[:,2]));error=float(np.linalg.norm(delta))
                rotation_error=float((Rotation.from_matrix(R).inv()*Rotation.from_matrix(AR)).magnitude())
                guard=source_contact_clearance(actual,AR,.095,scene,p)
                if not (axial<=.0007 and error<=.0015 and rotation_error<.03 and local[2]<=.0145
                        and abs(local[0])<=.003 and abs(local[1])<=.002 and guard['passed']):
                    raise c.PlanningRefusal('Fresh source settling endpoint no longer meets the original RGB/FK gates')
                retry_closure=[source_contact_clearance(actual,AR,float(width),scene,p)
                               for width in np.linspace(.095,.020,31)]
                if not all(item['passed']for item in retry_closure):
                    raise c.PlanningRefusal('Fresh source settling RGB/FK95-to20mm closure CAD refused')
                records.append(dict(stage='readonly_endpoint_settling_retry',timestamp=fresh['timestamp'],
                    frame_sha256=deepcopy(fresh['frame_sha256']),actual_bead0_in_tcp_m=local.tolist(),
                    axial_residual_m=axial,position_residual_m=error,rotation_residual_rad=rotation_error,
                    actual_open_source_guard=guard,actual_RGB_FK_loaded_closure_sweep=retry_closure))
            row['bounded_encoder_stability_windows']=windows
            # Settling is evidence of bounded motion, not permission to reuse
            # the earlier pose. Recheck the final FK before the caller closes.
            SR=Rotation.from_quat(c.qarr(sq)).as_matrix()
            settled_closure=[source_contact_clearance(settled,SR,float(width),scene,p)
                             for width in np.linspace(.095,.020,31)]
            row['settled_FK_loaded_closure_sweep']=settled_closure
            if not all(item['passed']for item in settled_closure):
                raise c.PlanningRefusal('Settled FK95-to20mm source closure CAD refused')
            plan={**state['plan'],'body':body.tolist(),'grasp':target.tolist(),'rotation':q,
                  's':g.rgb_record(fresh),'vertical_source_feedback':records}
            updated={**state,'plan':plan,'planning_scene':scene,'observation':g.rgb_record(fresh),
                'memory':visual_memory.observe(state['memory'],fresh,p)}
            return updated
        if attempt==2 or error>.004 or rotation_error>.03 or error>=previous*.85:
            raise c.PlanningRefusal('Bounded vertical source encoder/RGB correction exhausted')
        previous=error;command_offset+=delta
        if np.linalg.norm(command_offset)>.003:
            raise c.PlanningRefusal('Vertical source correction exceeds3mm command bound')
        command=target+command_offset
        path=ctx.tool('puzzle.plan_path',position=command.tolist(),rotation=q,arm_id=arm,min_joint_limit_margin_rad=.035)
        if not path.get('feasible')or len(path.get('path',()))!=len(path.get('tcp_path',())):
            raise c.PlanningRefusal('Vertical source feedback native IK refused')
        checks=[source_contact_clearance(np.asarray(s['position']),Rotation.from_quat(c.qarr(s['rotation'])).as_matrix(),.095,scene,p)for s in path['tcp_path']]
        if not checks or not all(x['passed']for x in checks):
            raise c.PlanningRefusal('Vertical source feedback public CAD path refused')
        waypoints=[]
        for a,b in zip(np.asarray(path['path'])[:-1],np.asarray(path['path'])[1:]):
            count=max(1,int(np.ceil(np.max(np.abs(b-a))/.012)))
            waypoints.extend(dict(positions=(a+(b-a)*u).tolist())for u in np.linspace(0.,1.,count+1)[1:])
        row['screened_correction_offset_m']=command_offset.tolist();row['CAD_samples']=checks
        moved=ctx.tool('robot.execute_trajectory',trajectory=dict(waypoints=waypoints),arm_id=arm,max_steps_per_waypoint=2)
        if moved.get('status')!='ok':raise c.PlanningRefusal('Vertical source feedback motor refused')
        ctx.tool('robot.wait_steps',steps=10)
    raise c.PlanningRefusal('Vertical source feedback did not earn an endpoint')
