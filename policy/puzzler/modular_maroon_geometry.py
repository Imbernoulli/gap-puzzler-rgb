"""MAROON geometry regenerated from RGB, robot proprioception and public CAD.

The learned grasp is a static body-relative SE(3), not a robot/world command.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from . import model,vision_controller as c,conditional_scene
from .vision_observer import goals

def profile():
    return json.loads((model.ROOT/'profiles/maroon_achieved_grasp.json').read_text())

def _plain(v):
    return json.loads(json.dumps(v,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else float(x),allow_nan=False))

def _ik_pass(result):
    values=[result.get('position_error_m',np.nan),result.get('rotation_error_rad',np.nan),result.get('joint_limit_margin_rad',np.nan)]
    return np.isfinite(values).all()and 0<=values[0]<=.002 and 0<=values[1]<=.03 and values[2]>=.035

def plan_from_rgb(ctx,state,observed):
    p=state['parameters'];learned=profile();scene=state['planning_scene']
    if not {'board','maroon','navy','pink','red'}.issubset(observed['poses']):raise c.PlanningRefusal('MAROON plan requires currentRGB board and every unplaced object')
    if not conditional_scene.matches_measurement(scene,observed):raise c.PlanningRefusal('MAROON scene measurement is stale')
    body=np.asarray(observed['poses']['maroon']);br=np.asarray(observed['poses']['board']);r=Rotation.from_quat(body[3:]).as_matrix();rb=Rotation.from_quat(br[3:]).as_matrix()
    relative=rb.T@r;source_yaw=np.arctan2(relative[1,0],relative[0,0]);source_plane=rb@Rotation.from_euler('z',source_yaw).as_matrix()
    local=model.local_beads('maroon');sym=np.diag([-1.,-1.,1.])
    distances=np.linalg.norm((local@sym.T)[:,None]-local[None,:],axis=2)
    permutation=np.argmin(distances,axis=1)
    if np.max(distances[np.arange(len(local)),permutation])>1e-10:raise c.PlanningRefusal('MAROON source symmetry is not exact publicCAD')
    center,goal=goals('pyramid',p['pyramid_plan'])['maroon'];screens=[];selected=None
    for source_id,ss in enumerate((np.eye(3),sym)):
      base=body[:3]+r@ss@local[3]+source_plane@ss@(np.asarray(learned['grasp_tcp_center_body_m'])-local[3])
      hand_r=source_plane@ss@np.asarray(learned['grasp_rotation_body']);q=c.qdict(Rotation.from_matrix(hand_r).as_quat())
      for opening,correction in[(w,h)for h in p.get('source_clearance_corrections_m',[0.,.001,.002,.003,.004,.006])for w in p.get('source_open_width_options_m',[.055,.045,.035])]:
        grasp=base+[0.,0.,correction]
        source_clear=[conditional_scene.hand_clearance(grasp+[0,0,float(dz)],hand_r,opening,scene,plane_z=float(br[2]-.003))for dz in np.linspace(0,learned['source_hover'],13)]
        gap=min(v['clearance_m']for v in source_clear)
        if gap<.0005:
            screens.append(dict(source_symmetry=source_id,source_bead_index=int(permutation[3])if source_id else 3,source_open_width_m=float(opening),source_globalZ_correction_m=float(correction),source_min_clearance_m=gap,source_witness=min(source_clear,key=lambda x:x['clearance_m']),green=False,refusal='source_publicCAD'))
            continue
        if p.get('defer_target_until_held'):
            positions=[grasp+[0,0,learned['source_hover']],grasp,grasp+[0,0,learned['lift']]]
            checks=[ctx.tool('puzzle.plan_pose',position=pos.tolist(),rotation=q,arm_id=1)for pos in positions]
            if all(_ik_pass(v)for v in checks):
                return _plain(dict(s=observed,body=body,grasp=grasp,rotation=q,index=int(permutation[3])if source_id else 3,heading=270. if source_id==0 else 90.,lean=-30.,grasp_height_m=float(learned['achieved_tcp_height_body_m']+correction),source_open_width_m=opening,target_center=center,target_R=goal,
                    source_symmetry=source_id,target_symmetry=None,source_globalZ_correction_m=correction,source_support_plane=source_plane,learned_body_relative_grasp=learned,source_native_IK=checks,geometric_candidate_screens=screens,
                    target_feasibility_status='DEFERRED: new actual heldRGB/FK required before any target move',causal_dependency=dict(current_RGB_body=body.tolist(),current_RGB_board=br.tolist(),timestamp=observed['timestamp'],frame_sha256=observed['frame_sha256'],source_open_width_m=opening,source_globalZ_correction_m=correction,source_support_normal='current RGB board normal + source imageyaw, raw bodypose not modified')))
            screens.append(dict(source_symmetry=source_id,green=False,refusal='source_IK',native_IK_checks=checks));continue
        for target_id,ts in enumerate((np.eye(3),sym)):
            target_r=goal@ts;desired=rb@np.asarray(learned['release_rotation_bias_board'])@target_r;delta=desired@source_plane.T
            zero=br[:3]+rb@(center+np.asarray(learned['release_translation_bias_board_m']))+delta@(grasp-body[:3]);qr=c.qdict(Rotation.from_matrix(delta@hand_r).as_quat())
            row=dict(source_symmetry=source_id,source_bead_index=int(permutation[3])if source_id else 3,target_symmetry=target_id,source_open_width_m=float(opening),source_globalZ_correction_m=float(correction),source_min_clearance_m=gap)
            try:height=c.clear_release_height('maroon',observed['poses'],zero,qr,{**p,'release_height':0.,'max_release_height':0.,'close_width':0.,'release_width':learned['release_open_width']},planning_scene=scene)
            except c.PlanningRefusal as error:
                screens.append({**row,'green':False,'refusal':'target_publicCAD','error':str(error)});continue
            retreat=zero+desired[:,2]*learned['retreat']+[0,0,.010]
            positions=[grasp+[0,0,learned['source_hover']],grasp,grasp+[0,0,learned['lift']],zero+[0,0,learned['transport']],zero+[0,0,height],retreat]
            rotations=[q,q,q,qr,qr,qr]
            checks=[ctx.tool('puzzle.plan_pose',position=pos.tolist(),rotation=rot,arm_id=1)for pos,rot in zip(positions,rotations)]
            row.update(green=all(_ik_pass(v)for v in checks),native_IK_checks=checks,release_height_m=height)
            screens.append(row)
            if row['green']:
                dependency=dict(current_RGB_body=body.tolist(),current_RGB_board=br.tolist(),timestamp=observed['timestamp'],frame_sha256=observed['frame_sha256'],profile_sha256=hashlib.sha256(json.dumps(learned,sort_keys=True).encode()).hexdigest(),source_symmetry=source_id,target_symmetry=target_id,actual_robot_IK_queries=[dict(position=pos.tolist(),rotation=rot,arm_id=1)for pos,rot in zip(positions,rotations)])
                selected=dict(s=observed,body=body,grasp=grasp,rotation=q,index=row['source_bead_index'],heading=270. if source_id==0 else 90.,lean=-30.,grasp_height_m=float(learned['achieved_tcp_height_body_m']+correction),target_center=center,target_R=target_r,
                    planned_release_height_m=height,learned_body_relative_grasp=learned,source_symmetry=source_id,target_symmetry=target_id,source_open_width_m=opening,source_globalZ_correction_m=correction,
                    geometric_candidate_screens=screens,causal_dependency=dependency)
                return _plain(selected)
    raise c.PlanningRefusal('No RGB/CAD/nativeIK achieved-grasp candidate. '+json.dumps(_plain(screens),separators=(',',':')))
