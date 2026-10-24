"""Arm-0 RED low segmented carry and guarded release.

Runs only after fresh source RGB, q16/FK, opposed-blade close and post-lift
held RGB have passed.  Every segment is replanned from current q16/FK with the
native .035-rad reserve and the public hand/table/board/held-prefix guards.
"""
import json, numpy as np
from scipy.spatial.transform import Rotation
from . import conditional_scene, visual_memory, vision_controller as c
from . import modular_red_placement as base
from .modular_controller import result, save_observation
from .vision_stages import StageRefusal

DEFAULT_ARM=1; RESERVE=.035; FLOOR=.0005

def _arm(state):
    """Use the source fan's selected arm for every carry/release action."""
    arm=int(state.get('arm',DEFAULT_ARM))
    if arm not in (0,1):
        raise StageRefusal('RED segmented carry requires a selected source arm')
    return arm

def _row(state,scene,board,belief,point,rotation,width,*,descent=False):
    rr=Rotation.from_quat(c.qarr(rotation)).as_matrix()
    centers,radii=base._target_centers(np.asarray(point),rr,belief)
    hand=base._guard(np.asarray(point),rr,float(width),scene,board,target_centers=centers,target_radii=radii)
    obs,sizes,nominal,support,names=base._named_obstacles(state,scene,board,rr)
    pair=np.linalg.norm(centers[:,None]-obs[None],axis=2)-radii[:,None]-sizes[None]-.006
    nominal_pair=np.linalg.norm(centers[:,None]-obs[None],axis=2)-radii[:,None]-nominal[None]
    return dict(position=np.asarray(point).tolist(),hand_clearance_m=float(hand['clearance_m']),
                hand_components=hand.get('components',{}),hand_witness=hand.get('witness'),
                held_min_envelope_clearance_m=float(pair.min()),
                held_min_nominal_clearance_m=float(nominal_pair.min()),
                support_pairs=int(support.sum()),names=[n for n in names if n['named_support_scope']])

def _screen(ctx,state,scene,board,belief,endpoint,rotation,width,*,descent=False):
    arm=_arm(state)
    path=ctx.tool('puzzle.plan_path',position=np.asarray(endpoint).tolist(),rotation=rotation,
                  arm_id=arm,min_joint_limit_margin_rad=RESERVE)
    if not path.get('feasible') or len(path.get('path',()))<2 or len(path.get('path',()))!=len(path.get('tcp_path',())):
        raise c.PlanningRefusal(f'arm{arm} path reserve refusal '+json.dumps((path.get('errors')or[{}])[-1],separators=(',',':')))
    start=np.asarray(path['tcp_path'][0]['position']);end=np.asarray(endpoint)
    rows=[]; final_R=Rotation.from_quat(c.qarr(rotation)).as_matrix()
    for sample in path['tcp_path']:
        point=np.asarray(sample['position']); rr=Rotation.from_quat(c.qarr(sample['rotation'])).as_matrix()
        row=_row(state,scene,board,belief,point,sample['rotation'],width,descent=descent)
        rel=point[2]-end[2]; horizontal=np.linalg.norm(point[:2]-end[:2])
        descending=end[2]<start[2]-.001 and horizontal<=.001
        terminal=descent and descending and -.00025<=rel<=.012 and horizontal<=.001 and (Rotation.from_matrix(final_R).inv()*Rotation.from_matrix(rr)).magnitude()<=.04
        allowed=np.asarray([])
        if terminal and row['support_pairs']:
            # Final support contact is tested by nominal penetration in the
            # existing named-public corridor; all other portions remain held.
            bad=row['held_min_nominal_clearance_m'] < -.001
        else:
            bad=row['held_min_envelope_clearance_m'] < FLOOR
        if row['hand_clearance_m']<FLOOR or bad:
            raise c.PlanningRefusal(f'arm{arm} CAD/held refusal '+json.dumps(row,separators=(',',':')))
        rows.append(row)
    return dict(endpoint=np.asarray(endpoint).tolist(),rotation=rotation,native_path=path,checks=rows,
                min_joint_margin_rad=float(path.get('min_joint_limit_margin_rad')))

def _exec(ctx,state,scene,board,belief,endpoint,rotation,width,*,descent=False):
    arm=_arm(state)
    s=_screen(ctx,state,scene,board,belief,endpoint,rotation,width,descent=descent); path=s['native_path']; way=[]
    for a,b in zip(np.asarray(path['path'])[:-1],np.asarray(path['path'])[1:]):
        n=max(1,int(np.ceil(np.max(np.abs(b-a))/.016)))
        way.extend(dict(positions=(a+(b-a)*u).tolist()) for u in np.linspace(0,1,n+1)[1:])
    runs=[];wr=float(width)
    # The source contract's retained-jaw bound is the heldness gate.  A fixed
    # 30 mm floor would reject every valid closed YAM grasp before the first
    # carry segment; it is unrelated to the native .035-rad IK reserve.
    min_width=float(state.get('parameters',{}).get('source_contact_grasp',{}).get('min_retained_jaw_width_m',.008))
    for i in range(0,len(way),8):
        actual=base.proprio(ctx,arm)
        if actual['width_m']<max(min_width,wr-.0025): raise StageRefusal(f'arm{arm} carry jaw closure')
        wr=actual['width_m']; moved=ctx.tool('robot.execute_trajectory',trajectory={'waypoints':way[i:i+8]},arm_id=arm,max_steps_per_waypoint=3)
        if moved.get('status')!='ok': raise StageRefusal(f'arm{arm} native trajectory failure')
        runs.append(dict(actual_proprio_before=actual,motor_result=moved))
    c.settle_motion(ctx,np.asarray(endpoint),rotation,arm,False,.0025); s['executed_segments']=runs;s['actual_after']=base.proprio(ctx,arm); return s

def _post_release_active_look(ctx,state,arm,point,hint):
    """Probe empty-hand wrist orientations at a safe height with fresh RGB.

    This is an observation experiment after physical withdraw.  Every
    orientation is independently screened by native IK with the unchanged
    .035-rad reserve, then moved and followed by a fresh vision.observe.  It
    never reads evaluator/object truth and never substitutes a remembered
    pose when the red body is occluded.
    """
    rows=[]; accepted=[]; width_floor=.045
    for heading in (0.,90.,180.,270.):
        row=dict(heading_deg=heading,arm=arm,requested_position=np.asarray(point).tolist())
        try:
            q=c.frame(ctx,heading,arm)
            ik=ctx.tool('puzzle.plan_pose',position=np.asarray(point).tolist(),rotation=q,arm_id=arm)
            row['native_IK']=ik
            if (ik.get('position_error_m',float('inf'))>.002 or ik.get('rotation_error_rad',float('inf'))>.03
                    or ik.get('joint_limit_margin_rad',-1.)<RESERVE):
                row.update(status='refused_native_IK',accepted=False);rows.append(row);continue
            before=base.proprio(ctx,arm)
            if before['width_m']<width_floor:
                row.update(status='refused_empty_hand_width',accepted=False,proprio=before);rows.append(row);continue
            c.move(ctx,np.asarray(point),q,arm,cartesian=False,accuracy=.0025)
            after=base.proprio(ctx,arm);row['proprio']=after
            if after['width_m']<width_floor:
                row.update(status='refused_empty_hand_width_after_move',accepted=False);rows.append(row);continue
            observed=c.state(ctx,hint)
            row['observation']=observed
            row['red_visible']='red' in observed.get('poses',{})
            row['board_visible']='board' in observed.get('poses',{})
            from .red_source_geometry import calibrated_source_rgb_supported
            row['accepted']=bool(row['red_visible'] and row['board_visible'] and
                (observed.get('observation_source')=='rgb_shape_geometry' or calibrated_source_rgb_supported(observed)))
            row['status']='accepted_rgb_view' if row['accepted'] else 'refused_rgb_visibility'
            rows.append(row)
            if row['accepted']:accepted.append(observed)
        except Exception as error:
            row.update(status='refused_exception',accepted=False,error=f'{type(error).__name__}: {error}');rows.append(row)
    return dict(contract='RED_post_release_active_look_v1',arm=arm,safe_height_m=float(np.asarray(point)[2]),orientations=rows,accepted_observations=accepted,accepted_count=len(accepted),scope='Empty-hand safe-height orientation fanout with fresh RGB only; evaluator/object-GT reads prohibited.')

def carry(ctx,state):
    arm=_arm(state); p=state['parameters']; belief=ctx.tool('puzzle.propagate_pyramid_held_belief',source=p['owned_held_belief_source'],arm_id=arm)
    scene=base.scene_belief(state,state['observation']); board=np.asarray(scene['board_for_planning']['pose']); cur=base.proprio(ctx,arm)
    selected,candidates=base.choose_target(ctx,state,belief,scene,cur,for_transport=True)
    state={**state,'plan':{**state['plan'],'target_R':selected['target_R'],'target_center':selected['target_center'],'red_target_selection':selected,'red_target_candidates':candidates}}
    tcp=np.asarray(cur['tcp'][:3]); qcur=c.qdict(np.asarray(cur['tcp'][3:])); width=float(cur['width_m']); path=[]
    # Current post-lift q16 can be below the carry reserve.  We record two
    # source-local vertical prepositions, then search a same-grasp shoulder-out
    # pose whose native endpoint has .035 reserve before enabling plan_path.
    pre=[]
    for label,dz in [('lift25',.025),('lift40',.040)]:
        endpoint=tcp+[0,0,dz]; ik=ctx.tool('puzzle.plan_pose',position=endpoint.tolist(),rotation=qcur,arm_id=arm)
        if ik['position_error_m']>.002 or ik['rotation_error_rad']>.03:
            pre.append(dict(label=label,endpoint=endpoint.tolist(),native_IK=ik,refusal='vertical_preposition_unreachable'))
            continue
        c.move(ctx,endpoint,qcur,arm,cartesian=False,accuracy=.0025);cur=base.proprio(ctx,arm);pre.append(dict(label=label,endpoint=endpoint.tolist(),native_IK=ik,actual=cur));tcp=np.asarray(cur['tcp'][:3]);qcur=c.qdict(np.asarray(cur['tcp'][3:]));width=float(cur['width_m'])
    sign=-1.0 if arm==1 else 1.0; shoulder=None
    for heading in (150.,165.,180.,195.,210.):
        try:
            q=c.frame(ctx,heading,arm); endpoint=tcp+[0,sign*.030,0.]; ik=ctx.tool('puzzle.plan_pose',position=endpoint.tolist(),rotation=q,arm_id=arm)
            row=dict(heading_deg=heading,endpoint=endpoint.tolist(),native_IK=ik)
            if ik['position_error_m']>.002 or ik['rotation_error_rad']>.03 or ik.get('joint_limit_margin_rad',-1.)<RESERVE: pre.append(dict(label='shoulder_out30_refused',**row));continue
            row['guard']=_row(state,scene,board,belief,endpoint,q,width)
            if row['guard']['hand_clearance_m']<FLOOR or row['guard']['held_min_envelope_clearance_m']<FLOOR: pre.append(dict(label='shoulder_out30_guard_refused',**row));continue
            c.move(ctx,endpoint,q,arm,cartesian=False,accuracy=.0025);cur=base.proprio(ctx,arm);width=float(cur['width_m']); qcur=c.qdict(np.asarray(cur['tcp'][3:])); shoulder=dict(label='shoulder_out30',requested=row,actual=cur);break
        except Exception as error: pre.append(dict(label='shoulder_out30_error',heading_deg=heading,error=str(error)))
    if shoulder is None: raise c.PlanningRefusal(f'arm{arm} shoulder-away could not reach .035 reserve: '+json.dumps(pre,separators=(',',':')))
    path.extend([dict(label='source_preposition',motion=pre),shoulder])
    # Return to the low shoulder-away layer through a strict reserved path.
    endpoint=np.asarray(cur['tcp'][:3]); endpoint[2]=tcp[2]-.015
    path.append(dict(label='retreat25',motion=_exec(ctx,state,scene,board,belief,endpoint,qcur,width)))
    cur=base.proprio(ctx,arm); tcp=np.asarray(cur['tcp'][:3]); qcur=c.qdict(np.asarray(cur['tcp'][3:])); width=float(cur['width_m'])
    target=np.asarray(selected['hover_tcp']); high=max(float(tcp[2]),float(target[2])+.035)
    for label,endpoint,rotation in [('translate_high',np.array([target[0],target[1],high]),qcur),('rotate_high',np.array([target[0],target[1],high]),selected['rotation'])]:
        path.append(dict(label=label,motion=_exec(ctx,state,scene,board,belief,endpoint,rotation,width)))
        cur=base.proprio(ctx,arm); qcur=c.qdict(np.asarray(cur['tcp'][3:])); width=float(cur['width_m'])
    rec=dict(contract='RED_arm_aware_low_segmented_carry_v3',arm=arm,source_belief=belief,selected_source_candidate=state.get('plan',{}).get('selected_contact_candidate',{}),selected_target=selected,all_public_symmetry_candidates=candidates,segments=path,actual_robot=cur,state_route='carried',reserve_rad=RESERVE)
    state['transport_record']=rec
    return result('carried',state,state['observation'],'transport',transport_record=rec)

def release(ctx,state):
    arm=_arm(state); p=state['parameters']; belief=ctx.tool('puzzle.propagate_pyramid_held_belief',source=p['owned_held_belief_source'],arm_id=arm)
    scene=base.scene_belief(state,state['observation']); board=np.asarray(scene['board_for_planning']['pose']); selected=state['plan']['red_target_selection']; br=Rotation.from_quat(board[3:]).as_matrix(); desired=br@np.asarray(state['plan']['target_R']); target=board[:3]+br@np.asarray(state['plan']['target_center']); hand_R=desired@np.asarray(belief['object_in_tcp_rotation']).T; requested=target-hand_R@np.asarray(belief['object_in_tcp_position'])+[0,0,.003]; q=c.qdict(Rotation.from_matrix(hand_R).as_quat()); cur=base.proprio(ctx,arm)
    motion=_exec(ctx,state,scene,board,belief,requested,q,float(cur['width_m']),descent=True); actual=base.proprio(ctx,arm); actual_R=Rotation.from_quat(actual['tcp'][3:]).as_matrix(); centers,radii=base._target_centers(np.asarray(actual['tcp'][:3]),actual_R,belief); sweep=[base._guard(np.asarray(actual['tcp'][:3]),actual_R,float(w),scene,board,target_centers=centers,target_radii=radii) for w in np.linspace(actual['width_m'],.09,25)]
    if min(x['clearance_m'] for x in sweep)<FLOOR: raise c.PlanningRefusal(f'arm{arm} drop3 jaw sweep refused')
    opening=ctx.tool('robot.set_grip',arm_id=arm,width_m=.09,ramp_steps=18,settle_steps=15); opened=base.proprio(ctx,arm)
    opening_settle=[]
    for attempt in range(3):
        if opened['width_m']>=.0895:break
        actual_R_open=Rotation.from_quat(opened['tcp'][3:]).as_matrix()
        stationary_guard=base._guard(np.asarray(opened['tcp'][:3]),actual_R_open,opened['width_m'],scene,board,target_centers=centers,target_radii=radii,all_target=True)
        if stationary_guard['clearance_m']<FLOOR:raise c.PlanningRefusal('Release actual-jaw settling CAD refusal')
        if np.linalg.norm(np.asarray(opened['tcp'][:3])-np.asarray(actual['tcp'][:3]))>.001:raise StageRefusal('Release arm drift during jaw settle')
        waited=ctx.tool('robot.wait_steps',steps=50)
        after_wait=base.proprio(ctx,arm)
        row=dict(attempt=attempt,actual_before=opened,stationary_guard=stationary_guard,wait_result=waited,actual_after_wait=after_wait)
        if after_wait['width_m']<.0895 and attempt==1:
            row['reissue']=ctx.tool('robot.set_grip',arm_id=arm,width_m=.09,ramp_steps=0,settle_steps=25)
            after_wait=base.proprio(ctx,arm);row['actual_after_reissue']=after_wait
        opening_settle.append(row);opened=after_wait
    if opened['width_m']<.0895: raise StageRefusal(f'arm{arm} release opening incomplete after bounded actual-jaw settle')
    origin=np.asarray(opened['tcp'][:3]); opened_R=Rotation.from_quat(opened['tcp'][3:]).as_matrix(); opened_q=c.qdict(np.asarray(opened['tcp'][3:])); withdraw=origin+[0,0,.03]; withdrawal=[base._guard(point,opened_R,opened['width_m'],scene,board,target_centers=centers,target_radii=radii,all_target=True) for point in np.linspace(origin,withdraw,31)]
    if min(x['clearance_m'] for x in withdrawal)<FLOOR: raise c.PlanningRefusal(f'arm{arm} opened withdrawal guard refused')
    c.move(ctx,withdraw,opened_q,arm,cartesian=True,accuracy=.0025)
    # Endpoint home/clearance is a required viewpoint gate before minting a
    # receipt. First probe several empty-hand wrist orientations at this safe
    # withdrawn height; each pose gets a fresh RGB observation. Home both arms
    # only after the fanout completes.
    other=1-arm;other_tcp,other_q=c.hand(ctx,other)
    if other_tcp[2]<.90:
        c.move(ctx,other_tcp+[0,0,.05],other_q,other,cartesian=True,accuracy=.0025)
    other_home=ctx.tool('robot.go_home',arm_id=other)
    ctx.tool('robot.wait_steps',steps=10)
    hint={'red':[*target.tolist(),*Rotation.from_matrix(desired).as_quat().tolist()]}
    look=dict(contract='RED_post_release_home_observation_v1',orientations=[],accepted_observations=[])
    if p.get('post_release_wrist_fanout',False):look=_post_release_active_look(ctx,state,arm,withdraw,hint)
    home=ctx.tool('robot.go_home',arm_id=arm)
    ctx.tool('robot.wait_steps',steps=10)
    accepted=look.get('accepted_observations',[])
    first=accepted[0] if accepted else c.state(ctx,hint)
    ctx.tool('robot.wait_steps',steps=50)
    second=c.state(ctx,{'red':first['poses']['red']} if 'red' in first.get('poses',{}) else hint)
    receipt=None;refusal=None;memory=state['memory']
    try:
        receipt=visual_memory.release_receipt('red',first,second,np.asarray(base.proprio(ctx,arm)['tcp'][:3]),p);memory=visual_memory.mint_placement(visual_memory.observe(memory,second,p),receipt,second,p)
    except visual_memory.MemoryRefusal as err: refusal=str(err)
    record=dict(contract='RED_arm_aware_release_receipt_v2',arm=arm,owned_source=p['owned_held_belief_source'],selected_source_candidate=state.get('plan',{}).get('selected_contact_candidate',{}),selected_public_symmetry=selected['public_symmetry'],requested_release_tcp=requested.tolist(),motion=motion,actual_preopen=actual,opening_result=opening,opening_settle=opening_settle,actual_postopen=opened,withdrawal=withdrawal,home_result=home,other_home_result=other_home,post_release_active_look=look,endpoint_viewpoint=dict(active_arm=arm,active_home=home,other_arm=other,other_home=other_home,first_rgb_timestamp=first.get('timestamp'),second_rgb_timestamp=second.get('timestamp')),new_visual_release_receipt=receipt,receipt_refusal=refusal,release_rgb_pair=[first,second],target_3d_status='measured_in_new_release_pair' if 'red' in first.get('poses',{}) and 'red' in second.get('poses',{}) else 'UNKNOWN')
    earned=list(state['executed_placements'])
    if receipt and 'red' not in earned:earned.append('red')
    state={**state,'memory':memory,'release_execution':record,'execution_tokens':{**state['execution_tokens'],'red':record},'executed_placements':earned};state=save_observation(save_observation(state,first),second);ctx.tool('robot.wait_steps',steps=45)
    return result('placed' if receipt else 'uncertain',state,second,'release',release_execution=record)
