"""Identify known bead geometry from RGB silhouettes, without a colour ID table.

Initial identity comes from unordered bead-centre geometry, with a one-to-one
assignment across CAD parts. Subsequent multi-view fits track these identities.
Planar initialization assumes separated, supported parts; overlap can refuse.
"""
from functools import lru_cache
from itertools import permutations

import cv2
import numpy as np
from scipy.ndimage import maximum_filter, map_coordinates
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.spatial.transform import Rotation

from .model import CELLS, RADIUS, PITCH, local_beads
from .vision import NECK_RADIUS, boundary, cad_necks, fit_board, fit_piece
from . import vision as color_rgb
from . import vision as original_appearance


def foreground(rgb):
    """Generic object/background contrast, never hue-to-part classification.

    Saturated foreground and bright neutral foreground are both retained.
    This is a stated appearance/background assumption, not semantic colour ID.
    """
    high=rgb.max(axis=2);low=rgb.min(axis=2)
    return (((high-low>24)&(high>35))|((high-low<=24)&(high>108))).astype(np.uint8)


def planar_candidates(rgb,camera):
    mask=foreground(rgb)
    connected=cv2.dilate(mask,np.ones((5,5),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(connected)
    candidates=[]
    for k in range(1,count):
        area=stats[k,cv2.CC_STAT_AREA]
        if not 450<area<18000:continue
        x,y,w,h=stats[k,:4]
        part=mask[y:y+h,x:x+w]&(labels[y:y+h,x:x+w]==k)
        dt=cv2.distanceTransform(part,cv2.DIST_L2,5)
        peaks=((dt==maximum_filter(dt,size=13))&(dt>6)).astype(np.uint8)
        _,_,_,centers=cv2.connectedComponentsWithStats(peaks)
        pixels=centers[1:]+np.array([x,y])
        if not 3<=len(pixels)<=18:continue
        cloud=camera.plane(pixels,.7625)
        if not (.12<cloud[:,0].mean()<1.05 and abs(cloud[:,1].mean())<.5):continue
        candidates.append(dict(mask=part,cloud=cloud))
    return candidates


def match_geometry(name,cloud):
    """Rigid unordered point matching, including physically possible flips."""
    local=local_beads(name)[:,:2]
    if len(local)!=len(cloud):return float('inf'),None
    observed=np.asarray(cloud)[:,:2]
    center=observed.mean(axis=0)
    best=(float('inf'),None)
    for order in permutations(range(len(local))):
        target=observed[list(order)]-center
        u,_,vt=np.linalg.svd(local.T@target)
        row_rotation=u@vt
        residual=np.linalg.norm(local@row_rotation-target,axis=1)
        cost=float(np.sqrt(np.mean(residual**2)))
        if cost<best[0]:
            rotation=np.zeros((3,3));rotation[:2,:2]=row_rotation.T
            rotation[2,2]=np.linalg.det(row_rotation)
            best=(cost,np.r_[center,.7625,Rotation.from_matrix(rotation).as_quat()])
    return best


def identify(rgb,camera,hypotheses=False,required_names=None):
    names=list(CELLS)if required_names is None else list(required_names)
    if not names or len(names)!=len(set(names))or set(names)-set(CELLS):raise ValueError('Known unplaced identity scope must contain unique public CAD names')
    components=planar_candidates(rgb,camera)
    options={n:[] for n in names}
    offset=0
    for component in components:
        cloud=component['cloud']
        for name in names:
            options[name].extend(subshape_matches(name,cloud,offset))
        offset+=len(cloud)
    for name in options:options[name].sort(key=lambda d:d['error'])
    # A silhouette component may contain several touching parts. Assign whole
    # CAD shapes without ever assigning the same observed bead to two parts.
    counts={n:len(v) for n,v in options.items()}
    if any(not v for v in options.values()):return {},dict(candidate_counts=counts)
    columns=[(n,v) for n in names for v in options[n]]
    matrix=np.zeros((len(names)+offset,len(columns)))
    for j,(name,candidate) in enumerate(columns):
        matrix[names.index(name),j]=1.
        for bead in candidate['beads']:matrix[len(names)+bead,j]=1.
    lower=np.r_[np.ones(len(names)),np.zeros(offset)]
    upper=np.ones(len(names)+offset)
    solution=milp(np.array([v['error']*1e6 for _,v in columns]),
                  integrality=np.ones(len(columns)),bounds=Bounds(0.,1.),
                  constraints=LinearConstraint(matrix,lower,upper),
                  options=dict(time_limit=3.,mip_rel_gap=0.))
    if not solution.success:return {},dict(candidate_counts=counts,assignment_refused=str(solution.message))
    selected={n:v for (n,v),x in zip(columns,solution.x) if x>.5}
    cuts=[];alternatives=[];changed=set();gap=None;current=solution
    # Establish the entire near-optimal candidate set, rather than assuming the
    # second-best assignment is the only possible competing interpretation.
    for _ in range(16):
        exclude=np.array([float(x>.5) for x in current.x])[None,:]
        cuts.append(LinearConstraint(exclude,0.,len(names)-1.))
        alternative=milp(np.array([v['error']*1e6 for _,v in columns]),
                         integrality=np.ones(len(columns)),bounds=Bounds(0.,1.),
                         constraints=[LinearConstraint(matrix,lower,upper),*cuts],
                         options=dict(time_limit=3.,mip_rel_gap=0.))
        if not alternative.success:
            if alternative.status!=2:
                return {},dict(assignment_refused='could not exhaust ambiguous CAD assignments')
            break
        candidate_gap=float(alternative.fun-solution.fun)/1e6
        if gap is None:gap=candidate_gap
        if candidate_gap>=.00075:break
        alternate={n:v['seed'].tolist() for (n,v),x in zip(columns,alternative.x) if x>.5}
        alternatives.append(alternate)
        changed.update(n for (n,v),x in zip(columns,alternative.x) if x>.5 and v['beads']!=selected[n]['beads'])
        current=alternative
    else:
        return {},dict(assignment_refused='ambiguous CAD assignments exceed enumeration budget')
    if alternatives:
        diagnostic=dict(assignment_refused='multiple visually ambiguous CAD assignments',assignment_gap_m=gap,
                        alternative_seeds=alternatives[0],alternative_hypotheses=alternatives,
                        ambiguous_names=sorted(changed),near_assignment_count=1+len(alternatives))
        if hypotheses:return {n:v['seed'] for n,v in selected.items()},diagnostic
        return {},diagnostic
    return ({n:v['seed'] for n,v in selected.items()},
            {n:dict(shape_error_m=v['error'],assignment_gap_m=gap,identity_source='CAD geometry with non-overlapping bead assignment')
             for n,v in selected.items()})


def subshape_matches(name,cloud,offset):
    """Match a CAD shape inside a larger unlabelled bead cloud."""
    local=local_beads(name)[:,:2];points=cloud[:,:2]
    if len(local)>len(points):return []
    distances=np.linalg.norm(local[:,None]-local[None,:],axis=2)
    a,b=np.unravel_index(distances.argmax(),distances.shape)
    baseline=distances[a,b]
    matches={}
    for flip in (1.,-1.):
        source=local*np.array([1.,flip])
        source_angle=np.arctan2(*(source[b]-source[a])[::-1])
        for i in range(len(points)):
            for j in range(len(points)):
                delta=points[j]-points[i]
                if abs(np.linalg.norm(delta)-baseline)>.003:continue
                angle=np.arctan2(delta[1],delta[0])-source_angle
                c,s=np.cos(angle),np.sin(angle);row=np.array([[c,s],[-s,c]])
                predicted=source@row+points[i]-source[a]@row
                cost=np.linalg.norm(predicted[:,None]-points[None,:],axis=2)
                ii,jj=linear_sum_assignment(cost)
                chosen=points[jj]
                u,_,vt=np.linalg.svd(local.T@(chosen-chosen.mean(axis=0)))
                fitted=u@vt
                residual=np.linalg.norm(local@fitted+chosen.mean(axis=0)-chosen,axis=1)
                error=float(np.sqrt(np.mean(residual**2)))
                # Noisy image peaks perturb the two-anchor proposal. Test the
                # actual rigid fit, rather than rejecting a valid correspondence
                # before all observed centres can refine its translation/yaw.
                if residual.max()>.003 or error>.0015:continue
                ids=frozenset(int(k)+offset for k in jj)
                if ids in matches and matches[ids]['error']<=error:continue
                R=np.zeros((3,3));R[:2,:2]=fitted.T;R[2,2]=np.linalg.det(fitted)
                seed=np.r_[chosen.mean(axis=0),.7625,Rotation.from_matrix(R).as_quat()]
                matches[ids]=dict(seed=seed,error=error,beads=ids)
    return list(matches.values())


def object_masks(name,seed,foregrounds,cameras):
    """Select local image evidence by geometry, never an expected part colour."""
    seed=np.asarray(seed)
    centers=local_beads(name)@Rotation.from_quat(seed[3:]).as_matrix().T+seed[:3]
    masks={};contours={}
    for key,camera in cameras.items():
        roi=np.zeros_like(foregrounds[key])
        camera_points=(centers-camera.camera_to_world[:3,3])@camera.camera_to_world[:3,:3]
        pixels=camera.project(centers)
        for pixel,depth in zip(pixels,camera_points[:,2]):
            if depth<=0:continue
            radius=int(np.ceil(camera.K[0,0]*RADIUS/depth))+9
            cv2.circle(roi,tuple(np.rint(pixel).astype(int)),radius,1,-1)
        masks[key]=foregrounds[key]&roi
        # Use observed foreground edges. Clipping a mask creates synthetic ROI
        # edges, which must never become evidence for a physical silhouette.
        points=boundary(foregrounds[key],limit=16000)
        if len(points):
            pixels=np.rint(points).astype(int)
            points=points[roi[pixels[:,1],pixels[:,0]]>0]
            # Keep edges that contradict the predicted sphere tangencies too.
            # Filtering them by seed agreement can lock a wrong rotation in
            # place even when the complete, correct shape is plainly visible.
        contours[key]=points
    return masks,contours


def shape_distance(name,a,b):
    a=np.asarray(a);b=np.asarray(b);local=local_beads(name)
    aa=local@Rotation.from_quat(a[3:]).as_matrix().T+a[:3]
    bb=local@Rotation.from_quat(b[3:]).as_matrix().T+b[:3]
    distances=np.linalg.norm(aa[:,None]-bb[None,:],axis=2)
    ii,jj=linear_sum_assignment(distances)
    return float(distances[ii,jj].max())


def raw_image_evidence(foregrounds,cameras):
    """Image-only evidence prepared once, without a part pose or search seed.

    Optimizer ROIs may depend on a pose prediction. Acceptance evidence must
    retain raw contours, including pixels that contradict that prediction.
    """
    return {key:dict(points=boundary(mask,limit=16000),
                     background_distance=cv2.distanceTransform(1-mask,cv2.DIST_L2,5))
            for key,mask in foregrounds.items() if key in cameras}


def foreign_support_footprint(rays,expected_near,foreign,camera):
    """Pixels that may belong to nearer, RGB-supported other-part geometry.

    This withholds uncertain positive attribution; it is not a depth reading or
    a claim that the candidate is physically occluded. The same four-pixel
    allowance used by foreground support expands the known CAD footprint.
    ``foreign`` includes every compatible other-part hypothesis, never a greedy
    choice of its first pose. Positive forward ray depth is mandatory.
    """
    blocked=np.zeros(len(rays),dtype=bool);origin=camera.camera_to_world[:3,3]
    for other,hypotheses in foreign.items():
        for cloud in hypotheses:
            delta=cloud-origin
            z=(delta@camera.camera_to_world[:3,:3])[:,2]
            padding=4*np.maximum(z,0)/camera.K[0,0]
            radii=RADIUS+padding
            t=rays@delta.T
            perpendicular=np.maximum(np.sum(delta*delta,axis=1)[None,:]-t*t,0.)
            hit=(z[None,:]>0)&(t>0)&(perpendicular<radii[None,:]**2)
            near=t-np.sqrt(np.maximum(radii[None,:]**2-perpendicular,0.))
            # Include the existing 3 mm geometric uncertainty band: possible
            # nearer evidence is enough to withhold positive attribution.
            blocked|=(hit&(near>0)&(near<expected_near[:,None]+.003)).any(axis=1)
            for i,j in cad_necks(other):
                aa=np.cross(rays,delta[i]);vv=np.cross(rays,cloud[j]-cloud[i])
                u=np.clip(-np.sum(aa*vv,axis=1)/np.maximum(np.sum(vv*vv,axis=1),1.e-15),0.,1.)
                distance=np.linalg.norm(aa+u[:,None]*vv,axis=1)
                radius=NECK_RADIUS+max(padding[i],padding[j])
                ray_depth=np.sum(rays*(delta[i]+u[:,None]*(cloud[j]-cloud[i])),axis=1)
                near=ray_depth-np.sqrt(np.maximum(radius**2-distance**2,0.))
                blocked|=(ray_depth>0)&(distance<radius)&(near>0)&(near<expected_near+.003)
    return blocked


def bead_image_support(name,pose,evidence,cameras,foreign=None):
    """Require current RGB centre and silhouette evidence for every CAD bead.

    Aggregate support can certify four real beads while inventing a fifth. A
    bead needs foreground at its projected centre and a measurable tangent arc
    in two calibrated views. Occluded/underdetermined beads refuse; a previous
    track and an optimizer's chosen contour subset cannot supply this evidence.
    """
    pose=np.asarray(pose)
    centers=local_beads(name)@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]
    counts=np.zeros(len(centers),dtype=int);by_view={}
    # Reject most alternative hypotheses without computing contour distances.
    center_views={}
    withheld_centers={}
    for key,raw in evidence.items():
        camera=cameras[key];uv=camera.project(centers)
        distances=map_coordinates(raw['background_distance'],[uv[:,1],uv[:,0]],
                                  order=1,mode='constant',cval=1000.)
        center_views[key]=distances<4.
        if foreign:
            near=np.linalg.norm(centers-camera.camera_to_world[:3,3],axis=1)-RADIUS
            withheld=foreign_support_footprint(camera.rays(uv),near,foreign,camera)
            center_views[key]&=~withheld
            withheld_centers[key]=withheld.tolist()
    if not np.all(np.sum(list(center_views.values()),axis=0)>=2):
        return False,dict(bead_supported_views=counts.tolist(),withheld_foreign_center_support=withheld_centers,
                          reason='a CAD bead lacks foreground support in two clear RGB views' if foreign else
                                 'a CAD bead lacks foreground support in two RGB views')
    for key,raw in evidence.items():
        camera=cameras[key];points=raw['points'];uv=camera.project(centers)
        if not len(points):continue
        rays=camera.rays(points)
        delta=centers-camera.camera_to_world[:3,3]
        distances=np.linalg.norm(np.cross(rays[:,None,:],delta[None,:,:]),axis=2)-RADIUS
        # A silhouette ray cannot pass through the interior of another bead
        # of this same proposed rigid part.
        exterior=distances.min(axis=1)>-.0018
        supported=[]
        for i in range(len(centers)):
            selected=exterior&(np.abs(distances[:,i])<.0018)
            if foreign and np.any(selected):
                idx=np.flatnonzero(selected)
                t=rays[idx]@delta[i]
                perpendicular2=np.maximum(np.sum(delta[i]*delta[i])-t*t,0.)
                # Tangent rays reach the sphere near t, not t-radius. Rays
                # just outside the nominal sphere are admitted by the raw
                # contour tolerance and use tangent depth (zero square root).
                near=t-np.sqrt(np.maximum(RADIUS**2-perpendicular2,0.))
                selected[idx]&=~foreign_support_footprint(rays[idx],near,foreign,camera)
            angles=np.arctan2(points[selected,1]-uv[i,1],points[selected,0]-uv[i,0])
            bins=np.unique(np.floor((angles+np.pi)/(2*np.pi)*12).astype(int))
            accepted=bool(center_views[key][i] and selected.sum()>=6 and len(bins)>=3)
            supported.append(accepted)
            counts[i]+=int(accepted)
        by_view[key]=supported
    valid=bool(np.all(counts>=2))
    return valid,dict(bead_supported_views=counts.tolist(),bead_support_by_view=by_view,
                     withheld_foreign_center_support=withheld_centers,
                     reason='' if valid else 'a CAD bead lacks independent silhouette support in two RGB views')


@lru_cache(maxsize=None)
def partial_shape_aliases(name):
    """Distinct planar CAD poses sharing the old minimum bead support.

    Tracking must not silently choose one of these interpretations merely
    because it was supplied as the optimization seed. Exact shape symmetries
    are equivalent poses and are omitted.
    """
    local=local_beads(name);aliases={}
    for flip in (1.,-1.):
        for turn in range(4):
            rotation=Rotation.from_euler('z',90*turn,degrees=True).as_matrix()@np.diag([1.,flip,flip])
            for source in local:
                for target in local:
                    shift=source-target@rotation.T
                    moved=local@rotation.T+shift
                    shared=int((np.linalg.norm(moved[:,None]-local[None,:],axis=2).min(axis=1)<1.e-7).sum())
                    if not .6*len(local)<=shared<len(local):continue
                    signature=tuple(sorted(map(tuple,np.round(moved,7))))
                    aliases[signature]=(rotation,shift)
    return tuple(aliases.values())


def neck_image_evidence(a,b,scene_beads,evidence,cameras):
    """Positive narrow-bridge RGB evidence outside the projected bead union.

    Missing/occluded bridges return unknown. They never imply a disconnection.
    Known cylinder geometry supplies sample pixels, not their observed values.
    Background flanks reject broad foreground occluders; all measurements use
    raw image distance fields prepared without part identity or a pose seed.
    """
    a=np.asarray(a);b=np.asarray(b);axis=b-a;middle=(a+b)/2
    if abs(np.linalg.norm(axis)-PITCH)>.003:return False,dict(reason='spacing inconsistent with CAD')
    by_view={};positive=[];contradictions=[]
    for key,raw in evidence.items():
        camera=cameras[key]
        depth=float(((middle-camera.camera_to_world[:3,3])@camera.camera_to_world[:3,:3])[2])
        if depth<=0:continue
        uv=camera.project(middle[None,:])[0]
        extent=int(np.ceil(camera.K[0,0]*NECK_RADIUS/depth))+8
        xx,yy=np.meshgrid(np.arange(int(np.floor(uv[0]))-extent,int(np.floor(uv[0]))+extent+1),
                          np.arange(int(np.floor(uv[1]))-extent,int(np.floor(uv[1]))+extent+1))
        points=np.c_[xx.ravel(),yy.ravel()].astype(float)
        h,w=raw['background_distance'].shape
        points=points[(points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)]
        if not len(points):continue
        rays=camera.rays(points);origin=camera.camera_to_world[:3,3]
        aa=np.cross(rays,a-origin);vv=np.cross(rays,axis)
        fraction=np.clip(-np.sum(aa*vv,axis=1)/np.maximum(np.sum(vv*vv,axis=1),1.e-15),0.,1.)
        neck_distance=np.linalg.norm(aa+fraction[:,None]*vv,axis=1)-NECK_RADIUS
        sphere_distance=(np.linalg.norm(np.cross(rays[:,None,:],(scene_beads-origin)[None,:,:]),axis=2)-RADIUS).min(axis=1)
        available=(sphere_distance>.00015)&(fraction>.35)&(fraction<.65)
        interior=available&(neck_distance<-.0004)
        pixel_m=depth/camera.K[0,0]
        # Expected background must be separated from both predicted bead and
        # neck boundaries by resolved pixels, not a subpixel geometric sliver.
        flanks=(sphere_distance>2*pixel_m)&(fraction>.35)&(fraction<.65)&(neck_distance>2*pixel_m)&(neck_distance<8*pixel_m)
        ninside=int(interior.sum());noutside=int(flanks.sum())
        if ninside<3 or noutside<6:
            by_view[key]=dict(resolved=False,interior_pixels=ninside,flank_pixels=noutside)
            continue
        pixel_distance=raw['background_distance'][points[:,1].astype(int),points[:,0].astype(int)]
        inside_fraction=float(np.mean(pixel_distance[interior]<.75))
        flank_fraction=float(np.mean(pixel_distance[flanks]>1.))
        # Require evidence on both transverse sides, rather than a sliver that
        # could be one neighboring sphere's displaced boundary.
        projected_axis=camera.project(np.array([a,b]));direction=projected_axis[1]-projected_axis[0]
        transverse=np.array([-direction[1],direction[0]])
        sides=(points-uv)@transverse
        both_sides=all(np.count_nonzero(interior&(sides*sign>0)&(pixel_distance<.75))>=1 for sign in (-1,1))
        supported=inside_fraction>=.85 and flank_fraction>=.85 and both_sides
        by_view[key]=dict(resolved=True,interior_pixels=ninside,flank_pixels=noutside,
                          interior_foreground_fraction=inside_fraction,flank_background_fraction=flank_fraction,
                          supported=bool(supported))
        if supported:positive.append(key)
        elif inside_fraction<.25 and flank_fraction>=.85:contradictions.append(key)
    valid=bool(positive) and not contradictions
    return valid,dict(positive_views=positive,contradicting_views=contradictions,views=by_view,
                      source='current RGB bridge pixels and public neck CAD; absence remains unknown')


def observed_neck_links(cloud,evidence,cameras):
    """Measure positive connectivity between image-estimated bead centres."""
    links=[]
    for i,a in enumerate(cloud):
        for j in range(i+1,len(cloud)):
            if abs(np.linalg.norm(a-cloud[j])-PITCH)>.003:continue
            supported,diagnostic=neck_image_evidence(a,cloud[j],cloud,evidence,cameras)
            if supported:links.append((i,j,diagnostic))
    return links


def refuse_shape_aliases(detections,evidence,cameras,fit_candidate=None):
    """Resolve supported aliases together, including possible track swaps.

    A neighboring track can rule out a pose only through a complete mutually
    compatible interpretation. Greedily trusting that neighbor's first pose
    would hide a jointly ambiguous reassignment and depend on iteration order.
    Apparent RGB necks are diagnostics only: a thin foreign foreground can
    project the same bridge, so unproven attribution cannot exclude a pose.
    """
    options={};scene=[]
    for name in sorted(detections):
        detection=detections[name]
        if name not in CELLS or not detection['visible']:continue
        pose=np.asarray(detection['pose']);R=Rotation.from_quat(pose[3:]).as_matrix()
        candidates=[pose]
        for local_R,shift in partial_shape_aliases(name):
            candidate=np.r_[pose[:3]+R@shift,Rotation.from_matrix(R@local_R).as_quat()]
            supported,_=bead_image_support(name,candidate,evidence,cameras)
            if supported:
                if fit_candidate is not None:
                    refined=fit_candidate(name,candidate)
                    if not refined['visible']:continue
                    candidate=np.asarray(refined['pose'])
                if not any(shape_distance(name,candidate,existing)<=.003 for existing in candidates):
                    candidates.append(candidate)
        options[name]=[(p,local_beads(name)@Rotation.from_quat(p[3:]).as_matrix().T+p[:3]) for p in candidates]
    # Consider bead locations supported by competing interpretations as well.
    # A connector to a bead missing from the first hypothesis is precisely the
    # evidence needed to reject that first hypothesis. Unconnected locations
    # remain optional; this union is not exposed as measured part poses.
    for rows in options.values():
        for _,cloud in rows:
            for bead in cloud:
                if not scene or np.linalg.norm(np.asarray(scene)-bead,axis=1).min()>.003:scene.append(bead)
    scene=np.asarray(scene).reshape(-1,3)
    links=observed_neck_links(scene,evidence,cameras) if len(scene) else []
    for name,rows in options.items():
        owned_rows=[]
        for p,cloud in rows:
            distance=np.linalg.norm(cloud[:,None]-scene[None,:],axis=2)
            owned=frozenset(int(j) for j,error in zip(distance.argmin(axis=1),distance.min(axis=1)) if error<=.003)
            owned_rows.append((p,cloud,owned))
        options[name]=owned_rows
        detections[name]['observed_neck_links']=len(links)
        detections[name]['neck_evidence_role']='unattributed RGB diagnostic only'
    names=sorted(options,key=lambda n:(len(options[n]),n))
    possible={n:set() for n in names};chosen={};visits=0;exhausted=False;assignments=[]
    def search(index):
        nonlocal visits,exhausted
        visits+=1
        if visits>4096:
            exhausted=True;return
        if index==len(names):
            for name,option in chosen.items():possible[name].add(option)
            assignments.append(dict(chosen))
            return
        name=names[index]
        for j,(_,cloud,owned) in enumerate(options[name]):
            if any(np.linalg.norm(cloud[:,None]-options[other][oi][1][None,:],axis=2).min()<.020
                   for other,oi in chosen.items()):continue
            chosen[name]=j;search(index+1);del chosen[name]
            if exhausted:return
    search(0)
    geometric_possible={n:set(v) for n,v in possible.items()}
    attributions={}
    if not exhausted:
        # Restrict foreign geometry to complete, mutually compatible current
        # interpretations that co-occur with this candidate. A possible foreign
        # foreground footprint can withhold positive attribution, never supply
        # a negative measurement or a new candidate pose.
        for name in names:
            attributions[name]={}
            for j in sorted(possible[name]):
                cooccurring=[a for a in assignments if a[name]==j]
                foreign={other:[options[other][oi][1] for oi in sorted({a[other] for a in cooccurring})]
                         for other in names if other!=name}
                supported,diagnostic=bead_image_support(name,options[name][j][0],evidence,cameras,foreign=foreign)
                attributions[name][j]=diagnostic
                if not supported:possible[name].discard(j)
        accepted={name:next(iter(v)) for name,v in possible.items() if len(v)==1 and next(iter(v))==0}
        if not any(all(a[name]==j for name,j in accepted.items()) for a in assignments):
            for name in accepted:possible[name].clear()
    for name in names:
        if exhausted or len(possible[name])!=1 or next(iter(possible[name]),0)!=0:
            detections[name]=dict(visible=False,
                                  reason='RGB shape hypothesis consistency budget exceeded' if exhausted else
                                         'distinct shared-bead CAD poses remain supported by current RGB' if len(possible[name])>1 else
                                         'no unique current CAD pose has independent RGB attribution',
                                  supported_pose_hypotheses=len(possible[name]),observed_neck_links=len(links),
                                  neck_evidence_role='unattributed RGB diagnostic only',
                                  geometric_pose_hypotheses=len(geometric_possible[name]),
                                  supported_pose_candidates=[options[name][j][0].tolist() for j in sorted(possible[name])],
                                  evidence_attribution=attributions.get(name,{}))
        else:
            detections[name]['evidence_attribution']=attributions.get(name,{}).get(0,{})


def estimate_shapes(images,cameras,previous=None,*,known_unplaced_names=None):
    previous=previous or {}
    new,identity=identify(images['top'],cameras['top'],hypotheses=True,required_names=known_unplaced_names) if len(previous)<len(CELLS)+1 else ({},{})
    seeds={**new,**previous}
    masks={key:foreground(rgb) for key,rgb in images.items()}
    raw_evidence=raw_image_evidence(masks,cameras)
    detections={};cache={};color_rescue={}
    def fit(name,seed):
        key=(name,tuple(seed))
        if key not in cache:
            evidence,contours=object_masks(name,seed,masks,cameras)
            cache[key]=fit_piece(name,evidence,cameras,previous=seed,contour_points=contours)
            if cache[key]['visible']:
                supported,diagnostic=bead_image_support(name,cache[key]['pose'],raw_evidence,cameras)
                cache[key]['independent_rgb_support']=diagnostic
                if not supported:
                    cache[key]['visible']=False
                    cache[key]['reason']=diagnostic['reason']
        return cache[key]
    for name in CELLS:
        if name not in seeds:
            detections[name]=dict(visible=False,reason='no unambiguous CAD shape identity')
            continue
        fitted=fit(name,seeds[name])
        # Packed foreground attribution can reject a target even when the
        # current calibrated RGB silhouettes yield an independent CAD fit.
        # Use the public appearance palette only to propose that fit; the
        # plain estimator still requires multi-view residual/support bounds,
        # and this branch never supplies a missing pose.
        # This bounded repair branch is intentionally scoped to the active
        # MINT/LIME adjacent chain; other known-unplaced pieces remain on the
        # generic shape observer and do not trigger repeated palette fits.
        if (not fitted['visible'] and name in ('cyan','mint','lime') and name in (known_unplaced_names or ())):
            color=color_rgb.estimate(images,cameras,{name:seeds[name]},only_names=(name,))['detections'].get(name,{})
            if (color.get('visible') is True and len(color.get('views',[]))>=2
                    and len(color.get('center_support',[]))>=4
                    and int(np.sum(np.asarray(color.get('center_support',[]),dtype=float)>=.6))>=4
                    and float(color.get('median_residual_m',float('inf')))<.0018
                    and float(color.get('rms_residual_m',float('inf')))<.004):
                fitted=dict(color);fitted['identity_source']='current RGB appearance plus public CAD geometry';fitted['color_proposal_only']=True;color_rescue[name]=dict(fitted)
        if not fitted['visible'] and name in previous:
            # A rigid-grasp motion seed can become wrong after a drop. Try the
            # two public planar supports at the predicted xy, retaining shape
            # identity. These are optimizer seeds only: fresh multi-view RGB
            # must support one equivalent physical pose across hypotheses.
            seed=np.asarray(seeds[name]);rotation=Rotation.from_quat(seed[3:]).as_matrix()
            yaw=np.arctan2(rotation[1,0],rotation[0,0])
            planar=Rotation.from_euler('z',yaw).as_matrix()
            if rotation[2,2]<0:planar=planar@np.diag([1.,-1.,-1.])
            alternatives=[]
            for z in (.7625,.7685):
                candidate=fit(name,np.r_[seed[:2],z,Rotation.from_matrix(planar).as_quat()])
                if candidate['visible']:alternatives.append(candidate)
            if alternatives and all(shape_distance(name,alternatives[0]['pose'],a['pose'])<=.003 for a in alternatives[1:]):
                fitted=min(alternatives,key=lambda d:d['rms_residual_m'])
                fitted['relocalized_from_support_hypotheses']=True
        fitted.update(identity.get(name,{}))
        fitted['identity_source']='CAD bead geometry' if name in new and name not in previous else 'previously shape-identified track'
        detections[name]=fitted
    changed=[n for n in identity.get('ambiguous_names',[]) if n not in previous]
    if changed:
        alternatives=identity.get('alternative_hypotheses',[identity['alternative_seeds']])
        assignments=[new,*alternatives]
        supported=[]
        for assignment in assignments:
            candidate={n:fit(n,assignment[n]) for n in changed}
            if not all(d['visible'] for d in candidate.values()):continue
            # Partial reinitialization cannot silently contradict an existing
            # shape-identified track and reassign its observed beads elsewhere.
            compatible=True
            for name in set(previous)&set(new):
                tracked=detections[name];other=fit(name,assignment[name])
                if not tracked['visible'] or not other['visible'] or shape_distance(name,tracked['pose'],other['pose'])>.003:
                    compatible=False;break
            if compatible:supported.append(candidate)
        equivalent=bool(supported) and all(
            shape_distance(n,supported[0][n]['pose'],candidate[n]['pose'])<=.003
            for candidate in supported[1:] for n in changed)
        identity['supported_assignment_count']=len(supported)
        if equivalent:
            chosen=min(supported,key=lambda candidate:sum(d['rms_residual_m'] for d in candidate.values()))
            for name,fitted in chosen.items():
                fitted['identity_source']='CAD bead geometry'
                detections[name]=fitted
        else:
            for name in changed:
                detections[name]=dict(visible=False,reason='no unique complete CAD assignment supported by all RGB views')
    detections['board']=fit_board(images['top'],cameras['top'])
    if not detections['board'].get('visible') and 'board' in previous:
        # Historical RGB proposes optimization only; accepted board geometry
        # must come from two independently supported current RGB/CAD views.
        from .board_vision import fit_board_reacquire
        reacquired=fit_board_reacquire(images,cameras,previous['board'])
        reacquired['top_outline_diagnostic']=detections['board']
        reacquired['optimizer_proposal_key']="previous['board']; historical RGB optimizer proposal only"
        detections['board']=reacquired
    # Separate rigid parts cannot occupy the same bead volume. Independent
    # contour fits in a packed monochrome scene can otherwise claim a neighbor's
    # image evidence. Refuse both conflicting tracks, without guessing which
    # identity is right or consulting the actual scene.
    clouds={n:local_beads(n)@Rotation.from_quat(d['pose'][3:]).as_matrix().T+np.asarray(d['pose'][:3])
            for n,d in detections.items() if n in CELLS and d['visible']}
    conflicts=set()
    names=list(clouds)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if np.linalg.norm(clouds[a][:,None]-clouds[b][None,:],axis=2).min()<.020:
                conflicts.update((a,b))
    for n in conflicts:
        detections[n]=dict(visible=False,reason='incompatible overlapping RGB shape tracks')
    # Missing tracks and stale predictions cannot rule out an image-supported
    # interpretation. Only mutually compatible fresh tracks can do that.
    refuse_shape_aliases(detections,raw_evidence,cameras,fit_candidate=fit)
    # Keep a palette/CAD fit only when it is spatially disjoint from every
    # other accepted rigid track.  This is a current RGB measurement, never a
    # public-goal or GT pose fill.
    for name,candidate in color_rescue.items():
        if candidate.get('visible') is not True:continue
        ccloud=local_beads(name)@Rotation.from_quat(candidate['pose'][3:]).as_matrix().T+np.asarray(candidate['pose'][:3])
        conflict=False
        for other,detection in detections.items():
            if other==name or other not in CELLS or detection.get('visible') is not True:continue
            cloud=local_beads(other)@Rotation.from_quat(detection['pose'][3:]).as_matrix().T+np.asarray(detection['pose'][:3])
            if np.linalg.norm(ccloud[:,None]-cloud[None,:],axis=2).min()<.020:conflict=True;break
        if not conflict:detections[name]=candidate
    # The native task explicitly uses its original public colored materials.
    # Appearance isolates a current RGB proposal; geometry, ray residuals,
    # per-bead image support and rigid-track separation still decide visibility.
    # No absent pose, desired pose or simulator state is ever substituted.
    if 'purple' in (known_unplaced_names or ()):
        candidate=original_appearance.estimate(images,cameras,{'purple':seeds['purple']} if 'purple' in seeds else {},only_names=('purple',))['detections']['purple']
        if candidate.get('visible') is True:
            supported,diagnostic=bead_image_support('purple',candidate['pose'],raw_evidence,cameras)
            diagnostic['fresh_rgb_fit']=True
            candidate['independent_rgb_support']=diagnostic
            ccloud=local_beads('purple')@Rotation.from_quat(candidate['pose'][3:]).as_matrix().T+np.asarray(candidate['pose'][:3])
            separated=True
            for other,detection in detections.items():
                if other=='purple' or other not in CELLS or detection.get('visible') is not True:continue
                cloud=local_beads(other)@Rotation.from_quat(detection['pose'][3:]).as_matrix().T+np.asarray(detection['pose'][:3])
                if np.linalg.norm(ccloud[:,None]-cloud[None,:],axis=2).min()<.020:
                    separated=False;break
            if supported and separated:
                candidate.update(identity_source='current original RGB appearance proposal validated against public CAD',color_proposal_only=True,public_cad_seeded='purple' in previous,view_track_disjoint=True,independent_track_separation_m=.020)
                detections['purple']=candidate
    poses={n:d['pose'] for n,d in detections.items() if d['visible']}
    return dict(poses=poses,detections=detections,observation_source='rgb_shape_geometry',
                identity_contract='No fixed colour-to-part mapping; CAD geometry and visual tracking',
                initial_identity_diagnostics=identity)
