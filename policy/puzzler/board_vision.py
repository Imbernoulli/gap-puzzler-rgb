"""Robust planar case-outline fit from calibrated overhead RGB only."""
import math
from functools import lru_cache

import cv2
import numpy as np
from scipy.optimize import least_squares


def fit_board_outline(rgb,camera):
    h,w=rgb.shape[:2]
    yy,xx=np.indices((h,w))
    world=camera.plane(np.c_[xx.ravel(),yy.ravel()],.766).reshape(h,w,3)
    roi=(world[...,0]>.25)&(world[...,0]<.85)&(np.abs(world[...,1])<.22)
    mask=((rgb.max(axis=2)<32)&roi).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:return dict(visible=False,reason='case outline absent')
    contour=max(contours,key=cv2.contourArea)
    points=camera.plane(contour[:,0],.766)[:,:2].astype(float)
    rect=cv2.minAreaRect(points.astype(np.float32));corners=cv2.boxPoints(rect).astype(float)
    sides=np.roll(corners,-1,axis=0)-corners;lengths=np.linalg.norm(sides,axis=1)
    if not (.20<lengths.min()<.24 and .30<lengths.max()<.33):
        return dict(visible=False,reason='case outline dimensions inconsistent')
    ex=sides[int(lengths.argmin())]/lengths.min()
    if ex[0]<0:ex=-ex
    start=np.r_[corners.mean(axis=0),math.atan2(ex[1],ex[0])]
    points=points[::max(1,len(points)//800)]
    half=np.array([.110,.1575])  # Public case CAD: rim, hinge and open lid.

    def local(x):
        c,s=math.cos(x[2]),math.sin(x[2]);R=np.array([[c,-s],[s,c]])
        return (points-x[:2])@R

    def residual(x):
        delta=np.abs(local(x))-half
        # Signed distance to the finite rectangle boundary. Gross occluder
        # edges get bounded influence; fixed CAD dimensions cannot expand to
        # enclose a robot finger connected to the dark outline.
        return np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(delta.max(axis=1),0)

    fit=least_squares(residual,start,loss='soft_l1',f_scale=.001,max_nfev=160,ftol=1.e-5,xtol=1.e-6,
                      bounds=(start-np.array([.02,.02,.10]),start+np.array([.02,.02,.10])))
    loc=local(fit.x);errors=np.abs(residual(fit.x))
    spans=[]
    for axis in (0,1):
        for sign in (-1,1):
            selected=(np.abs(loc[:,axis]-sign*half[axis])<.0015)&(np.abs(loc[:,1-axis])<half[1-axis]+.0015)
            spans.append(float(np.ptp(loc[selected,1-axis])) if selected.sum()>1 else 0.)
    # Three supported sides, including an opposite pair, are necessary to
    # constrain translation/yaw. A prior pose cannot replace missing edges.
    supported=sum(s>.035 for s in spans)>=3 and (min(spans[:2])>.035 or min(spans[2:])>.035)
    diagnostic=dict(edge_support_m=spans,median_outline_residual_m=float(np.median(errors)),iterations=fit.nfev)
    if not fit.success:
        return dict(visible=False,reason='case outline optimization did not converge',**diagnostic)
    if not supported:
        return dict(visible=False,reason='case outline lacks three supported sides',**diagnostic)
    if float(np.median(errors))>.0015:
        return dict(visible=False,reason='case outline is inconsistent with CAD',**diagnostic)
    yaw=float(fit.x[2]);ex=np.array([math.cos(yaw),math.sin(yaw)])
    center=fit.x[:2]-ex*.0275
    return dict(visible=True,pose=[*map(float,center),.753,0.,0.,math.sin(yaw/2),math.cos(yaw/2)],
                assumption='level table support; board yaw within +/-90 degrees',
                extent_m=[.220,.315],edge_support_m=spans,
                median_outline_residual_m=float(np.median(errors)),method='robust fixed-CAD outline')


@lru_cache(maxsize=1)
def _case_components():
    """Public assets.py case boxes and 24-section hinge, without a renderer."""
    def box(ext, at):
        return np.array([[sx*ext[0]/2,sy*ext[1]/2,sz*ext[2]/2]
                         for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])+at
    parts=[box((.146,.296,.006),(0,0,0))]
    parts.extend(box((.009,.315,.016),(a,0,.005)) for a in (-.078,.078))
    parts.extend(box((.165,.009,.016),(0,a,.005)) for a in (-.153,.153))
    angles=np.arange(24)*2*math.pi/24
    parts.append(np.array([[.095+.013*math.cos(a),y,.014+.013*math.sin(a)]
                           for a in angles for y in (-.1575,.1575)]))
    parts.append(box((.045,.315,.004),(.115,0,.003)))
    return tuple(parts)


def _projected_case(camera,x):
    from scipy.spatial import ConvexHull
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    c,s=math.cos(x[2]),math.sin(x[2]);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    polygons=[];source=[]
    for vertices in _case_components():
        world=vertices@rotation.T+np.r_[x[:2],.753]
        points=camera.plane(camera.project(world),.766)[:,:2]
        ids=ConvexHull(points).vertices
        poly=points[ids];local=vertices[ids]
        polygons.append(Polygon(poly))
        source.extend((poly[j],poly[(j+1)%len(poly)],local[j],local[(j+1)%len(poly)])
                      for j in range(len(poly)))
    outline=unary_union(polygons)
    if outline.geom_type!='Polygon':
        raise ValueError('case CAD projection is disconnected')
    return np.asarray(outline.exterior.coords)[:-1],source


def _fit_case_view(rgb,camera,proposal):
    h,w=rgb.shape[:2];yy,xx=np.indices((h,w))
    world=camera.plane(np.c_[xx.ravel(),yy.ravel()],.766).reshape(h,w,3)
    roi=(world[...,0]>.25)&(world[...,0]<.85)&(np.abs(world[...,1])<.22)
    mask=((rgb.max(axis=2)<32)&roi).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:return dict(visible=False,reason='case outline absent')
    contour=max(contours,key=cv2.contourArea)
    points=camera.plane(contour[:,0],.766)[:,:2].astype(float)
    points=points[::max(1,len(points)//800)]
    if len(points)<80:return dict(visible=False,reason='case outline has too few pixels')

    def residual(x, details=False):
        polygon,source=_projected_case(camera,x)
        ends=np.roll(polygon,-1,axis=0);axes=ends-polygon
        fractions=np.clip(np.sum((points[:,None]-polygon[None])*axes[None],axis=2)
                          /np.sum(axes**2,axis=1),0,1)
        nearest=polygon[None]+fractions[...,None]*axes[None]
        distances=np.linalg.norm(points[:,None]-nearest,axis=2)
        closest=distances.argmin(axis=1)
        poly32=polygon.astype(np.float32)
        inside=np.array([cv2.pointPolygonTest(poly32,tuple(p),False)>=0 for p in points])
        signed=distances.min(axis=1)*np.where(inside,-1,1)
        return (signed,closest,polygon,source) if details else signed

    bounds=(proposal-np.array([.02,.02,.10]),proposal+np.array([.02,.02,.10]))
    fit=least_squares(residual,proposal,loss='soft_l1',f_scale=.001,max_nfev=160,
                      ftol=1.e-5,xtol=1.e-6,bounds=bounds)
    first_errors=residual(fit.x)
    inliers=(first_errors>-.0014)&(first_errors<.0025)
    if fit.success and inliers.sum()>=80:
        fit=least_squares(lambda x:residual(x)[inliers],fit.x,loss='soft_l1',
                          f_scale=.0005,max_nfev=160,ftol=1.e-5,xtol=1.e-6,bounds=bounds)
    errors,closest,polygon,source=residual(fit.x,True)
    # Attribute supported pixels to real fixed-CAD outer edges. Perspective
    # plane lengths cannot substitute for physical CAD edge support lengths.
    groups=[[],[],[],[]]
    for j,(a,b) in enumerate(zip(polygon,np.roll(polygon,-1,axis=0))):
        midpoint=(a+b)/2
        candidates=[]
        for start,end,local_start,local_end in source:
            axis=end-start;fraction=np.clip(np.dot(midpoint-start,axis)/np.dot(axis,axis),0,1)
            candidates.append((np.linalg.norm(midpoint-start-fraction*axis),start,end,local_start,local_end))
        distance,start,end,local_start,local_end=min(candidates,key=lambda row:row[0])
        if distance>1.e-7:continue
        delta=local_end-local_start
        physical_mid=(local_start+local_end)/2
        if abs(delta[1])>.035 and abs(delta[0])<1.e-7:
            side=0 if abs(physical_mid[0]+.0825)<.0015 else 1 if abs(physical_mid[0]-.1375)<.0015 else None
            axis=1
        elif abs(delta[0])>.035 and abs(delta[1])<1.e-7:
            side=2 if abs(physical_mid[1]+.1575)<.0015 else 3 if abs(physical_mid[1]-.1575)<.0015 else None
            axis=0
        else:continue
        if side is None:continue
        chosen=(closest==j)&(np.abs(errors)<.0015)
        # Intersect each actual image ray with this known 3-D CAD edge.
        c,s=math.cos(fit.x[2]),math.sin(fit.x[2]);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        origin=local_start@rotation.T+np.r_[fit.x[:2],.753]
        direction=delta@rotation.T
        plane_points=np.c_[points[chosen],np.full(chosen.sum(),.766)]
        rays=plane_points-camera.camera_to_world[:3,3]
        for ray in rays:
            fraction=np.linalg.lstsq(np.c_[direction,-ray],camera.camera_to_world[:3,3]-origin,rcond=None)[0][0]
            groups[side].append(float(local_start[axis]+np.clip(fraction,0,1)*delta[axis]))
    spans=[float(np.ptp(group)) if len(group)>1 else 0. for group in groups]
    supported=sum(span>.035 for span in spans)>=3 and (min(spans[:2])>.035 or min(spans[2:])>.035)
    median=float(np.median(np.abs(errors)))
    diagnostic=dict(edge_support_m=spans,median_outline_residual_m=median,
                    contour_points=len(points),inlier_points=int(inliers.sum()),iterations=fit.nfev)
    if not fit.success:return dict(visible=False,reason='case outline optimization did not converge',**diagnostic)
    if not supported:return dict(visible=False,reason='case outline lacks three supported sides',**diagnostic)
    if median>.0015:return dict(visible=False,reason='case outline is inconsistent with CAD',**diagnostic)
    yaw=float(fit.x[2])
    return dict(visible=True,pose=[*map(float,fit.x[:2]),.753,0.,0.,math.sin(yaw/2),math.cos(yaw/2)],**diagnostic)


def _fit_board_front_rear(images,cameras,proposal_pose):
    """Current RGB reacquisition; historical RGB pose proposes optimization only.

    Both front and rear must independently pass the original 1.5mm complete
    contour residual and three physical CAD edge support gates. Their maximum
    corresponding case-corner separation must also be below 1.5mm. No pose is
    returned for an absent/unsupported current image, regardless of proposal.
    """
    proposal_pose=np.asarray(proposal_pose,dtype=float)
    if proposal_pose.shape!=(7,) or not np.isfinite(proposal_pose).all():
        return dict(visible=False,reason='invalid historical RGB optimizer proposal')
    if not all(key in images and key in cameras for key in ('front','rear')):
        return dict(visible=False,reason='case reacquisition needs front and rear RGB')
    proposal=np.r_[proposal_pose[:2],2*math.atan2(proposal_pose[5],proposal_pose[6])]
    views={key:_fit_case_view(images[key],cameras[key],proposal) for key in ('front','rear')}
    if not all(view.get('visible') for view in views.values()):
        return dict(visible=False,reason='case reacquisition lacks two independent supported views',views=views)
    def corners(pose):
        yaw=2*math.atan2(pose[5],pose[6]);c,s=math.cos(yaw),math.sin(yaw)
        local=np.array([[.0275+sx*.110,sy*.1575] for sx in (-1,1) for sy in (-1,1)])
        return local@np.array([[c,s],[-s,c]])+pose[:2]
    difference=float(np.linalg.norm(corners(views['front']['pose'])-corners(views['rear']['pose']),axis=1).max())
    if difference>=.0015:
        return dict(visible=False,reason='independent case RGB corner fits disagree',corner_difference_m=difference,views=views)
    selected=min(views,key=lambda key:views[key]['median_outline_residual_m'])
    result=views[selected]
    return dict(result,method='two-view projected fixed-CAD outline reacquisition',
                optimizer_proposal_source='historical RGB only; not current measurement',
                assumption='level table support; board yaw within +/-90 degrees',extent_m=[.220,.315],
                independent_corner_difference_m=difference,selected_view=selected,views=views)


def fit_board_reacquire(images,cameras,proposal_pose):
    """Retain front/rear first; otherwise require two supported current views.

    Every adopted view passes the same complete-contour residual and physical
    CAD edge-support gates. All adopted independent corner fits must agree.
    A historical RGB pose remains an optimizer proposal, without a timestamp
    or measured-pose substitution. Images from different epochs are not merged.
    """
    original=_fit_board_front_rear(images,cameras,proposal_pose)
    if original.get('visible') or 'views' not in original:return original
    proposal_pose=np.asarray(proposal_pose,dtype=float)
    proposal=np.r_[proposal_pose[:2],2*math.atan2(proposal_pose[5],proposal_pose[6])]
    views=dict(original['views'])
    for key in ('top','side'):
        if key in images and key in cameras:
            views[key]=_fit_case_view(images[key],cameras[key],proposal)
    accepted={key:view for key,view in views.items() if view.get('visible') is True}
    if len(accepted)<2:
        return dict(visible=False,reason='case reacquisition lacks two independent supported views',views=views)
    def corners(pose):
        yaw=2*math.atan2(pose[5],pose[6]);c,s=math.cos(yaw),math.sin(yaw)
        local=np.array([[.0275+sx*.110,sy*.1575] for sx in (-1,1) for sy in (-1,1)])
        return local@np.array([[c,s],[-s,c]])+pose[:2]
    clouds=[corners(view['pose']) for view in accepted.values()]
    difference=max(float(np.linalg.norm(a-b,axis=1).max())
                   for i,a in enumerate(clouds) for b in clouds[i+1:])
    if difference>=.0015:
        return dict(visible=False,reason='independent case RGB corner fits disagree',corner_difference_m=difference,views=views)
    selected=min(accepted,key=lambda key:accepted[key]['median_outline_residual_m'])
    return dict(accepted[selected],method='two-view projected fixed-CAD outline reacquisition',
                optimizer_proposal_source='historical RGB only; not current measurement',
                assumption='level table support; board yaw within +/-90 degrees',extent_m=[.220,.315],
                independent_corner_difference_m=difference,selected_view=selected,views=accepted,
                rejected_view_diagnostics={key:view for key,view in views.items() if key not in accepted})
