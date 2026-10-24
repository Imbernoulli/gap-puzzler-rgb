"""Exact cover of the 5x5 + 4x4 + 3x3 + 2x2 + 1 sphere pyramid.

The integer lattice has metric diag(1,1,2)/4, so its 12 nearest-neighbour
vectors have identical physical length. All orthogonal direction pairs give
real rotations of the same flat rigid pieces, including tilted placements.
"""
import itertools,json,time
from functools import lru_cache
import numpy as np
from . import model as m

SITES=tuple((2*i+k,2*j+k,k) for k in range(5) for i in range(5-k) for j in range(5-k))
INDEX={p:i for i,p in enumerate(SITES)}
DIRS=[(a,b,c) for a,b,c in itertools.product(range(-2,3),range(-2,3),range(-1,2)) if a*a+b*b+2*c*c==4]

def physical(points):
    p=np.array(points,dtype=float)
    return np.column_stack(((p[:,0]-4)*m.PITCH/2,(p[:,1]-4)*m.PITCH/2,m.BEAD_Z+p[:,2]*m.PITCH/np.sqrt(2)))

def candidates(upright=False,robot_facing=False,terminal_top=False,outward=False):
    ans=[]
    for name,cells in m.CELLS.items():
        native=[(-r,-c) for r,c in cells];u0,v0=native[0];native=[(u-u0,v-v0) for u,v in native]
        degree=[sum(abs(u-x)+abs(v-y)==1 for x,y in native) for u,v in native]
        seen=set()
        for a in DIRS:
            for b in DIRS:
                if a[0]*b[0]+a[1]*b[1]+2*a[2]*b[2]:continue
                offsets=[tuple(u*a[k]+v*b[k] for k in range(3)) for u,v in native]
                aa=np.array([a[0]/2,a[1]/2,a[2]/np.sqrt(2)])
                bb=np.array([b[0]/2,b[1]/2,b[2]/np.sqrt(2)])
                R=np.column_stack((aa,bb,np.cross(aa,bb)))
                if upright and R[2,2]<-.01:continue
                if robot_facing and (robot_facing is True or name in robot_facing) and abs(R[2,2])<.01:
                    from .task import staging
                    arm=staging()[name][2]
                    if R[0,2]>-.01:continue
                    if name=='maroon' and R[1,2]*(1 if arm==0 else -1)<.01:continue
                top_indices=[i for i,o in enumerate(offsets) if o[2]==max(p[2] for p in offsets) and degree[i]<=1]
                require_terminal = terminal_top is True or bool(terminal_top and name in terminal_top)
                require_outward = outward is True or bool(outward and name in outward)
                if require_terminal and abs(R[2,2])<.01 and not top_indices:continue
                for t in SITES:
                    pts=tuple(sorted(tuple(x[k]+t[k] for k in range(3)) for x in offsets))
                    if pts in seen or not all(p in INDEX for p in pts):continue
                    if require_outward and abs(R[2,2])<.01:
                        exposed=False
                        for i in top_indices:
                            site=np.array(offsets[i])+t;z=site[2]
                            if z==4 or (site[0] in (z,8-z) and site[1] in (z,8-z)
                                        and np.dot(site[:2]-4,R[:2,2])>.1):exposed=True
                        if not exposed:continue
                    seen.add(pts);mask=sum(1<<INDEX[p] for p in pts)
                    ans.append(dict(name=name,sites=pts,mask=mask,R=R.tolist()))
    return ans

def solve(timeout=120,vertical=False,upright=False,robot_facing=False,terminal_top=False,outward=False):
    start=time.monotonic();options=candidates(upright=upright,robot_facing=robot_facing,terminal_top=terminal_top,outward=outward);names=list(m.CELLS)
    per_cell=[[] for _ in SITES]
    for o in options:
        o['piecebit']=1<<names.index(o['name'])
        o['belowmask']=sum(1<<INDEX[p] for p in set((x+dx,y+dy,z-1) for x,y,z in o['sites'] for dx in [-1,1] for dy in [-1,1]) if p in INDEX)
        for p in o['sites']:per_cell[INDEX[p]].append(o)
    calls=0
    def dfs(used,occupied,path,adj):
        nonlocal calls
        calls+=1
        if calls%4096==0 and time.monotonic()-start>timeout:raise TimeoutError('Pyramid search budget exhausted')
        if len(path)==12:return path
        best=None
        for i in range(55):
            if occupied>>i&1:continue
            valid=[o for o in per_cell[i] if not(o['piecebit']&used or o['mask']&occupied)]
            if not valid:return None
            if best is None or len(valid)<len(best):best=valid
            if len(best)==1:break
        # Prefer pieces resting on the floor, with shallow spans, to ease assembly.
        best.sort(key=lambda o:(min(p[2] for p in o['sites']),max(p[2] for p in o['sites'])))
        for o in best:
            nex=adj.copy();ni=names.index(o['name'])
            if vertical:
                for older in path:
                    oi=names.index(older['name'])
                    if o['belowmask']&older['mask']:nex[oi]|=o['piecebit']
                    if older['belowmask']&o['mask']:nex[ni]|=older['piecebit']
                for k in range(12):
                    for i in range(12):
                        if nex[i]>>k&1:nex[i]|=nex[k]
                if any(v>>i&1 for i,v in enumerate(nex)):continue
            ans=dfs(used|o['piecebit'],occupied|o['mask'],path+[o],nex)
            if ans is not None:return ans
        return None
    ans=dfs(0,0,[],[0]*12)
    if ans is None:raise ValueError('No pyramid packing')
    result={o['name']:dict(sites=o['sites'],R=o['R'],center=physical(o['sites']).mean(axis=0).tolist()) for o in ans}
    return dict(placements=result,search_nodes=calls,search_wall_s=time.monotonic()-start)

@lru_cache(maxsize=8)
def get_plan(filename='pyramid_upright_plan.json'):
    p=m.ROOT/'assets'/filename
    if not p.exists():p=m.ROOT/'assets/pyramid_plan.json'
    if not p.exists():p.write_text(json.dumps(solve(),indent=2))
    return json.loads(p.read_text())

if __name__=='__main__':
    p=get_plan();print(json.dumps(p,indent=2))
