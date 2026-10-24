"""The 55 beads transcribed from source video at 61 s; metres, bench frame."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PITCH = .025
RADIUS = PITCH/2
ORIGIN = np.array([.6, 0., .75])
BOARD_AT = np.array([-.135, 0., .003])
BEAD_Z = .003 + RADIUS
# Rows top to bottom, columns left to right in video/flat_solution.png.
GRID = [
    "YPCCLLLMMMD",
    "YPPCLBLMMDD",
    "YYPPOBKKKDU",
    "YTOOOBBRKKU",
    "TTTORRRRUUU",
]
NAMES = dict(Y="yellow", P="purple", C="cyan", L="lime", M="mint", D="maroon",
             T="teal", O="orange", B="navy", K="pink", R="red", U="azure")
RGB = dict(Y=(1.,.69,.01),P=(.31,.04,.46),C=(.07,.73,.98),L=(.53,.85,.03),
           M=(.25,.88,.66),D=(.64,.025,.10),T=(.01,.48,.44),O=(1.,.25,.01),
           B=(.08,.08,.68),K=(.96,.19,.59),R=(.88,.035,.06),U=(.015,.43,.91))
CELLS = {NAMES[k]: [(r,c) for r,row in enumerate(GRID) for c,v in enumerate(row) if v==k] for k in NAMES}
COLORS = {NAMES[k]:v for k,v in RGB.items()}
MISSING = ("navy", "red", "azure")

def xy(cells):
    return np.array([[(2-r)*PITCH, (5-c)*PITCH] for r,c in cells])

def local_beads(name):
    p = xy(CELLS[name]); p -= p.mean(axis=0)
    return np.c_[p, np.zeros(len(p))]

def target_center(name):
    return np.r_[xy(CELLS[name]).mean(axis=0),BEAD_Z]

def rotation(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])

def orientations(cells):
    """All planar rotations, canonicalized for an exact-cover solver."""
    out=[]
    for k in range(4):
        pts=[(int(r),int(c)) for r,c in cells]
        for _ in range(k): pts=[(-c,r) for r,c in pts]
        lo=np.min(pts,axis=0)
        shape=tuple(sorted((r-int(lo[0]),c-int(lo[1])) for r,c in pts))
        if shape not in [p[1] for p in out]:out.append((k,shape))
    return out

def solve(occupied, remaining=MISSING):
    """Exact cover of the unoccupied cells, without a pre-recorded placement answer."""
    empty={(r,c) for r in range(5) for c in range(11)}-set(map(tuple,occupied))
    if len(empty)!=sum(len(CELLS[n]) for n in remaining):return None
    options={}
    for name in remaining:
        opts=[]
        for k,shape in orientations(CELLS[name]):
            for r in range(5-max(v[0] for v in shape)):
                for c in range(11-max(v[1] for v in shape)):
                    footprint=frozenset((r+a,c+b) for a,b in shape)
                    if footprint<=empty: opts.append((k,footprint))
        options[name]=opts
    def visit(free,left):
        if not left:return {} if not free else None
        candidates={cell:[(n,k,f) for n in left for k,f in options[n] if cell in f and f<=free] for cell in free}
        cell=min(candidates,key=lambda c:len(candidates[c]))
        for n,k,f in candidates[cell]:
            ans=visit(free-f,[x for x in left if x!=n])
            if ans is not None:return {n:dict(turns=k,cells=sorted(f)),**ans}
        return None
    return visit(empty,list(remaining))
