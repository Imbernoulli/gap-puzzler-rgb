"""Dynamic board, 12 rigid bead pieces and unassisted final-state grading."""
from dataclasses import dataclass,asdict
import contextlib
import numpy as np
from . import model as m

@dataclass(frozen=True)
class Variant:
    mode:str='flat'
    embodiment:str='yam_bimanual'
    pyramid_plan:str='pyramid_upright_plan.json'
    preset_pieces:tuple[str,...]=()
    seed:int=0
    board_dx:float=0.
    board_dy:float=0.
    board_yaw:float=0.
    piece_jitter:float=0.
    piece_yaw:float=0.
    friction:float=.65
    grasp_bias:float=0.
    scattered:bool=False
    appearance:str='original'

def sample(seed,mode='flat'):
    rng=np.random.default_rng(seed)
    return Variant(seed=seed,mode=mode,board_dx=float(rng.uniform(-.008,.008)),board_dy=float(rng.uniform(-.008,.008)),
                   board_yaw=float(rng.uniform(-.07,.07)),piece_jitter=.008,piece_yaw=.28,
                   friction=float(rng.uniform(.45,.85)),grasp_bias=float(rng.uniform(-.002,.002)))

CURRENT={}
COMPLETE_ATOM='puzzle_complete_home_v2'

def placement(v,name):
    if v.mode=='pyramid':
        from .pyramid import get_plan
        p=get_plan(v.pyramid_plan)['placements'][name]
        return np.array(p['center']),np.array(p['R'])
    return m.target_center(name),np.eye(3)

def staging():
    """Loose source pieces on both sides of the empty case, within arm reach."""
    result={}
    groups={0:['orange','purple','yellow','lime','teal','cyan'],1:['azure','navy','pink','maroon','mint','red']}
    for side,names in groups.items():
        cursor=-.29
        for name in names:
            p=m.local_beads(name)
            yaw=np.pi/2 if np.ptp(p[:,0])>np.ptp(p[:,1]) else 0.
            width=float(np.ptp((p@m.rotation(yaw).T)[:,0])+2*m.RADIUS)
            center=cursor+width/2;cursor+=width+.012
            result[name]=(np.array([center,.27 if side==0 else -.27,m.RADIUS+.001]),yaw,side)
    return result

def scattered_staging(seed):
    """Episode generation only: random non-overlapping loose pieces, both sides.

    This function is never imported by the visual controller. It defines the
    scene distribution; it is not an observation or a source-position lookup.
    """
    rng=np.random.default_rng(seed)
    for restart in range(30):
        occupied=[];result={}
        for name in rng.permutation(list(m.CELLS)):
            for attempt in range(1500):
                side=int(rng.integers(2));yaw=float(rng.uniform(-np.pi,np.pi))
                center=np.array([rng.uniform(-.28,.23),rng.uniform(.235,.34)*(1 if side==0 else -1),m.RADIUS+.001])
                points=m.local_beads(name)@m.rotation(yaw).T+center
                if (np.abs(points[:,1]).max()>.39 or np.abs(points[:,1]).min()<.19
                    or points[:,0].min()<-.31 or points[:,0].max()>.29):continue
                if occupied and np.linalg.norm(points[:,None]-np.concatenate(occupied)[None,:],axis=2).min()<.030:continue
                occupied.append(points);result[name]=(center,yaw,side);break
            else:break
        if len(result)==len(m.CELLS):return result
    raise RuntimeError('Could not sample a non-overlapping loose-piece layout')

def scene(v=Variant(), *, completed=False, robot=True):
    import robosimstudio as rss
    from robosimstudio.specs import AssetSpec
    from .assets import manifest
    a=manifest();sc=rss.Scene(cell='bench_075',physics=rss.MUJOCO_500HZ,name='IQ Puzzler Pro: full loose-piece assembly')
    b=m.BOARD_AT+np.array([v.board_dx,v.board_dy,0])
    R=m.rotation(v.board_yaw);q=(0,0,float(np.sin(v.board_yaw/2)),float(np.cos(v.board_yaw/2)))
    sc.add(rss.body(name='board',asset=AssetSpec(**a['board']),at=tuple(b),quat=q,mass=.28,
                    friction=v.friction,ke=1.e5,kd=1000.,color=(.055,.06,.065)))
    rng=np.random.default_rng(v.seed)
    stage=scattered_staging(v.seed) if v.scattered else staging()
    part_colors=dict(m.COLORS)
    if v.appearance=='shuffled':
        names=list(m.CELLS);palette=list(m.COLORS.values())
        order=np.random.default_rng(v.seed+104729).permutation(len(names))
        part_colors={n:palette[int(i)] for n,i in zip(names,order)}
    elif v.appearance=='monochrome':part_colors={n:(.70,.70,.70) for n in m.CELLS}
    elif v.appearance!='original':raise ValueError('Unknown appearance '+v.appearance)
    for name in m.CELLS:
        if not completed and v.mode!='completion' and name not in v.preset_pieces:
            at,yaw,side=stage[name];at=at.copy()
            at[:2]+=rng.uniform(-v.piece_jitter,v.piece_jitter,2)
            yaw+=float(rng.uniform(-v.piece_yaw,v.piece_yaw));quat=(0,0,np.sin(yaw/2),np.cos(yaw/2))
        elif name in m.MISSING and not completed and name not in v.preset_pieces:
            idx=m.MISSING.index(name)
            at=np.array([-.23+idx*.10,-.255,m.RADIUS+.001])
            at[:2]+=rng.uniform(-v.piece_jitter,v.piece_jitter,2)
            yaw=float(rng.uniform(-v.piece_yaw,v.piece_yaw));quat=(0,0,np.sin(yaw/2),np.cos(yaw/2))
        else:
            from scipy.spatial.transform import Rotation
            center,local_R=placement(v,name);at=b+R@center;quat=Rotation.from_matrix(R@local_R).as_quat()
        sc.add(rss.body(name=name,asset=AssetSpec(**a[name]),at=tuple(at),quat=tuple(quat),mass=.012*len(m.CELLS[name]),
                        friction=v.friction,ke=1.e5,kd=1000.,contact_gap=.0003,color=part_colors[name]))
    if v.embodiment!='yam_bimanual':
        from .hardware import register
        register(v.embodiment)
    if robot:sc.add(rss.robot(v.embodiment))
    from robosimstudio.core.render.finish import Finish
    sc.add(rss.slab(name='countertop_look',half=(.55,.65,.001),at=(0,0,-.001),collide=False,
           color=(.72,.69,.61),finish=Finish(roughness=.85,texture=str(m.ROOT/'assets/table_patch.png'),uv_scale=(6,6))))
    return sc

def validate_start_poses(path):
    """Validate a complete recorded native initialization, never a policy input."""
    import json
    from pathlib import Path
    path=Path(path).resolve()
    document=json.loads(path.read_text())
    objects=document.get('objects',{})
    if set(objects)!={'board',*m.CELLS}:
        raise ValueError('Recorded start poses must contain board and all twelve known pieces')
    for name,entry in objects.items():
        p=np.asarray(entry.get('pose',[]),dtype=float)
        q=np.asarray(entry.get('quat',[]),dtype=float)
        if (p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(p).all() or
                not np.isfinite(q).all() or abs(float(np.linalg.norm(q))-1.)>1.e-4 or
                entry.get('world_xform') is not True):
            raise ValueError('Invalid recorded world pose for '+name)
    return path


def task_spec(v=Variant(),max_steps=15000,*,completed=False,suite='',start_poses=None,terminate_on_success=True,contact_capacity=None):
    """Register the unchanged task document without constructing an environment.

    ``suite`` is the native ``bench.task`` registry namespace. Pool preload
    modules can give distinct initial variants distinct names and use the
    returned ``TaskSpec.full_name`` in ``JobSpec.task``. The default preserves
    the existing single-episode task name.
    """
    import robosimstudio_bench as bench
    from robosimstudio.domains.rigid import Body
    from robosimstudio.assetlib import register_asset
    register_goal()
    CURRENT.clear();CURRENT.update(variant=v,env=None)
    sc=scene(v,completed=completed)
    # Reuse the scene's specs through the task compiler and native graph executor.
    specs=list(sc.specs) if hasattr(sc,'specs') else None
    if specs is None: specs=list(sc._specs)
    entries=[]
    for spec in specs:
        if spec.name in ['board',*m.CELLS]:
            # The live process keeps its registry across scene reloads. Loose,
            # completed and randomized initial specs must have distinct keys.
            import hashlib
            asset_name='iq_'+spec.name+'_'+hashlib.sha256(repr(spec).encode()).hexdigest()[:12]
            record=Body.from_spec(spec,name=asset_name)
            register_asset(record)
            entries.append(f'{spec.name} - {asset_name}')
    start_clause=''
    if start_poses is not None:
        import json
        start_clause='(:start-poses '+json.dumps(str(validate_start_poses(start_poses)))+')'
    termination_clause='' if terminate_on_success else '(:terminate-on-success false)'
    from .rss_contact_capacity import arrangement_name
    physics_name=arrangement_name(contact_capacity)
    task=f'''(define (problem iq_puzzler_completion)
      (:staging (:scene bench_075) (:embodiment {v.embodiment}) (:physics {physics_name}) (:max-steps {max_steps}) {start_clause} {termination_clause})
      (:objects {' '.join(entries)})
      (:goal (once (hold-for 2.0 (>= ({COMPLETE_ATOM} board) 1.0)))))'''
    return bench.task(task,suite=suite)

def build(v=Variant(),max_steps=15000,*,completed=False):
    import robosimstudio_bench as bench
    env=bench.Env(task_spec(v,max_steps,completed=completed));CURRENT.clear();CURRENT.update(env=env,variant=v)
    return env

_GOAL_REGISTERED=False
def register_goal():
    """RSS's own temporal evaluator uses the same *frozen snapshot* geometry gates."""
    global _GOAL_REGISTERED
    if _GOAL_REGISTERED:return
    from types import SimpleNamespace
    from robosimstudio.assetlib.semantics.atoms import ArgSpec,AtomSpec,register_atom
    from robosimstudio.assetlib.semantics.handlers import handler
    def complete(snapshot,board):
        env=CURRENT.get('env')
        if env is None:return 0.
        class FrozenView:
            state=snapshot.ctx.state
            def __getitem__(self,name):
                i=int(env.scene.by_name[name].body)
                return SimpleNamespace(pose=snapshot.body_q[i],velocity=snapshot.body_qd[i])
            def arm(self,name):return env.arm(name)
            def gripper(self,name):return env.gripper(name)
            def state_is_finite(self):return bool(np.isfinite(snapshot.body_q).all())
        home=all(np.linalg.norm(np.asarray(env.arm(name).ee_pose(snapshot.ctx.state)[0])
                                -np.asarray(env.arm(name).home_pose[0]))<.015
                 for name in ('left','right'))
        return float(home and grade(FrozenView())['success'])
    with contextlib.suppress(ValueError):
        register_atom(AtomSpec(name=COMPLETE_ATOM,summary='All 55 beads seated, still, board resting, hands open and returned home.',
            fn=complete,args=(ArgSpec(name='board',reads='handle'),),directions=('check','observe'),
            returns=('value',),reads_scene=True,provenance='puzzler strict release and placement grader'))
        @handler(COMPLETE_ATOM,substrates=('rigid',),summary='Strict flat or pyramid completion on the current frozen state.')
        def _h(snapshot,board):return {'value':complete(snapshot,board)}
    _GOAL_REGISTERED=True

def poses(env):
    return {n:np.asarray(env[n].pose,dtype=float) for n in ['board',*m.CELLS]}

def beads(pose,name):
    from scipy.spatial.transform import Rotation
    return m.local_beads(name)@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]

def grade(env, *, previous=None):
    from scipy.spatial.transform import Rotation
    ps=poses(env);b=ps['board'];R=Rotation.from_quat(b[3:]).as_matrix()
    errors={};stable={}
    for name in m.CELLS:
        actual=(beads(ps[name],name)-b[:3])@R
        if CURRENT.get('variant',Variant()).mode=='pyramid':
            from .pyramid import get_plan,physical
            target=physical(get_plan(CURRENT['variant'].pyramid_plan)['placements'][name]['sites'])
        else:target=np.c_[m.xy(m.CELLS[name]),np.full(len(m.CELLS[name]),m.BEAD_Z)]
        from scipy.optimize import linear_sum_assignment
        cost=np.linalg.norm(actual[:,None,:]-target[None,:,:],axis=2)
        ii,jj=linear_sum_assignment(cost)
        errors[name]=float(cost[ii,jj].max())
        stable[name]=float(np.linalg.norm(np.asarray(env[name].velocity)[:3]))
    hands=[]
    for arm in ('left','right'):
        p,_=env.arm(arm).ee_pose(env.state)
        dist=min(float(np.linalg.norm(beads(ps[n],n)-p,axis=1).min()) for n in m.CELLS)
        hands.append(dict(arm=arm,clearance=dist,width=float(env.gripper(arm).width(env.state))))
    finite=bool(env.state_is_finite())
    board_level=float((R@np.array([0,0,1.]))[2])>.995
    board_resting=abs(float(b[2]-m.ORIGIN[2]-.003))<.006 and float(np.linalg.norm(np.asarray(env['board'].velocity)[:3]))<.015
    ok=finite and board_level and board_resting and max(errors.values())<.004 and max(stable.values())<.015 and all(h['clearance']>.055 and h['width']>.045 for h in hands)
    return dict(success=bool(ok),bead_error_m=errors,speed_mps=stable,hands=hands,board_pose=b.tolist(),finite=finite,board_level=board_level,board_resting=board_resting)
