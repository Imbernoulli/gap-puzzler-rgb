"""GaP execution restricted to RGB estimates and robot proprioception.

The whitelist is enforced at invocation, including late skill registration.
This is an observation boundary for our authored policy, not a security sandbox
for arbitrary hostile Python programs.
"""
from gap_core.tools import ToolRegistry
from robosimstudio_policy_agent import SolutionGraph
import json
from pathlib import Path
import traceback
from copy import deepcopy
import hashlib
import importlib.util
import sys

from . import vision_planning
from .vision_observer import RGBSensor, VisualObserver, goals


ALLOWED_TOOLS = frozenset({
    'vision.observe', 'puzzle.plan_pose', 'puzzle.plan_path', 'puzzle.plan_joint_path', 'puzzle.plan_pose_sequence', 'puzzle.propagate_pyramid_held_belief', 'puzzle.capture_conditional_attachment', 'puzzle.propagate_conditional_attachment', 'robot.execution_journal_cursor', 'robot.read_proprioception',
    'robot.get_ee_pose', 'robot.grasp_frame', 'robot.set_grip',
    'robot.execute_trajectory', 'robot.wait_steps', 'robot.go_home',
})


def required_checkpoint_mode(graph):
    """Read the frozen graph's execution requirement; never silently disable it."""
    document = json.loads((Path(graph) / 'workflow.json').read_text())
    mode = document.get('meta', {}).get('required_checkpoint_mode', 'warn')
    if mode not in ('raise', 'warn'):
        raise ValueError(f'Unsupported visual checkpoint mode: {mode!r}')
    return mode


def required_skill_registries(graph, override=None):
    """Resolve task contracts before native RSS stages scripts and pins skills."""
    graph = Path(graph).resolve()
    document = json.loads((graph / 'workflow.json').read_text())
    declared = document.get('meta', {}).get('skill_registry_paths', [])
    if not declared:
        return override
    required = [str((graph / path).resolve()) for path in declared]
    for path in required:
        if not Path(path).is_dir():
            raise ValueError(f'Required visual skill registry does not exist: {path}')
    if override is None:
        return required
    if isinstance(override, Path):
        explicit = [override]
    else:
        explicit = override.split(':') if isinstance(override, str) else override
    explicit = [str(Path(path).resolve()) for path in explicit]
    if not set(required).issubset(explicit):
        raise ValueError('Skill override omits a registry required by the frozen visual graph')
    return explicit


def load_graph_schema(graph):
    """Load the graph's frozen schema extension before native cold validation.

    The pinned engine validates custom type names before importing node scripts.
    In-process builder tests can accidentally hide that ordering by registering
    types first. Only the module vendored inside this graph supplies its types.
    """
    graph = Path(graph).resolve()
    document = json.loads((graph / 'workflow.json').read_text())
    relative = document.get('meta', {}).get('schema_module_path')
    if relative is None:
        return None
    path = (graph / relative).resolve()
    if not path.is_relative_to(graph) or path.suffix != '.py' or not path.is_file():
        raise ValueError('Frozen schema module must be a Python file inside the graph')
    key = '_puzzler_frozen_' + hashlib.sha256(str(path).encode()).hexdigest()[:20]
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(key, None)
            raise
    return sys.modules[key]


class RestrictedRegistry(ToolRegistry):
    def __init__(self, source, record_path=None, clock=None):
        super().__init__()
        self.source = source
        self.record_path = Path(record_path) if record_path else None
        self.call_count = 0
        self.clock = clock
        import uuid
        self.session_nonce=uuid.uuid4().hex
        self.pyramid_held_belief_history=[]
        self.proprioception_history = {0: [], 1: []}
        # Copy only reviewed descriptors. Never retain hidden names in catalog.
        self._tools = {n: d for n, d in source._tools.items() if n in ALLOWED_TOOLS}

    def get(self, name):
        if name not in ALLOWED_TOOLS:
            raise KeyError(f'Not in RGB policy tool contract: {name}')
        return super().get(name)

    def invoke(self, name, ctx=None, **kwargs):
        self.get(name)
        if name=='robot.set_grip'and kwargs.get('width_m',0.)>.00475:
            invalidate=getattr(self.source,'_invalidate_conditional_attachments',None)
            if invalidate is not None:invalidate(int(kwargs.get('arm_id',0)),'actualrobot openingcommand attempted; oldattachment cannotrevive')
            invalidate_owned=getattr(self.source,'_invalidate_owned_held_sources',None)
            if invalidate_owned is not None:invalidate_owned(int(kwargs.get('arm_id',0)),'actualrobot openingcommand attempted; oldownedRGB source cannotrevive')
        self.call_count += 1
        call_id = self.call_count
        args=deepcopy(kwargs)
        args_hash=hashlib.sha256(json.dumps(args,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        self._record(dict(call_id=call_id,tool=name,event='call',arguments=args,arguments_sha256=args_hash))
        from time import perf_counter
        tool_started = perf_counter()
        try:
            result=self.source.invoke(name,ctx,**kwargs)
        except Exception as error:
            self._record(dict(call_id=call_id,tool=name,event='error',error=str(error),wall_duration_s=perf_counter()-tool_started))
            raise
        tool_elapsed = perf_counter()-tool_started
        if name == 'robot.get_ee_pose' and isinstance(result, dict):
            measured = result.get('pose', result)
            if 'position' in measured and 'rotation' in measured:
                arm = int(kwargs.get('arm_id', 0))
                sample = dict(sample_id=call_id,
                              timestamp=self.clock() if self.clock else None,
                              pose=[*[float(measured['position'][k]) for k in 'xyz'],
                                    *[float(measured['rotation'][k]) for k in ('x','y','z','w')]])
                self.proprioception_history.setdefault(arm, []).append(sample)
                self.proprioception_history[arm] = self.proprioception_history[arm][-8:]
        if name in('puzzle.propagate_pyramid_held_belief','puzzle.propagate_conditional_attachment'):self.pyramid_held_belief_history.append(deepcopy(result))
        recorded=result
        if name=='vision.observe' and isinstance(result,dict):
            recorded={k:result.get(k) for k in ('timestamp','observation_source')}
            recorded['visible_names']=sorted(result.get('poses',{}))
        self._record(dict(call_id=call_id,tool=name,event='return',result=recorded,wall_duration_s=tool_elapsed))
        return result

    def _record(self,event):
        if self.record_path is None:
            return
        event.update(session_nonce=self.session_nonce,timestamp=self.clock() if self.clock else None)
        self.record_path.parent.mkdir(parents=True,exist_ok=True)
        with self.record_path.open('a') as stream:
            stream.write(json.dumps(event,default=str)+'\n')

    def runtime_tools(self):
        return {n: d for n, d in super().runtime_tools().items() if n in ALLOWED_TOOLS}

    def codegen_tools(self):
        return {n: d for n, d in super().codegen_tools().items() if n in ALLOWED_TOOLS}


class VisionGraph(SolutionGraph):
    def __init__(self, env, *, visual_mode='flat', visual_plan='pyramid_robot6_plan.json',
                 observation_dir=None, visual_identity='palette', **kwargs):
        self.sensor = RGBSensor(env)
        self.tool_record_path=Path(observation_dir)/'tools.jsonl' if observation_dir else None
        estimator=None
        if visual_identity=='shape':
            from .shape_vision import estimate_shapes
            estimator=estimate_shapes
        self.visual = VisualObserver(self.sensor.capture, self.sensor.cameras,
                                     goals(visual_mode, visual_plan), observation_dir, estimator=estimator)
        kwargs.update(privileged=False, perception='vision', task_tools=self._register_visual)
        source = SolutionGraph._graph_source(env, kwargs.get('graph', 'gt'))
        load_graph_schema(source)
        skills = required_skill_registries(source, kwargs.get('skills'))
        if skills is not None:
            kwargs['skills'] = skills
        super().__init__(env, **kwargs)
        self.visual_checkpoint_mode = required_checkpoint_mode(self.graph_path)

    def _run_graph(self, generation):
        """Keep native RSS lifecycle semantics with an explicit checkpoint mode.

        The pinned RSS runner omits this argument, which selects GaP's ``warn``
        default. Staged graphs require failures to stop execution. Their retry
        exits use separate factual checkpoints, so a legitimate safe retry is
        not tested against the place-success postcondition.
        """
        try:
            from gap import execute
            from robosimstudio_policy_agent.graph_solution import _accepts_node_callbacks

            kwargs = dict(connector=self.connector, skills=self.skills or None,
                          inputs=dict(instruction=self.env.episode.instruction,
                                      perception=self.perception),
                          trace_dir=str(self.episode_trace_dir) if self.episode_trace_dir else None,
                          checkpoints=self.visual_checkpoint_mode)
            if _accepts_node_callbacks(execute):
                kwargs['on_node_start'] = lambda name, ok: self._on_graph_node_start(name, ok, generation)
                kwargs['on_node_end'] = lambda name, ok: self._on_graph_node_end(name, ok, generation)
            result = execute(self.graph_path, **kwargs)
            if generation == self._generation:
                self._result = result
        except BaseException as error:
            if generation == self._generation:
                self._error = error
                self.trace['traceback'] = traceback.format_exc()

    def _register_visual(self, registry, env, bus):
        vision_planning.register(registry, env, bus)

        def observe(hints: dict = None):
            return bus.call(lambda: deepcopy(self.visual.observe(hints)))

        registry.register_callable('vision.observe', observe,
                                   summary='RGB-only calibrated detections, rigid fits and visual goal residuals.',
                                   tags=('vision',))

    def _new_connector(self):
        connector = super()._new_connector()
        connector._tool_registry = RestrictedRegistry(
            connector.tool_registry, self.tool_record_path,
            clock=lambda: float(self.env.step_count/self.env.pipeline.control_hz))
        # Remove alternate observation/evaluator paths from the connector API.
        connector.get_observation = None
        connector.check_success = None
        connector.facade.compute_reward = lambda: 0.
        connector.facade.task_completed = lambda: False

        def world_snapshot():
            from gap.runtime.verify import World
            # Re-read robot-only feedback through the reviewed, bus-safe tools.
            # Checkpoints cannot turn a stage's own success flags into proof.
            registry = connector.tool_registry
            for arm in (0, 1):
                registry.invoke('robot.get_ee_pose', arm_id=arm)
            history = {str(arm): deepcopy(rows) for arm, rows in registry.proprioception_history.items()}
            visual_history = [{key: deepcopy(row[key]) for key in
                               ('poses','timestamp','frame_sha256','assessment','observation_source')
                               if key in row} for row in self.visual.records[-4:]]
            return World(env_id=0, bodies={}, time_s=0., sim_state={
                'vision': deepcopy(self.visual.last or {}),
                'vision_history': visual_history,
                'robot_tcp_poses': {arm: rows[-1]['pose'] for arm, rows in history.items() if rows},
                'robot_tcp_history': history,
            })

        connector.world_snapshot = world_snapshot
        return connector
