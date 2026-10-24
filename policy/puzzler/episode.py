import argparse
from dataclasses import asdict,replace
import hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
from . import task


def capture_development_physics(env):
    """Owner-thread geometry helper for compatibility exports, never a tool."""
    poses={name:np.asarray(pose,dtype=float).tolist() for name,pose in task.poses(env).items()}
    velocities={name:np.asarray(env[name].velocity,dtype=float).tolist() for name in poses}
    hands=[]
    for arm in ('left','right'):
        position,quaternion=env.arm(arm).ee_pose(env.state)
        hands.append([*map(float,position),*map(float,quaternion),float(env.gripper(arm).width(env.state))])
    if any(not np.isfinite(value).all() for value in [*poses.values(),*velocities.values(),*hands]):
        raise ValueError('Nonfinite evaluator geometry')
    return dict(step=int(env.step_count),time_s=float(env.step_count/env.pipeline.control_hz),
                poses=poses,hands=hands,body_velocity=velocities)


class NativeCheckpointIndexRecorder:
    """Index native canonical capture steps without additional simulator reads."""
    def __init__(self,out):
        self.directory=Path(out)/'development'
        self.path=self.directory/'native_checkpoint_indices.jsonl'
        self.meta_path=self.directory/'native_checkpoint_indices.meta.json'
        self.rows=[];self.errors=[];self.rollout_returned=False

    def record(self,boundary_step,*,generation,sensor):
        row=dict(schema='puzzler.native_checkpoint_index.v1',sequence=len(self.rows),
                 policy_input=False,generation=generation,sensor_provenance=sensor)
        try:
            row.update(status='ok',boundary_step=int(boundary_step()))
            json.dumps(row,allow_nan=False)
        except Exception as error:
            row.update(status='error',error=f'{type(error).__name__}: {error}')
        self.rows.append(row)
        try:
            self.directory.mkdir(parents=True,exist_ok=True)
            with self.path.open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            self.meta_path.write_text(json.dumps(self.metadata(),indent=2))
        except Exception as error:
            self.errors.append(f'{type(error).__name__}: {error}')

    def metadata(self):
        return dict(schema='puzzler.native_checkpoint_index_metadata.v1',policy_input=False,
                    rows=len(self.rows),capture_errors=sum(row['status']!='ok' for row in self.rows),
                    persistence_errors=list(self.errors),rollout_returned=self.rollout_returned,
                    additional_bus_calls=0,object_truth_read=False,
                    scope='Native checkpoint callback canonical-step reference; not an exact continuous wall-time instant or RGB capture-time claim.',
                    stage_entry_capture=False,error_exit_capture=False)

    def finalize(self):
        self.rollout_returned=True
        try:
            self.directory.mkdir(parents=True,exist_ok=True)
            self.meta_path.write_text(json.dumps(self.metadata(),indent=2))
        except Exception as error:
            self.errors.append(f'{type(error).__name__}: {error}')
        return dict(native_checkpoint_indices='development/native_checkpoint_indices.jsonl',
                    native_checkpoint_indices_metadata='development/native_checkpoint_indices.meta.json')


def wrap_native_checkpoint_index(snapshot,recorder,*,boundary_step,generation):
    """Call the original native World supplier once; record only its step index."""
    def indexed_snapshot():
        world=snapshot()
        visual=getattr(world,'sim_state',{}).get('vision',{})
        recorder.record(boundary_step,generation=generation(),
                        sensor=dict(timestamp=visual.get('timestamp'),frame_sha256=visual.get('frame_sha256')))
        return world
    return indexed_snapshot


def snapshot_sources(root,out,paths):
    """Save the exact project bundle before a rollout can be interrupted."""
    root=Path(root).resolve();out=Path(out)
    hashes={}
    for path in sorted(set(map(Path,paths))):
        relative=path.resolve().relative_to(root)
        content=path.read_bytes()
        hashes[str(relative)]=hashlib.sha256(content).hexdigest()
        target=out/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    return hashes


def changed_sources(root,hashes):
    """Detect edits/deletions; this is provenance, never a policy observation."""
    root=Path(root)
    return [relative for relative,expected in hashes.items()
            if not (root/relative).is_file() or
            hashlib.sha256((root/relative).read_bytes()).hexdigest()!=expected]


def write_development_spans(out,rows,*,capture_step,control_hz):
    """Persist native RSS lifecycle evidence outside policy observations.

    Rows must be captured at rollout end, before cleanup can turn an inflight
    tool into a shutdown error. RSS already bounds/serializes requests and
    results; preserve its schema and interrupted markers without interpretation.
    This is not a native full-state capture tape or a contact recorder.
    """
    d=Path(out)/'development';d.mkdir(parents=True,exist_ok=True)
    (d/'rss_tool_spans.jsonl').write_text(''.join(json.dumps(row,default=str)+'\n' for row in rows))
    metadata=dict(schema='puzzler.development_tool_spans.v1',
                  source_api='robosimstudio_policy_agent.SolutionGraph.raw_spans',
                  policy_input=False,capture_step=int(capture_step),control_hz=float(control_hz),
                  row_count=len(rows),scope='Native RSS sim_step-tagged tool lifecycle metadata only',
                  full_native_state_tape=False,contacts_recorded=False,
                  state_samples=dict(path='../frames.json',interval_steps=5,
                                     exact_endpoint_alignment_guaranteed=False))
    (d/'rss_tool_spans.meta.json').write_text(json.dumps(metadata,indent=2))
    return dict(rss_tool_spans='development/rss_tool_spans.jsonl',
                rss_tool_spans_manifest='development/rss_tool_spans.meta.json')

