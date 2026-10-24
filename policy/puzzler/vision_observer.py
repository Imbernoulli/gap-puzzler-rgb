"""Sensor adapter and observable goal assessment, separate from the evaluator."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from . import model
from .vision import calibration, estimate, rig_specs


def goals(mode='flat', pyramid_plan='pyramid_robot6_plan.json'):
    """Public task specification/CAD. These are desired poses, never observations."""
    if mode == 'pyramid':
        raw = json.loads((model.ROOT/'assets'/pyramid_plan).read_text())
        return {n: (np.array(v['center']), np.array(v['R'])) for n, v in raw['placements'].items()}
    return {n: (model.target_center(n), np.eye(3)) for n in model.CELLS}


def assess(observation, targets):
    """Image-derived goal residuals; missing/uncertain pieces cannot pass."""
    poses = observation['poses']
    errors = {n: None for n in targets}
    if 'board' in poses:
        b = np.array(poses['board'])
        Rb = Rotation.from_quat(b[3:]).as_matrix()
        for n, (center, R) in targets.items():
            if n not in poses:
                continue
            p = np.array(poses[n])
            actual = model.local_beads(n) @ Rotation.from_quat(p[3:]).as_matrix().T+p[:3]
            desired = (model.local_beads(n) @ R.T+center) @ Rb.T+b[:3]
            cost = np.linalg.norm(actual[:, None]-desired[None, :], axis=2)
            ii, jj = linear_sum_assignment(cost)
            errors[n] = float(cost[ii, jj].max())
    return dict(bead_error_m=errors, all_visible=all(n in poses for n in ['board', *targets]),
                goal_geometry_consistent=all(e is not None and e < .004 for e in errors.values()),
                source='RGB pose estimates and public goal CAD; not evaluator success')


class RGBSensor:
    """The only adapter holding a simulator: it outputs RGB, calibration and time."""
    def __init__(self, env):
        self.env = env
        self.specs = rig_specs()
        self.cameras = calibration(self.specs)
        self.renderer = env.make_renderer('tiled', cameras=self.specs, channels=['rgb'])

    def capture(self):
        frames = self.renderer.render(self.env.state)
        q16=np.asarray([float(self.env.state.joint_q[i:i+1].numpy()[0])for i in range(16)],np.float32)
        self.robot_geometry_provenance=dict(step=int(self.env.step_count),timestamp=float(self.env.step_count/self.env.pipeline.control_hz),coordinate_indices=list(range(16)),robot_encoder_values=q16.tolist(),robot_encoder_sha256=hashlib.sha256(q16.tobytes()).hexdigest(),source='Synchronous robot encoder slices only; no object coordinates.')
        if any(set(frame) != {'rgb'} for frame in frames.values()):
            raise ValueError('RGB sensor returned an undeclared channel')
        return ({key: np.asarray(frame['rgb'])[..., :3].copy() for key, frame in frames.items()},
                float(self.env.step_count/self.env.pipeline.control_hz))


class VisualObserver:
    """Does not hold the environment or evaluator; accepts a sensor function."""
    def __init__(self, capture, cameras, targets, output=None, estimator=None):
        self.capture = capture
        self.cameras = cameras
        self.targets = targets
        self.estimator = estimator or estimate
        self.previous = {}
        self.last = None
        self.records = []
        self.output = Path(output) if output else None
        if self.output:
            self.output.mkdir(parents=True, exist_ok=True)

    def observe(self, hints=None):
        images, timestamp = self.capture()
        supplied=dict(hints or {})
        observer_mode=supplied.pop('__observer_mode__',None)
        prefix_tracks=supplied.pop('__conditional_prefix_tracks__',{})
        seeds = {**self.previous, **supplied}
        result = (estimate(images,self.cameras,seeds) if observer_mode in ('plain_rgb_cad','prefix_rgb_cad')
                  else self.estimator(images, self.cameras, seeds))
        if observer_mode == 'released_cyan_rgb_cad':
            from .cyan_released_rgb_recovery import recover_released_cyan
            result = recover_released_cyan(result, images, self.cameras, self.targets['cyan'])
            result['observer_mode'] = observer_mode
        if observer_mode in ('plain_rgb_cad','prefix_rgb_cad'):
            # This observer is appearance-segmented calibrated RGB/CAD. Four
            # independent image views and disjoint fitted tracks are required.
            from scipy.spatial.transform import Rotation
            detections=result.get('detections',{});accepted={}
            for name,detection in detections.items():
                if name=='board' or name not in model.CELLS:continue
                support=np.asarray(detection.get('center_support',[]),dtype=float)
                passed=bool(detection.get('visible') is True and len(detection.get('views',[]))>=4
                    and support.shape==(4,) and np.all(support>=.6)
                    and float(detection.get('median_residual_m',float('inf')))<.0018
                    and float(detection.get('rms_residual_m',float('inf')))<.004)
                if passed:
                    pose=np.asarray(detection['pose']);accepted[name]=model.local_beads(name)@Rotation.from_quat(pose[3:]).as_matrix().T+pose[:3]
                else:detection['visible']=False
            conflicts=set()
            names=list(accepted)
            for i,a in enumerate(names):
                for b in names[i+1:]:
                    if float(np.linalg.norm(accepted[a][:,None]-accepted[b][None,:],axis=2).min())<.020:
                        conflicts.update((a,b))
            for name in conflicts:
                detections[name].update(visible=False,reason='incompatible overlapping current RGB/CAD tracks')
            for name,detection in detections.items():
                if name in model.CELLS and detection.get('visible'):
                    detection['public_cad_seeded']=name in supplied
                    detection['independent_rgb_support']={'fresh_rgb_fit':True,
                        'kind':'four_calibrated_RGB_views_public_appearance_CAD',
                        'center_support':detection.get('center_support',[]),'view_track_disjoint':True}
            result['poses']={n:d['pose'] for n,d in detections.items() if d.get('visible')}
            result['observation_source']='calibrated_rgb_cad_seeded'
            result['observer_mode']=observer_mode
        if observer_mode=='prefix_rgb_cad' and 'board' in result.get('poses',{}):
            # This is a separate conditional tracking witness. It neither
            # changes the plain observer's poses nor creates a unique source.
            from .prefix_track_reassociation import evaluate
            from scipy.spatial.transform import Rotation
            board=np.asarray(result['poses']['board'])
            rb=Rotation.from_quat(board[3:]).as_matrix()
            witnesses={}
            for name,request in prefix_tracks.items():
                if name!='maroon' or name not in self.targets:continue
                center,rotation=self.targets[name]
                goal=np.r_[board[:3]+rb@center,Rotation.from_matrix(rb@rotation).as_quat()]
                witnesses[name]=evaluate(name,images,self.cameras,goal,request['receipt_pose'],
                    request.get('conflicting_poses',()),current_poses=result['poses'])
            result['prefix_track_reassociation']=witnesses
        result['timestamp'] = timestamp
        if hasattr(self.capture,'__self__') and hasattr(self.capture.__self__,'robot_geometry_provenance'):
            result['robot_geometry_provenance']=dict(self.capture.__self__.robot_geometry_provenance)
        result['assessment'] = assess(result, self.targets)
        result['frame_sha256'] = {k: hashlib.sha256(a.tobytes()).hexdigest() for k, a in images.items()}
        for witness in result.get('prefix_track_reassociation',{}).values():
            witness.update(timestamp=result['timestamp'],frame_sha256=dict(result['frame_sha256']),
                           board_pose_at_reassociation=result['poses']['board'])
        # Bind any guarded fallback fit to this exact current sensor epoch.
        # The fitted pose remains the RGB/CAD optimizer output; this record
        # carries the board, pixel hashes, timestamp and synchronous q16/FK
        # provenance needed by downstream bounded-source gates.
        bounded=result.setdefault('bounded_rgb_source', {})
        for name,detection in result.get('detections',{}).items():
            source=detection.get('bounded_rgb_source') if isinstance(detection,dict) else None
            if not source:continue
            source.update(timestamp=result['timestamp'],
                          frame_sha256=dict(result['frame_sha256']),
                          board_pose=result.get('poses',{}).get('board'),
                          target_pose=result.get('poses',{}).get(name),
                          robot_geometry_provenance=result.get('robot_geometry_provenance'))
            bounded[name]=dict(source)
        if self.output:
            # Persist the actual sensor input. No ground truth is written here.
            np.savez_compressed(self.output/f'{len(self.records):04d}.npz', **images)
        self.previous.update(result['poses'])
        self.last = result
        self.records.append(result)
        if self.output:
            (self.output/'observations.json').write_text(json.dumps(self.records, indent=2))
        return result
