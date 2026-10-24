"""Explicit robot-material-only companion for a separate terminal contract.

This v2 helper intentionally makes a narrower claim than the frozen v1 full
geometry API.  The caller MUST independently check the carried object's
uncertainty envelope against the table, board and other objects at exactly
the same dense FK samples, using its existing named terminal-contact rules.
No object retention, released pose measurement, dynamics or physical success
is inferred here.  The frozen PublicBimanualGeometry API is not changed.
"""
from __future__ import annotations

import numpy as np
from puzzler.public_board_clearance import GeometryQueryRefusal, finite_array, validate_quaternion
from .public_bimanual_geometry import PublicBimanualGeometry, PUBLIC_TABLE_TOP_Z


def densify_q16_path(q16_path):
    """Return q16 samples with <=0.012rad revolute and <=1mm finger steps."""
    return PublicBimanualGeometry._dense_path(q16_path)[0]


class PublicBimanualMaterialGeometry(PublicBimanualGeometry):
    densify_q16_path = staticmethod(densify_q16_path)

    @staticmethod
    def _material_samples(q16_path, target_body_poses):
        original = np.asarray(q16_path,dtype=float)
        if original.ndim != 2 or original.shape[1:] != (16,) or len(original)==0:
            raise GeometryQueryRefusal("q16_path must be a nonempty (N,16) array")
        finite_array(original,original.shape,"robot material q16 path")
        steps=np.full(16,.012);steps[[6,7,14,15]]=.001
        already_dense=(len(original)==1 or np.all(np.abs(np.diff(original,axis=0)) <= steps+1e-12))
        q=original.copy() if already_dense else densify_q16_path(original)
        poses=np.asarray(target_body_poses,dtype=float)
        if poses.shape==(7,):
            # A released conditional body hypothesis can remain fixed while
            # empty hands withdraw or return home. This is not a new measure.
            poses=np.tile(poses,(len(q),1))
        elif poses.shape==(len(q),7):
            poses=poses.copy()
        elif poses.shape==(len(original),7) and len(q)!=len(original):
            raise GeometryQueryRefusal(
                "Dynamic target body poses must cover every dense sample; "
                "first call densify_q16_path and compute each body pose from its public FK")
        else:
            raise GeometryQueryRefusal("target_body_poses must be one fixed(7,) pose or exact dense(N,7) poses")
        finite_array(poses,(len(q),7),"target body hypotheses at dense samples")
        for pose in poses:
            validate_quaternion(pose[3:],"target body quaternion")
        return q,poses,len(original)

    def screen_robot_material_path(self,q16_path,scene,target="cyan",target_body_poses=None,
                                   finger_target_contract=None,board_anchor_exit=None):
        """Check robot material only; callers retain their full held guard.

        ``target_body_poses`` is either one fixed conditional release pose or
        the exact per-dense-sample body hypotheses computed by the caller's
        current RGB-owned attachment and public FK. Dynamic body poses are
        never linearly interpolated as a substitute for rigid-body FK.
        """
        q,poses,input_count=self._material_samples(q16_path,target_body_poses)
        raw_board=scene.get("measurement",{}).get("poses",{}).get("board")
        if raw_board is None:
            raw_board=scene.get("board_for_planning",{}).get("pose")
        board=self._pose(raw_board,"current RGB board pose")
        centers,radii,labels=self._scene_spheres(scene,target)
        local,sizes,allowed,normalized=self._target_contract(target,finger_target_contract or {})
        margin=self.minimum_clearance_m
        report=dict(safe=False,passed=False,robot_material_safe=False,
            held_environment_guard_certified=False,requires_caller_held_environment_guard=True,
            scope="Six robot-material groups at supplied dense q16/body hypotheses only; caller must separately pass original held-envelope/terminal environment contract at identical dense samples",
            target_body_pose_source="Caller-supplied RGB-owned public FK or conditional released-body hypothesis; no new pose measurement",
            input_samples=input_count,dense_samples=len(q),samples_checked=0,
            interpolation=dict(max_revolute_step_rad=.012,max_finger_step_m=.001),
            ordinary_clearance_m=margin,intended_distal_penetration_m=.003,
            table_top_world_z=PUBLIC_TABLE_TOP_Z,target_contact_contract=normalized,rows=[])
        anchor=None
        for k,(sample,pose) in enumerate(zip(q,poses)):
            cross=self.cross_arm(sample)
            own=self._same_arm_current()
            table=self._robot_table_current()
            distances=self._robot_sphere_distances(centers,radii)
            i,j=np.unravel_index(int(np.argmin(distances)),distances.shape)
            obstacle=dict(clearance_m=float(distances[i,j]),kind="robot_scene",witness=dict(
                robot_collider=self.shape_labels[int(self.physical_ids[i])],**labels[j]))
            from scipy.spatial.transform import Rotation
            target_centers=Rotation.from_quat(pose[3:]).apply(local)+pose[:3]
            nominal=self._robot_sphere_distances(target_centers,sizes)
            thresholds=np.where(allowed,-.003,margin)
            residual=nominal-thresholds
            i,j=np.unravel_index(int(np.argmin(residual)),residual.shape)
            target_check=dict(clearance_m=float(nominal[i,j]),required_m=float(thresholds[i,j]),
                gate_residual_m=float(residual[i,j]),allowed_selected_distal=bool(allowed[i,j]),
                witness=dict(robot_collider=self.shape_labels[int(self.physical_ids[i])],target_sphere=int(j)))
            board_d,pa,pb,flags=self._board_robot_current(board)
            if k==0:anchor=self._anchor_contract(board_anchor_exit,sample,board,board_d)
            permitted=self._anchor_mask(anchor,board_d)
            masked=board_d.copy();masked[permitted]=np.inf
            a,b=np.unravel_index(int(np.argmin(masked)),masked.shape)
            board_check=dict(clearance_m=float(masked[a,b]),witness=dict(board_mesh_index=int(a),
                board_mesh=str(self._board().paths[a]),robot_collider=self.shape_labels[int(self.physical_ids[b])]),
                original_named_anchor_pairs_exiting=int(np.count_nonzero(permitted)))
            groups=dict(cross_arm=cross,same_arm_nonadjacent=own,moving_robot_table=table,
                robot_non_target=obstacle,robot_target_nominal=target_check,robot_board79=board_check)
            failed=[name for name,value in groups.items() if value.get(
                "gate_residual_m",value["clearance_m"]-margin)<0]
            report["rows"].append(dict(sample=k,groups=groups,failed_groups=failed))
            report["samples_checked"]=k+1
            if failed:
                report["failed_sample"]=k;report["failed_groups"]=failed
                break
        else:
            report["safe"]=True;report["passed"]=True;report["robot_material_safe"]=True
        if anchor is not None:
            report["board_anchor_exit"]=dict(arm=anchor["arm"],maximum_distance_m=anchor["maximum_distance_m"],
                progress_m=anchor["last_progress"],named_pairs=[dict(board_mesh_index=a,
                    robot_collider=self.shape_labels[int(self.physical_ids[b])],exited=(a,b)in anchor["exited"])
                    for a,b in anchor["pairs"]])
        report["clearance_m"]=min(value["clearance_m"] for row in report["rows"] for value in row["groups"].values())
        return report
