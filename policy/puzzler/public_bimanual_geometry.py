"""Public robot-only CUDA FK and material clearance, without an RSS Env.

The caller supplies all 16 robot encoder coordinates.  Only native robot
colliders carrying COLLIDE_SHAPES are queried; visual meshes are excluded.
The model and its scratch FK state are private mathematical objects, never a
live simulation state.  A positive result covers the named geometry only.
"""
from __future__ import annotations

import numpy as np
import newton
import warp as wp
from scipy.spatial.transform import Rotation, Slerp

from newton._src.geometry.types import GeoType
from newton._src.geometry.support_function import GenericShapeData, SupportMapDataProvider
from puzzler.public_board_clearance import (
    GJK, MPR, GeometryQueryRefusal, descriptor, finite_array,
    validate_pair_outputs, validate_quaternion, pair_query,
)
from .public_bimanual_robot import build_public_robot


PUBLIC_TABLE_TOP_Z = 0.750
HELD_ENVELOPE_M = 0.006


@wp.kernel(enable_backward=False)
def _indexed_pair_query(
    shapes: wp.array[GenericShapeData], world: wp.array[wp.transform],
    pairs_a: wp.array[int], pairs_b: wp.array[int], distances: wp.array[float],
    points_a: wp.array[wp.vec3], points_b: wp.array[wp.vec3], flags: wp.array[int],
):
    k = wp.tid()
    a = pairs_a[k]
    b = pairs_b[k]
    frame_a = world[a]
    relative = wp.transform_multiply(wp.transform_inverse(frame_a), world[b])
    rotation = wp.transform_get_rotation(relative)
    position = wp.transform_get_translation(relative)
    provider = SupportMapDataProvider()
    separated, pa, pb, normal, dist = wp.static(GJK.core)(
        shapes[a], shapes[b], rotation, position, 0.0, provider, 96, 1.0e-7, 0.0)
    flag = int(0)
    if not wp.isfinite(dist) or not wp.isfinite(pa) or not wp.isfinite(pb) or not wp.isfinite(normal):
        dist = 0.0
        flag = 3
    elif not separated:
        collision, ma, mb, mn, penetration = wp.static(MPR.core)(
            shapes[a], shapes[b], rotation, position, 0.0, provider, 96, 1.0e-7)
        if not wp.isfinite(penetration) or not wp.isfinite(ma) or not wp.isfinite(mb) or not wp.isfinite(mn):
            dist = 0.0
            flag = 3
        elif collision:
            dist = -wp.abs(penetration)
            pa = ma
            pb = mb
            flag = 1
        else:
            dist = 0.0
            flag = 2
    distances[k] = dist
    points_a[k] = wp.transform_point(frame_a, pa)
    points_b[k] = wp.transform_point(frame_a, pb)
    flags[k] = flag


@wp.kernel(enable_backward=False)
def _robot_spheres(
    shapes: wp.array[GenericShapeData], world: wp.array[wp.transform],
    ids: wp.array[int], centers: wp.array[wp.vec3], radii: wp.array[float],
    distance: wp.array[float],
):
    k = wp.tid()
    i = ids[k // centers.shape[0]]
    j = k % centers.shape[0]
    p = wp.transform_point(wp.transform_inverse(world[i]), centers[j])
    shape = shapes[i]
    d = float(0.0)
    if shape.shape_type == wp.static(int(GeoType.SPHERE)):
        d = wp.length(p) - shape.scale[0]
    elif shape.shape_type == wp.static(int(GeoType.CAPSULE)):
        segment = wp.vec3(0.0, 0.0, wp.clamp(p[2], -shape.scale[1], shape.scale[1]))
        d = wp.length(p - segment) - shape.scale[0]
    elif shape.shape_type == wp.static(int(GeoType.BOX)):
        delta = wp.vec3(wp.abs(p[0]), wp.abs(p[1]), wp.abs(p[2])) - shape.scale
        outside = wp.vec3(wp.max(delta[0], 0.0), wp.max(delta[1], 0.0), wp.max(delta[2], 0.0))
        d = wp.length(outside) + wp.min(wp.max(delta[0], wp.max(delta[1], delta[2])), 0.0)
    distance[k] = d - radii[j]


@wp.kernel(enable_backward=False)
def _robot_plane(
    shapes: wp.array[GenericShapeData], world: wp.array[wp.transform],
    ids: wp.array[int], plane_z: float, distance: wp.array[float],
):
    k = wp.tid()
    i = ids[k]
    shape = shapes[i]
    q = wp.transform_get_rotation(world[i])
    down = wp.quat_rotate_inv(q, wp.vec3(0.0, 0.0, 1.0))
    extent = float(0.0)
    if shape.shape_type == wp.static(int(GeoType.SPHERE)):
        extent = shape.scale[0]
    elif shape.shape_type == wp.static(int(GeoType.CAPSULE)):
        extent = shape.scale[0] + wp.abs(down[2]) * shape.scale[1]
    elif shape.shape_type == wp.static(int(GeoType.BOX)):
        extent = wp.abs(down[0])*shape.scale[0] + wp.abs(down[1])*shape.scale[1] + wp.abs(down[2])*shape.scale[2]
    distance[k] = wp.transform_get_translation(world[i])[2] - extent - plane_z


@wp.kernel(enable_backward=False)
def _sphere_pairs(
    a: wp.array[wp.vec3], ar: wp.array[float], b: wp.array[wp.vec3],
    br: wp.array[float], distance: wp.array[float],
):
    k = wp.tid()
    i = k // b.shape[0]
    j = k % b.shape[0]
    distance[k] = wp.length(a[i]-b[j])-ar[i]-br[j]


@wp.kernel(enable_backward=False)
def _world_colliders(
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    world: wp.array[wp.transform],
):
    s = wp.tid()
    pose = shape_transform[s]
    body = shape_body[s]
    if body >= 0:
        pose = wp.transform_multiply(body_q[body], pose)
    world[s] = pose


@wp.kernel(enable_backward=False)
def _cross_pair_query(
    shapes: wp.array[GenericShapeData],
    world: wp.array[wp.transform],
    left: wp.array[int],
    right: wp.array[int],
    distances: wp.array[float],
    points_a: wp.array[wp.vec3],
    points_b: wp.array[wp.vec3],
    flags: wp.array[int],
):
    k = wp.tid()
    a = left[k // right.shape[0]]
    b = right[k % right.shape[0]]
    frame_a = world[a]
    relative = wp.transform_multiply(wp.transform_inverse(frame_a), world[b])
    rotation = wp.transform_get_rotation(relative)
    position = wp.transform_get_translation(relative)
    provider = SupportMapDataProvider()
    separated, pa, pb, normal, dist = wp.static(GJK.core)(
        shapes[a], shapes[b], rotation, position, 0.0, provider, 96, 1.0e-7, 0.0
    )
    flag = int(0)
    if not wp.isfinite(dist) or not wp.isfinite(pa) or not wp.isfinite(pb) or not wp.isfinite(normal):
        dist = 0.0
        flag = 3
    elif not separated:
        collision, ma, mb, mn, penetration = wp.static(MPR.core)(
            shapes[a], shapes[b], rotation, position, 0.0, provider, 96, 1.0e-7
        )
        if not wp.isfinite(penetration) or not wp.isfinite(ma) or not wp.isfinite(mb) or not wp.isfinite(mn):
            dist = 0.0
            flag = 3
        elif collision:
            dist = -wp.abs(penetration)
            pa = ma
            pb = mb
            flag = 1
        else:
            dist = 0.0
            flag = 2
    distances[k] = dist
    points_a[k] = wp.transform_point(frame_a, pa)
    points_b[k] = wp.transform_point(frame_a, pb)
    flags[k] = flag


class PublicBimanualGeometry:
    """Persistent CUDA robot-only geometry; keep this instance alive.

    ``cross_arm(q16)`` checks every actual left collider against every actual
    right collider.  It does not certify table, scene, held-object, same-arm,
    trajectory interpolation, dynamics, or grasp retention.
    """

    def __init__(self, device="cuda:0", minimum_clearance_m=0.0005):
        self.robot, self.model = build_public_robot(device)
        self.device = self.model.device
        minimum = finite_array(minimum_clearance_m, (), "minimum clearance")
        if float(minimum) < 0.0005:
            raise GeometryQueryRefusal("Cannot weaken the public 0.5mm material clearance gate")
        self.minimum_clearance_m = float(minimum)
        self.scratch = self.model.state()
        self.q = wp.zeros(16, dtype=float, device=self.device)
        self.qd = wp.zeros(self.model.joint_dof_count, dtype=float, device=self.device)
        self.world = wp.empty(self.model.shape_count, dtype=wp.transform, device=self.device)
        self.shape_labels = list(self.model.shape_label)
        self.body_labels = list(self.model.body_label)
        body = np.asarray(self.model.shape_body.numpy(), dtype=int)
        types = np.asarray(self.model.shape_type.numpy(), dtype=int)
        flags = np.asarray(self.model.shape_flags.numpy(), dtype=int)
        scales = self.model.shape_scale.numpy()
        physical = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
        allowed = {int(GeoType.SPHERE), int(GeoType.CAPSULE), int(GeoType.BOX)}
        bad = [int(i) for i in np.flatnonzero(physical) if int(types[i]) not in allowed]
        if bad:
            raise GeometryQueryRefusal("Unsupported physical robot collider types: " + str(bad))
        # Exact path components avoid treating left_link_right_finger as a
        # right-arm shape. Static base geometry is classified by shape label.
        def arm_of(i):
            label = self.shape_labels[i]
            if body[i] >= 0:
                label += "/" + self.body_labels[body[i]]
            left = "/left_arm/" in label
            right = "/right_arm/" in label
            if left == right:
                raise GeometryQueryRefusal("Unclassified/ambiguous physical robot collider: " + label)
            return "left" if left else "right"
        self.physical_ids = np.flatnonzero(physical).astype(np.int32)
        arm = {int(i): arm_of(int(i)) for i in self.physical_ids}
        self.left_ids = np.asarray([i for i in self.physical_ids if arm[int(i)] == "left"], dtype=np.int32)
        self.right_ids = np.asarray([i for i in self.physical_ids if arm[int(i)] == "right"], dtype=np.int32)
        if not len(self.left_ids) or not len(self.right_ids):
            raise GeometryQueryRefusal("Both arms must have physical colliders")
        # Nonphysical entries are never indexed; harmless sphere descriptors
        # keep indexing identical to the native robot model.
        data = [descriptor(GeoType.SPHERE, [0.001]) for _ in types]
        for i in self.physical_ids:
            data[int(i)] = descriptor(int(types[i]), scales[i])
        self.shapes = wp.array(data, dtype=GenericShapeData, device=self.device)
        self.left = wp.array(self.left_ids, dtype=int, device=self.device)
        self.right = wp.array(self.right_ids, dtype=int, device=self.device)
        self.pair_count = len(self.left_ids) * len(self.right_ids)
        self.distances = wp.empty(self.pair_count, dtype=float, device=self.device)
        self.points_a = wp.empty(self.pair_count, dtype=wp.vec3, device=self.device)
        self.points_b = wp.empty(self.pair_count, dtype=wp.vec3, device=self.device)
        self.flags = wp.empty(self.pair_count, dtype=int, device=self.device)
        self.shape_body_host = body
        self.shape_types_host = types
        self.shape_scales_host = scales
        self.shape_arm = arm
        self.physical_wp = wp.array(self.physical_ids, dtype=int, device=self.device)
        self.moving_ids = self.physical_ids[body[self.physical_ids] >= 0]
        self.moving_wp = wp.array(self.moving_ids, dtype=int, device=self.device)
        # The importer intentionally blanket-filters each arm's self contacts.
        # A geometry planner must not inherit that blanket: exclude only shapes
        # on the same body and directly joint-connected bodies. Fixed mounts
        # are the public world parent of their own first link.
        parents = self.model.joint_parent.numpy()
        children = self.model.joint_child.numpy()
        adjacent = {tuple(sorted((int(a), int(b)))) for a, b in zip(parents, children)}
        self.self_pairs = []
        self.self_exclusions = dict(same_body=0, adjacent_body=0)
        for arm_name, ids in (("left", self.left_ids), ("right", self.right_ids)):
            for ii, a in enumerate(ids):
                for b in ids[ii+1:]:
                    ba, bb = int(body[a]), int(body[b])
                    if ba == bb:
                        self.self_exclusions["same_body"] += 1
                    elif tuple(sorted((ba, bb))) in adjacent:
                        self.self_exclusions["adjacent_body"] += 1
                    else:
                        self.self_pairs.append((int(a), int(b)))
        self.self_pairs = np.asarray(self.self_pairs, dtype=np.int32).reshape(-1, 2)
        self.self_a = wp.array(self.self_pairs[:,0], dtype=int, device=self.device)
        self.self_b = wp.array(self.self_pairs[:,1], dtype=int, device=self.device)
        from puzzler.gripper_geometry import Primitive
        kinds = {int(GeoType.SPHERE):"sphere", int(GeoType.CAPSULE):"capsule", int(GeoType.BOX):"box"}
        self.physical_primitives = tuple(Primitive(kinds[int(types[i])], np.zeros(3), np.eye(3),
                                                   scales[i], self.shape_labels[i]) for i in self.physical_ids)
        self._board_query = None
        self._board_robot_buffers = None

    def fk(self, q16):
        """Set only a scratch mathematical FK state; return world transforms."""
        q = finite_array(q16, (16,), "robot encoders q16")
        self.q.assign(q.astype(np.float32))
        newton.eval_fk(self.model, self.q, self.qd, self.scratch)
        wp.launch(_world_colliders, self.model.shape_count,
                  inputs=[self.scratch.body_q, self.model.shape_body,
                          self.model.shape_transform], outputs=[self.world], device=self.device)
        return self.world

    def metadata(self):
        return dict(device=str(self.device), joint_coord_count=16,
                    physical_shapes=len(self.physical_ids),
                    visual_or_noncolliding_excluded=self.model.shape_count-len(self.physical_ids),
                    left_shapes=len(self.left_ids), right_shapes=len(self.right_ids),
                    cross_arm_pairs=self.pair_count,
                    minimum_clearance_m=self.minimum_clearance_m,
                    same_arm_nonadjacent_pairs=len(self.self_pairs),
                    public_self_exclusions=self.self_exclusions,
                    table_top_world_z=PUBLIC_TABLE_TOP_Z,
                    scope="Public native robot colliders and scratch FK only; no Env or scene state")

    def cross_arm(self, q16):
        self.fk(q16)
        wp.launch(_cross_pair_query, self.pair_count,
                  inputs=[self.shapes, self.world, self.left, self.right],
                  outputs=[self.distances, self.points_a, self.points_b, self.flags], device=self.device)
        d, pa, pb, flags = validate_pair_outputs(
            self.distances.numpy(), self.points_a.numpy(), self.points_b.numpy(),
            self.flags.numpy(), self.pair_count)
        k = int(np.argmin(d))
        a = int(self.left_ids[k // len(self.right_ids)])
        b = int(self.right_ids[k % len(self.right_ids)])
        return dict(clearance_m=float(d[k]),
                    safe=bool(d[k] >= self.minimum_clearance_m),
                    minimum_clearance_m=self.minimum_clearance_m,
                    pairs_checked=self.pair_count,
                    witness=dict(left_shape_id=a, right_shape_id=b,
                                 left_collider=self.shape_labels[a],
                                 right_collider=self.shape_labels[b],
                                 point_left_world=pa[k].tolist(),
                                 point_right_world=pb[k].tolist(),
                                 overlap_flag=int(flags[k])),
                    scope="All actual left-vs-right robot material colliders at supplied q16")

    def screen_cross_arm_path(self, q16_path):
        """Check supplied samples; never claim continuous interpolation safety."""
        q = np.asarray(q16_path, dtype=float)
        if q.ndim != 2 or q.shape[1] != 16 or len(q) == 0:
            raise GeometryQueryRefusal("q16_path must be a nonempty (N,16) array")
        finite_array(q, q.shape, "robot encoder path")
        rows = [self.cross_arm(value) for value in q]
        k = int(np.argmin([r["clearance_m"] for r in rows]))
        return dict(safe=all(r["safe"] for r in rows), clearance_m=rows[k]["clearance_m"],
                    worst_sample=k, samples=len(rows), witness=rows[k]["witness"],
                    rows=rows, scope="Supplied discrete q16 samples, cross-arm material only")

    def robot_material(self, q16):
        """Robot-only material groups at one q16: cross-arm, same-arm, table.

        Uses the unchanged public colliders and adjacency exclusions; it says
        nothing about scene, held object, board or interpolation between samples.
        """
        cross = self.cross_arm(q16)  # evaluates FK for the groups below
        return dict(cross_arm=cross, same_arm_nonadjacent=self._same_arm_current(),
                    moving_robot_table=self._robot_table_current())

    @staticmethod
    def _arm(arm):
        if arm in (0, "0", "left"):
            return "left"
        if arm in (1, "1", "right"):
            return "right"
        raise GeometryQueryRefusal("Unknown arm: "+str(arm))

    @staticmethod
    def _pose(pose, label):
        p = finite_array(pose, (7,), label)
        validate_quaternion(p[3:], label+" quaternion")
        return p

    @staticmethod
    def _quat(q, label):
        if isinstance(q, dict):
            q = [q[k] for k in ("x", "y", "z", "w")]
        return validate_quaternion(q, label)

    def _board(self):
        if self._board_query is None:
            from puzzler.public_board_clearance import PublicBoardQuery
            self._board_query = PublicBoardQuery()
            if self._board_query.device != self.device:
                raise GeometryQueryRefusal("Robot and board query CUDA devices differ")
        return self._board_query

    def _same_arm_current(self):
        n = len(self.self_pairs)
        d = wp.empty(n, dtype=float, device=self.device)
        pa = wp.empty(n, dtype=wp.vec3, device=self.device)
        pb = wp.empty(n, dtype=wp.vec3, device=self.device)
        flags = wp.empty(n, dtype=int, device=self.device)
        wp.launch(_indexed_pair_query, n, inputs=[self.shapes, self.world, self.self_a, self.self_b],
                  outputs=[d, pa, pb, flags], device=self.device)
        d, pa, pb, flags = validate_pair_outputs(d.numpy(), pa.numpy(), pb.numpy(), flags.numpy(), n)
        k = int(np.argmin(d)); a, b = self.self_pairs[k]
        return dict(clearance_m=float(d[k]), pairs_checked=n,
                    witness=dict(collider_a=self.shape_labels[a], collider_b=self.shape_labels[b],
                                 point_a_world=pa[k].tolist(), point_b_world=pb[k].tolist(),
                                 overlap_flag=int(flags[k])), exclusions=self.self_exclusions)

    def _robot_sphere_distances(self, centers, radii):
        centers = finite_array(centers, (len(centers),3), "sphere centers")
        radii = finite_array(radii, (len(centers),), "sphere radii")
        if not len(centers) or np.any(radii <= 0):
            raise GeometryQueryRefusal("Sphere query requires positive radii and nonempty geometry")
        n = len(self.physical_ids)*len(centers)
        out = wp.empty(n, dtype=float, device=self.device)
        wp.launch(_robot_spheres, n,
                  inputs=[self.shapes, self.world, self.physical_wp,
                          wp.array(centers, dtype=wp.vec3, device=self.device),
                          wp.array(radii, dtype=float, device=self.device)], outputs=[out], device=self.device)
        return finite_array(out.numpy(), (n,), "all robot/sphere distances").reshape(len(self.physical_ids), -1)

    def _robot_table_current(self):
        out = wp.empty(len(self.moving_ids), dtype=float, device=self.device)
        wp.launch(_robot_plane, len(self.moving_ids),
                  inputs=[self.shapes, self.world, self.moving_wp, PUBLIC_TABLE_TOP_Z],
                  outputs=[out], device=self.device)
        d = finite_array(out.numpy(), (len(self.moving_ids),), "moving-link/table clearance")
        k = int(np.argmin(d)); i = int(self.moving_ids[k])
        return dict(clearance_m=float(d[k]), witness=dict(robot_collider=self.shape_labels[i],
                    table_top_z=PUBLIC_TABLE_TOP_Z),
                    scope="All articulated colliders vs public 0.750m table plane; fixed mount shapes excluded")

    def _board_robot_current(self, board_pose):
        q = self._board()
        world = self.world.numpy()[self.physical_ids]
        rb = Rotation.from_quat(board_pose[3:])
        position = rb.inv().apply(world[:,:3]-board_pose[:3])
        rotation = (rb.inv()*Rotation.from_quat(world[:,3:])).as_quat()
        n = 79*len(self.physical_ids)
        # Native robot collider kinds/sizes never change after model build.
        # Reuse only scratch storage; every pose and every pair is recomputed.
        if self._board_robot_buffers is None:
            kinds = {"sphere": GeoType.SPHERE, "capsule": GeoType.CAPSULE, "box": GeoType.BOX}
            self._board_robot_buffers = (
                wp.array([descriptor(kinds[s.kind], s.size) for s in self.physical_primitives],
                         dtype=GenericShapeData, device=self.device),
                wp.empty(len(self.physical_ids), dtype=wp.vec3, device=self.device),
                wp.empty(len(self.physical_ids), dtype=wp.quat, device=self.device),
                wp.empty(n, dtype=float, device=self.device),
                wp.empty(n, dtype=wp.vec3, device=self.device),
                wp.empty(n, dtype=wp.vec3, device=self.device),
                wp.empty(n, dtype=int, device=self.device),
            )
        hand, positions, rotations, distances, points_a, points_b, pair_flags = self._board_robot_buffers
        positions.assign(np.asarray(position, dtype=np.float32))
        rotations.assign(np.asarray(rotation, dtype=np.float32))
        wp.launch(pair_query, n, inputs=[q.board, hand, positions, rotations],
                  outputs=[distances, points_a, points_b, pair_flags], device=self.device)
        d, pa, pb, flags = validate_pair_outputs(
            distances.numpy(), points_a.numpy(), points_b.numpy(), pair_flags.numpy(), n)
        return d.reshape(79,-1), pa, pb, flags

    def _sphere_board(self, board_pose, centers, radii):
        from puzzler.gripper_geometry import Primitive
        rb = Rotation.from_quat(board_pose[3:])
        primitives = tuple(Primitive("sphere", np.zeros(3), np.eye(3), np.asarray([r]),
                                     "held-envelope-sphere-"+str(i)) for i,r in enumerate(radii))
        local = rb.inv().apply(centers-board_pose[:3])
        quats = np.tile([0.,0.,0.,1.], (len(centers),1))
        result = self._board().query_shapes(primitives, local, quats)
        return result

    def _sphere_pair_distances(self, a, ar, b, br):
        n = len(a)*len(b)
        out = wp.empty(n, dtype=float, device=self.device)
        wp.launch(_sphere_pairs, n, inputs=[wp.array(a,dtype=wp.vec3,device=self.device),
                  wp.array(ar,dtype=float,device=self.device),wp.array(b,dtype=wp.vec3,device=self.device),
                  wp.array(br,dtype=float,device=self.device)], outputs=[out], device=self.device)
        return finite_array(out.numpy(), (n,), "held envelope/scene sphere distances").reshape(len(a),len(b))

    def _scene_spheres(self, scene, target):
        from puzzler import model as pm
        from puzzler.conditional_scene import local_sphere_cover
        if not isinstance(scene, dict) or not isinstance(scene.get("occupancy"), dict):
            raise GeometryQueryRefusal("Attributed conditional/RGB scene occupancy is required")
        expected = set(pm.CELLS)-{target}
        missing = expected-set(scene["occupancy"])
        if missing:
            raise GeometryQueryRefusal("Unknown objects cannot become free space: "+str(sorted(missing)))
        centers, radii, labels = [], [], []
        for name, entry in scene["occupancy"].items():
            if name == target:
                continue
            if name not in pm.CELLS or not entry.get("source"):
                raise GeometryQueryRefusal("Unknown/unattributed scene object: "+name)
            margin = float(finite_array(entry.get("inflation_m"), (), "scene inflation"))
            if margin < 0:
                raise GeometryQueryRefusal("Scene inflation must be nonnegative")
            hypotheses = entry.get("pose_hypotheses")
            if not isinstance(hypotheses, list) or not hypotheses:
                raise GeometryQueryRefusal("Scene object has no pose hypotheses: "+name)
            local, sizes = local_sphere_cover(name)
            for h, value in enumerate(hypotheses):
                pose = self._pose(value, name+" occupancy pose")
                centers.extend(Rotation.from_quat(pose[3:]).apply(local)+pose[:3])
                radii.extend(sizes+margin)
                labels.extend([dict(object=name, hypothesis=h, sphere=i,
                                    source=entry["source"], inflation_m=margin) for i in range(len(local))])
        return np.asarray(centers), np.asarray(radii), labels

    @staticmethod
    def _dense_path(q16_path):
        q = np.asarray(q16_path, dtype=float)
        if q.ndim != 2 or q.shape[1:] != (16,) or len(q) == 0:
            raise GeometryQueryRefusal("q16_path must be a nonempty (N,16) array")
        finite_array(q, q.shape, "q16 path")
        values, times = [q[0]], [0.]
        steps = np.full(16, .012)
        steps[[6,7,14,15]] = .001
        for i, (a,b) in enumerate(zip(q, q[1:])):
            count = max(1, int(np.ceil(np.max(np.abs(b-a)/steps))))
            if len(values)+count > 10000:
                raise GeometryQueryRefusal("Path interpolation exceeds 10000 samples")
            for j in range(1,count+1):
                u = j/count
                values.append((1-u)*a+u*b); times.append(i+u)
        return np.asarray(values), np.asarray(times), q

    def _target_contract(self, target, contract):
        from puzzler import model as pm
        from puzzler.conditional_scene import local_sphere_cover
        if target not in pm.CELLS:
            raise GeometryQueryRefusal("Unknown public target: "+str(target))
        beads = pm.local_beads(target)
        local, sizes = local_sphere_cover(target)
        allowed = np.zeros((len(self.physical_ids),len(local)), dtype=bool)
        normalized = {}
        if not isinstance(contract, dict):
            raise GeometryQueryRefusal("finger_target_contract must be an arm-to-bead-list dict")
        for arm_id, indices in contract.items():
            arm = self._arm(arm_id)
            if not isinstance(indices, (tuple,list,np.ndarray)) or not len(indices):
                raise GeometryQueryRefusal("Intended contact requires explicit selected bead indices")
            normalized[arm] = []
            for index in indices:
                if isinstance(index,bool) or int(index) != index or not 0 <= int(index) < len(beads):
                    raise GeometryQueryRefusal("Invalid selected bead index")
                index = int(index); normalized[arm].append(index)
                contained = np.linalg.norm(local-beads[index],axis=1)+sizes <= pm.RADIUS+1e-8
                for j, shape in enumerate(self.physical_ids):
                    label = self.shape_labels[shape]
                    distal = ("_lf_down/" in label or "_rf_down/" in label)
                    if self.shape_arm[int(shape)] == arm and distal:
                        allowed[j] |= contained
        return local, sizes, allowed, normalized

    def _anchor_contract(self, spec, q0, board_pose, first_board_distances):
        if spec is None:
            return None
        if not isinstance(spec,dict) or not isinstance(spec.get("source_anchor"),dict):
            raise GeometryQueryRefusal("Board anchor exit requires original source_anchor evidence")
        source = spec["source_anchor"]
        arm = self._arm(spec.get("arm"))
        if self._arm(source.get("arm",source.get("arm_id"))) != arm:
            raise GeometryQueryRefusal("Anchor arm contradicts original source evidence")
        source_board = source.get("source_board_rgb",{}).get("poses",{}).get("board")
        if source_board is None:
            source_board = source.get("input_dependencies",{}).get("board_current_RGB_pose")
        source_board = self._pose(source_board,"anchor source RGB board")
        if np.linalg.norm(source_board[:3]-board_pose[:3]) > .0015 or (
                Rotation.from_quat(source_board[3:]).inv()*Rotation.from_quat(board_pose[3:])).magnitude() > .01:
            raise GeometryQueryRefusal("Anchor source RGB board contradicts current board")
        actual_p = source.get("actual_tcp",source.get("actual_TCP"))
        actual_q = source.get("actual_rotation")
        p = finite_array(actual_p,(3,),"anchor source actual TCP")
        r = self._quat(actual_q,"anchor source actual rotation")
        now_p, now_r = self.robot.iks[arm].ee_pose(self.scratch)
        if np.linalg.norm(now_p-p) > .002 or (Rotation.from_quat(now_r).inv()*Rotation.from_quat(r)).magnitude() > .03:
            raise GeometryQueryRefusal("Path start FK does not match original actual anchor pose")
        direction = finite_array(spec.get("direction_world"),(3,),"anchor exit direction")
        if abs(np.linalg.norm(direction)-1.) > 1e-6 or np.linalg.norm(direction-[0,0,1]) > 1e-6:
            raise GeometryQueryRefusal("Anchor exit supports only declared world +Z withdrawal")
        maximum = float(finite_array(spec.get("maximum_distance_m"),(),"anchor exit distance"))
        if not 0 < maximum <= .0600001:
            raise GeometryQueryRefusal("Anchor withdrawal distance must be in (0,0.060]m")
        # Attribution is deliberately narrow: authentic board-edge brace,
        # actual original distal fingers, four public rim components only.
        contact_text = str(source.get("intended_contact",source.get("profile",{}).get("intentional_contact",""))).lower()
        if "board" not in contact_text or not ("brace" in contact_text or "support" in contact_text):
            raise GeometryQueryRefusal("Source evidence does not declare a board brace")
        allowed_names = spec.get("allowed_robot_shape_names")
        allowed_meshes = spec.get("allowed_board_mesh_indices",[73,74,75,76])
        if not set(allowed_meshes).issubset({73,74,75,76}):
            raise GeometryQueryRefusal("Anchor contact permission is limited to public board rim components")
        pairs = []
        for mesh in allowed_meshes:
            for j, shape in enumerate(self.physical_ids):
                label = self.shape_labels[shape]
                if (self.shape_arm[int(shape)] == arm and ("_lf_down/" in label or "_rf_down/" in label)
                        and (allowed_names is None or label in allowed_names)
                        and first_board_distances[mesh,j] < self.minimum_clearance_m):
                    pairs.append((int(mesh),j))
        return dict(arm=arm, initial_tcp=np.asarray(now_p),initial_rotation=np.asarray(now_r),
                    direction=direction,maximum_distance_m=maximum,pairs=pairs,
                    previous={pair:float(first_board_distances[pair]) for pair in pairs},
                    exited=set(),last_progress=0.)

    def _anchor_mask(self, anchor, board_d):
        mask = np.zeros(board_d.shape,dtype=bool)
        if anchor is None:
            return mask
        p, r = self.robot.iks[anchor["arm"]].ee_pose(self.scratch)
        delta = p-anchor["initial_tcp"]
        progress = float(delta@anchor["direction"])
        lateral = np.linalg.norm(delta-progress*anchor["direction"])
        angle = (Rotation.from_quat(anchor["initial_rotation"]).inv()*Rotation.from_quat(r)).magnitude()
        if (progress < anchor["last_progress"]-2e-5 or progress < -2e-5 or
                progress > anchor["maximum_distance_m"]+.002 or lateral > .002 or angle > .03):
            raise GeometryQueryRefusal("Original anchor exit is not a monotone bounded vertical withdrawal")
        anchor["last_progress"] = progress
        for pair in anchor["pairs"]:
            value = float(board_d[pair])
            if value >= self.minimum_clearance_m:
                anchor["exited"].add(pair)
            elif pair not in anchor["exited"]:
                if value < anchor["previous"][pair]-5e-6:
                    raise GeometryQueryRefusal("Named original anchor material contact became deeper during exit")
                mask[pair] = True
            anchor["previous"][pair] = value
        return mask

    def screen_q16_path(self, q16_path, scene, moving_arm, held_arm,
                        held_body_pose, object_in_tcp_position, object_in_tcp_rotation,
                        target="cyan", finger_target_contract=None, board_anchor_exit=None,
                        held_envelope_m=HELD_ENVELOPE_M):
        """Screen a supplied encoder path using RGB-owned held attachment.

        All revolute increments are interpolated to <=0.012rad and fingers to
        <=1mm. Target poses are recomputed from public FK at every sample.
        Results certify sampled geometry only, never retention or dynamics.
        """
        self._arm(moving_arm)
        owner = self._arm(held_arm)
        initial = self._pose(held_body_pose,"initial held body pose")
        offset = finite_array(object_in_tcp_position,(3,),"object in TCP position")
        rotation = np.asarray(object_in_tcp_rotation,dtype=float)
        if rotation.shape == (4,):
            rotation = Rotation.from_quat(validate_quaternion(rotation,"object in TCP quaternion")).as_matrix()
        from puzzler.public_board_clearance import validate_rotation
        rotation = validate_rotation(rotation,"object in TCP rotation")
        q, times, original = self._dense_path(q16_path)
        poses=[]
        for sample in q:
            self.fk(sample)
            p, r = self.robot.iks[owner].ee_pose(self.scratch)
            rr = Rotation.from_quat(r)
            poses.append(np.r_[p+rr.apply(offset),(rr*Rotation.from_matrix(rotation)).as_quat()])
        first = np.asarray(poses[0])
        if (np.linalg.norm(first[:3]-initial[:3]) > .006 or
                (Rotation.from_quat(first[3:]).inv()*Rotation.from_quat(initial[3:])).magnitude() > .10):
            raise GeometryQueryRefusal("Held attachment FK contradicts supplied current body belief")
        board = scene.get("measurement",{}).get("poses",{}).get("board")
        if board is None:
            board = scene.get("board_for_planning",{}).get("pose")
        return self._screen_dense(q, scene=scene, board_pose=board, target=target,
                                  target_poses=np.asarray(poses), contract=finger_target_contract or {},
                                  clearance_m=self.minimum_clearance_m,
                                  target_penetration_m=.003, board_anchor_exit=board_anchor_exit,
                                  input_samples=len(original),held_envelope_m=held_envelope_m)

    def screen_path(self, q16_path, *, scene, board_pose, target_name, target_body_poses,
                    allowed_target_distal_arms, target_selected_bead_by_arm,
                    clearance_m=.0005,target_finger_penetration_m=.003,board_anchor_exit=None,
                    held_envelope_m=HELD_ENVELOPE_M):
        """Compatibility form with explicit body poses corresponding to q16."""
        q, times, original = self._dense_path(q16_path)
        poses=finite_array(target_body_poses,(len(original),7),"target body poses")
        for pose in poses:
            validate_quaternion(pose[3:],"target body quaternion")
        if len(original)==1:
            dense=poses.copy()
        else:
            rotations=Slerp(np.arange(len(original)),Rotation.from_quat(poses[:,3:]))(times).as_quat()
            xyz=np.stack([np.interp(times,np.arange(len(original)),poses[:,j]) for j in range(3)],axis=1)
            dense=np.column_stack((xyz,rotations))
        allowed={self._arm(arm) for arm in allowed_target_distal_arms}
        contract={arm:indices for arm,indices in target_selected_bead_by_arm.items() if self._arm(arm) in allowed}
        if {self._arm(arm) for arm in target_selected_bead_by_arm} != allowed:
            raise GeometryQueryRefusal("Allowed distal arms and selected bead contract disagree")
        return self._screen_dense(q,scene=scene,board_pose=board_pose,target=target_name,target_poses=dense,
                                  contract=contract,clearance_m=clearance_m,
                                  target_penetration_m=target_finger_penetration_m,
                                  board_anchor_exit=board_anchor_exit,input_samples=len(original),
                                  held_envelope_m=held_envelope_m)

    def _screen_dense(self, q, *, scene, board_pose, target, target_poses,contract,
                      clearance_m,target_penetration_m,board_anchor_exit,input_samples,
                      held_envelope_m=HELD_ENVELOPE_M):
        board = self._pose(board_pose,"current RGB board pose")
        margin=float(finite_array(clearance_m,(),"clearance margin"))
        penetration=float(finite_array(target_penetration_m,(),"target material penetration"))
        held_bound=float(finite_array(held_envelope_m,(),"held object envelope"))
        if margin < .0005 or not 0 <= penetration <= .003:
            raise GeometryQueryRefusal("Cannot weaken 0.5mm ordinary clearance or 3mm intended contact gates")
        if held_bound < HELD_ENVELOPE_M:
            raise GeometryQueryRefusal("Cannot weaken the minimum 6mm held object envelope")
        centers,radii,labels=self._scene_spheres(scene,target)
        local,sizes,allowed,normalized=self._target_contract(target,contract)
        report=dict(safe=False,passed=False,input_samples=input_samples,samples_checked=0,
            interpolation=dict(max_revolute_step_rad=.012,max_finger_step_m=.001),
            ordinary_clearance_m=margin,intended_distal_penetration_m=penetration,
            held_envelope_m=held_bound,table_top_world_z=PUBLIC_TABLE_TOP_Z,
            target_contact_contract=normalized,
            envelope_scope="Declared held_envelope_m (minimum6mm) target sphere-cover inflation vs table, all79 board and all non-target occupancy; nominal target vs actual robot, selected bead-contained material vs named distal fingers only",
            scope="Public robot-only scratch FK and supplied RGB/conditional geometry; dense sampled checks, no dynamics/retention/continuous-path proof",rows=[])
        anchor=None
        def summary(d,kind,extra=None):
            i,j=np.unravel_index(int(np.argmin(d)),d.shape)
            w=dict(robot_collider=self.shape_labels[int(self.physical_ids[i])],sphere=int(j))
            if extra is not None:w.update(extra[j])
            return dict(clearance_m=float(d[i,j]),witness=w,kind=kind)
        for k,(sample,pose) in enumerate(zip(q,target_poses)):
            cross=self.cross_arm(sample)
            own=self._same_arm_current()
            table=self._robot_table_current()
            obstacle=summary(self._robot_sphere_distances(centers,radii),"robot_scene",labels)
            target_centers=Rotation.from_quat(pose[3:]).apply(local)+pose[:3]
            nominal=self._robot_sphere_distances(target_centers,sizes)
            thresholds=np.where(allowed,-penetration,margin)
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
            env_sizes=sizes+held_bound
            held_table_d=target_centers[:,2]-env_sizes-PUBLIC_TABLE_TOP_Z
            held_table=dict(clearance_m=float(held_table_d.min()),witness=dict(target_sphere=int(np.argmin(held_table_d))))
            held_board=self._sphere_board(board,target_centers,env_sizes)
            held_pieces_d=self._sphere_pair_distances(target_centers,env_sizes,centers,radii)
            a,b=np.unravel_index(int(np.argmin(held_pieces_d)),held_pieces_d.shape)
            held_pieces=dict(clearance_m=float(held_pieces_d[a,b]),witness=dict(target_sphere=int(a),**labels[b]))
            groups=dict(cross_arm=cross,same_arm_nonadjacent=own,moving_robot_table=table,
                robot_non_target=obstacle,robot_target_nominal=target_check,robot_board79=board_check,
                held_envelope_table=held_table,held_envelope_board79=held_board,held_envelope_non_target=held_pieces)
            failed=[name for name,value in groups.items() if (
                value.get("gate_residual_m",value["clearance_m"]-margin)<0)]
            row=dict(sample=k,groups=groups,failed_groups=failed)
            report["rows"].append(row);report["samples_checked"]=k+1
            if failed:
                report["failed_sample"]=k;report["failed_groups"]=failed
                break
        else:
            report["safe"]=True;report["passed"]=True
        if anchor is not None:
            report["board_anchor_exit"]=dict(arm=anchor["arm"],maximum_distance_m=anchor["maximum_distance_m"],
                progress_m=anchor["last_progress"],named_pairs=[dict(board_mesh_index=a,
                    robot_collider=self.shape_labels[int(self.physical_ids[b])],exited=(a,b)in anchor["exited"])
                    for a,b in anchor["pairs"]])
        report["clearance_m"]=min(value["clearance_m"] for row in report["rows"] for value in row["groups"].values())
        return report
