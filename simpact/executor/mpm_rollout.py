"""MPM dough/sand rollout — the deformable analogue of ``MuJoCoRollout`` for MPM.

Wraps ``simulators.mpm.MPM_Simulator_WARP``: loads a trial's particle cloud, applies a
plan to the initial EE pose to place two closing gripper jaws (dough) or a pusher
(sweep), time-steps the material with ``p2g2p``, renders before/after from the real
camera1 viewpoint, and writes the same rollout-JSON envelope the verifier/optimizer read
(``snapshots[*].{screenshot, objects, gripper}`` + ``object_names``) plus an MPM block
(``final_points_path`` + bounding box) for ``parse_mpm_rollout``.

The dough rollout is **multi-grasp by default** (the original ``shape_kinetic_sand_multi_step.py``):
``MPMRollout`` applies a *list* of squeezes (``grasps_from_plan``) in ONE continuous sim
— a single squeeze is simply the degenerate N=1 case (there is no separate single-squeeze
path). Grounded on the original pipeline (verified): a grid-centring shift (``_grid_shift``, replacing the original pipeline's
fixed ``+[0, 0.5, 0]`` ``# TODO: fix grid bounds`` hack) applied on load and undone on
save; the two jaw colliders added once then repositioned per grasp via
``set_collision_params``; a short jaw box, and every grasp at a fixed ``grasp_height``.

Accepts BOTH plan schemas (docs/DEFORMABLE_INTEGRATION.md §7): propose primitives
(``PUSH``/``ROTATE``/``DESCEND``/``GRASP``) and the optimizer's ``move``/
``gripper_control`` (one ``move``+``gripper_control`` pair per squeeze). Needs CUDA + warp.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from simpact.executor.render_deformable import render_deformable, VideoRecorder, IMG_W, IMG_H

# MPM grid: domain [0, GRID_LIM]^3 with GRID_N cells (dx = GRID_LIM/GRID_N = 0.01 m).
GRID_LIM = 1.0
GRID_N = 100


def _grid_shift(init_pts):
    """Translation that centres the robot-frame cloud in the MPM grid [0, GRID_LIM]^3.

    Replaces the old fixed ``[0, 0.5, 0]`` hack, which silently pushed particles outside
    the grid whenever the object's world coords left ``[0, GRID_LIM]`` (an audit finding
    §C). MPM on a uniform grid with uniform gravity is translation-invariant, so centring
    is behaviour-preserving — it just guarantees the cloud fits for *any* workspace, and
    raises loudly instead of failing silently if the object is too large for the grid.
    """
    pts = np.asarray(init_pts, dtype=float)
    dx = GRID_LIM / GRID_N
    # Snap to a whole number of grid cells so particles keep their sub-cell alignment
    # (the old [0,0.5,0] shift was 50 cells) -> MPM result is bit-for-bit unchanged.
    shift = np.round((GRID_LIM / 2.0 - pts.mean(axis=0)) / dx) * dx
    lo, hi = (pts + shift).min(0), (pts + shift).max(0)
    if lo.min() < 0.1 or hi.max() > GRID_LIM - 0.1:
        raise ValueError(
            f"object does not fit the MPM grid after centring: bounds "
            f"[{np.round(lo,3).tolist()}, {np.round(hi,3).tolist()}] outside "
            f"[0.1, {GRID_LIM-0.1}] — increase grid_lim.")
    return shift


# Gripper-jaw geometry in the EE local frame (from the original dough executor).
JAW_LOCAL = np.array([0.0, 0.01773615, 0.07102775])  # |y| offset + z drop of each jaw
JAW_BOX = np.array([0.06, 0.02, 0.15])               # short jaw box (rendered + sim)
DOUGH_TABLE_Z = 0.15
GRASP_HEIGHT = 0.267  # fixed EE height for every grasp (rig-tuned)
# Material physics (E/nu/yield_stress/density) is NOT hardcoded here — it is VLM-estimated
# per scene at build time and read from scene.yaml's `material:` block via
# simpact.generator.material.load_material (docs/DEFORMABLE_INTEGRATION.md §15). The
# non-physical solver setup per class lives in material.SOLVER_CONFIG.
SWEEP_BOX = np.array([0.02, 0.10, 0.10])   # thin pusher: sweeps along its x face
SWEEP_LOCAL = np.array([0.0, 0.0, 0.04])   # pusher box centre in the EE local frame
SWEEP_TABLE_Z = 0.145
SWEEP_GRASP_HEIGHT = 0.217                 # pusher height floor (min_height in the original pipeline)
PARTICLE_VOL = 2.5e-8  # per-particle volume the original pipeline used for the FULL cloud (scaled if downsampled)


def _load_mpm_scene(scene_dir, cam, downsample, seed):
    """Shared scene loader: (object_name, init_T, init_pts, n_full, K, cam_to_robot, image_size)."""
    from simpact.utils.layout import find_scene_file
    scene = Path(scene_dir)
    y = yaml.safe_load(find_scene_file(scene, "scene.yaml").read_text())
    object_name = y.get("object_name", "dough")
    init_T = np.asarray(y["initial_ee_pose"], dtype=float).reshape(4, 4)
    pcd_path = Path(y["raw_pcd_path"])
    if not pcd_path.is_absolute() or not pcd_path.exists():
        pcd_path = find_scene_file(scene, Path(y["raw_pcd_path"]).name)
    pts = np.load(pcd_path).astype(np.float64)
    n_full = len(pts)
    if downsample and len(pts) > downsample:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), downsample, replace=False)]
    from simpact.real2sim.camera_calibration import load_camera
    cp = load_camera(scene, cam)  # embedded per-scene, else scene.yaml profile ref
    img_size = tuple(cp.image_size) if cp.image_size else (IMG_W, IMG_H)
    return object_name, init_T, pts, n_full, cp.K, cp.cam_to_robot, img_size


def grasps_from_plan(plan, init_T, init_width: float, close_threshold: float = 0.1):
    """Parse a plan (either schema) into an ordered list of squeezes.

    Ported from the original ``shape_kinetic_sand_multi_step.parse_action_sequence_multi_steps``:
    accumulate in-plane x/y translation and yaw cumulatively; each *closing* GRASP /
    gripper_control (width < ``close_threshold``) commits one grasp at the pose so far.
    RELEASE / opening gripper_control repositions but does not squeeze. Returns
    ``[(center_xyz, yaw, width), ...]`` — one entry per squeeze, applied in order in a
    single continuous rollout (the dough state persists across them).
    """
    dx = dy = dyaw = 0.0
    grasps = []
    seq = plan.action_sequence if hasattr(plan, "action_sequence") else plan
    R0 = np.asarray(init_T, dtype=float)[:3, :3]
    base_yaw = float(np.arctan2(R0[1, 0], R0[0, 0]))
    for a in seq:
        t = a.TYPE.lower()
        if t in ("push", "move", "flick"):
            dx += float(getattr(a, "delta_x", 0.0)); dy += float(getattr(a, "delta_y", 0.0))
        if t in ("rotate", "move"):
            dyaw += float(getattr(a, "delta_yaw", 0.0))
        if t in ("grasp", "gripper_control"):
            w = float(getattr(a, "width", 0.0))
            if w < close_threshold:  # a squeeze (not an open/release)
                center = np.array([init_T[0, 3] + dx, init_T[1, 3] + dy, init_T[2, 3]])
                grasps.append((center, base_yaw + dyaw, w))
    if not grasps:
        raise ValueError("plan has no closing GRASP/gripper_control action")
    return grasps


def sweep_segments_from_plan(plan, init_T, grasp_height: float = SWEEP_GRASP_HEIGHT):
    """Parse a plan (either schema) into a list of sweep segments.

    the original ``sweep_sand_multi_step.py``: the pusher starts at the EE pose (z floored to
    ``grasp_height``) and each PUSH / DESCEND / move is one segment the pusher travels at
    constant velocity, pushing the pile; ROTATE only re-aims it. Returns
    ``[(start_xyz, yaw, delta_xyz), ...]`` — placed at ``start`` and moved by ``delta``
    over the rollout's steps; the next segment starts where this one ended.
    """
    R0 = np.asarray(init_T, dtype=float)[:3, :3]
    base_yaw = float(np.arctan2(R0[1, 0], R0[0, 0]))
    pos = np.array([init_T[0, 3], init_T[1, 3], max(init_T[2, 3], grasp_height)])
    yaw = base_yaw
    segments = []
    seq = plan.action_sequence if hasattr(plan, "action_sequence") else plan
    for a in seq:
        t = a.TYPE.lower()
        if t in ("rotate",):
            yaw += float(getattr(a, "delta_yaw", 0.0)); continue
        if t in ("push", "flick"):
            pos[2] = min(pos[2], grasp_height)  # a horizontal sweep is at table level
            delta = np.array([float(getattr(a, "delta_x", 0.0)),
                              float(getattr(a, "delta_y", 0.0)), 0.0])
        elif t == "descend":
            dz = -abs(float(getattr(a, "delta_z", 0.0)))
            if pos[2] + dz < grasp_height:      # the original clamps the pusher to the floor
                dz = grasp_height - pos[2]
            delta = np.array([0.0, 0.0, dz])
        elif t == "move":
            yaw += float(getattr(a, "delta_yaw", 0.0))
            dz = float(getattr(a, "delta_z", 0.0))
            if pos[2] + dz < grasp_height:
                dz = grasp_height - pos[2]
            delta = np.array([float(getattr(a, "delta_x", 0.0)),
                              float(getattr(a, "delta_y", 0.0)), dz])
        else:
            continue  # GRASP/RELEASE/gripper_control: no-op for a single pusher
        segments.append((pos.copy(), yaw, delta.copy()))
        pos = pos + delta
    if not segments:
        raise ValueError("plan has no PUSH/DESCEND/move sweep segment")
    return segments


def _step_capture(sim, num_steps, dt, device, recorder, frame_fn, stride, shift):
    """Step ``p2g2p`` ``num_steps`` times; every ``stride`` steps append a rendered
    frame to ``recorder`` (the full-simulation mp4). ``frame_fn(pts, frac) -> uint8``
    renders the current particles (+ the tool at progress ``frac``). ``shift`` is the
    grid-centring translation, subtracted to return to the robot frame for rendering."""
    for k in range(1, num_steps):
        sim.p2g2p(k, dt, device=device)
        if recorder is not None and (k % stride == 0 or k == num_steps - 1):
            pts = sim.mpm_state.particle_x.numpy() - shift
            recorder.add(frame_fn(pts, k / num_steps))


class MPMRollout:
    """Multi-grasp dough squeeze rollout (the original ``shape_kinetic_sand_multi_step.py``).

    Applies a *list* of squeezes (``grasps_from_plan`` → 1..N grasps; a single squeeze
    is just the N=1 case) in ONE continuous MPM sim: the two jaw colliders are added once,
    then repositioned at each grasp (fixed ``grasp_height``, initial spread, closing
    twist) via ``set_collision_params`` and stepped ``num_steps`` — the dough deformation
    persists across grasps (state is NOT reset). Accepts both plan schemas via
    ``grasps_from_plan`` (§7: propose primitives, or one ``move``+``gripper_control`` pair
    per squeeze). Writes a full-simulation mp4 (``video=True``) alongside the before/after
    PNGs the VLM sees.
    """

    def __init__(self, scene_dir, cam: int = 1, *, material_params: dict | None = None,
                 num_steps: int = 250, dt: float = 0.001, init_width: float = 0.15,
                 grasp_height: float = GRASP_HEIGHT, table_z: float = DOUGH_TABLE_Z,
                 downsample: int | None = 15000, device: str = "cuda:0", seed: int = 0,
                 video: bool = True, video_fps: int = 15, video_stride: int = 8):
        (self.object_name, self.init_T, self.init_pts, self.n_full,
         self.K, self.cam_to_robot, self.image_size) = _load_mpm_scene(scene_dir, cam, downsample, seed)
        from simpact.generator.material import load_material
        self.material_params = dict(material_params) if material_params else load_material(scene_dir, "dough")
        self.num_steps, self.dt, self.init_width = num_steps, dt, init_width
        self.grasp_height, self.table_z, self.device = grasp_height, table_z, device
        self.particle_vol = PARTICLE_VOL * (self.n_full / len(self.init_pts))
        self.video, self.video_fps, self.video_stride = video, video_fps, video_stride

    def _jaw_transform(self, center_xy, yaw):
        """4x4 world<-jaw-frame pose at a grasp: init orientation yawed, fixed height."""
        T = self.init_T.copy()
        T[:3, :3] = R.from_euler("z", yaw - float(np.arctan2(self.init_T[1, 0], self.init_T[0, 0]))).as_matrix() @ self.init_T[:3, :3]
        T[0, 3], T[1, 3], T[2, 3] = center_xy[0], center_xy[1], self.grasp_height
        return T

    def _viz_boxes(self, T_world, width):
        """Two jaw boxes (center, quat_xyzw, size) in world frame for rendering."""
        Rm, t = T_world[:3, :3], T_world[:3, 3]
        quat = R.from_matrix(Rm).as_quat()
        z_drop = JAW_LOCAL[2]
        boxes = []
        for sign in (-1.0, +1.0):
            local = np.array([0.0, sign * (JAW_LOCAL[1] + width / 2.0), z_drop])
            boxes.append(((Rm @ local) + t, quat, JAW_BOX))
        return boxes

    def run(self, plan, index: int, out_dir) -> str:
        import warp as wp
        import torch
        from simpact.simulators.mpm import MPM_Simulator_WARP
        from simpact.simulators.mpm.robot_utils import set_collision_params

        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        wp.init()
        grasps = grasps_from_plan(plan, self.init_T, self.init_width)

        # --- MPM setup (cloud centred in the grid) ---
        sim = MPM_Simulator_WARP(10, n_grid=GRID_N, grid_lim=GRID_LIM)
        shift = _grid_shift(self.init_pts)
        shifted = self.init_pts + shift
        x = torch.tensor(shifted, dtype=torch.float32, device=self.device)
        vol = torch.ones(len(shifted)) * self.particle_vol
        sim.load_initial_data_from_torch(x, vol, device=self.device)
        sim.set_parameters_dict(self.material_params, device=self.device)
        sim.finalize_mu_lam_bulk(device=self.device)
        sim.add_surface_collider((0.0, 0.0, self.table_z + shift[2]), (0.0, 0.0, 1.0), "sticky", 0.0)

        # add the two jaw colliders once (repositioned per grasp below)
        for _ in range(2):
            sim.add_rotate_box_collider(point=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0),
                                        twist=[0.0] * 6, surface="sticky", friction=0.0,
                                        width=JAW_BOX[0], height=JAW_BOX[1],
                                        length=JAW_BOX[2])

        stem = f"rollout_{index:02d}"
        rec = (VideoRecorder(out_dir / f"{stem}.mp4", self.video_fps, img_size=self.image_size)
               if self.video else None)
        first_boxes = None
        for center_xy, yaw, width in grasps:
            T = self._jaw_transform(center_xy, yaw)
            Rm = T[:3, :3]; quat = R.from_matrix(Rm).as_quat()
            y_speed = (self.init_width - width) / (2 * self.num_steps * self.dt)
            v = Rm @ np.array([0.0, y_speed, 0.0])
            if first_boxes is None:
                first_boxes = self._viz_boxes(T, self.init_width)
            for j, sign in enumerate((-1.0, +1.0)):  # collider_params[-2], [-1]
                local = JAW_LOCAL.copy(); local[1] = sign * (JAW_LOCAL[1] + self.init_width / 2.0)
                center = (Rm @ local) + T[:3, 3] + shift
                set_collision_params(sim.collider_params[-2 + j], center, quat,
                                     [0.0, 0.0, 0.0, -sign * v[0], -sign * v[1], -sign * v[2]])

            def frame_fn(pts, frac, T=T, width=width):
                spread = self.init_width - (self.init_width - width) * frac
                return render_deformable(pts, self.K, self.cam_to_robot, title="dough (MPM)",
                                         tool_boxes=self._viz_boxes(T, spread), return_array=True,
                                         img_size=self.image_size)

            _step_capture(sim, self.num_steps, self.dt, self.device, rec, frame_fn, self.video_stride, shift)
        final_pts = sim.mpm_state.particle_x.numpy() - shift
        if rec is not None:
            rec.save()

        before = render_deformable(self.init_pts, self.K, self.cam_to_robot,
                                   out_dir / f"{stem}_0.png", title="before (dough)",
                                   tool_boxes=first_boxes, img_size=self.image_size)
        after = render_deformable(final_pts, self.K, self.cam_to_robot,
                                  out_dir / f"{stem}_1.png", title="after (MPM, N grasps)",
                                  img_size=self.image_size)
        np.save(out_dir / f"{stem}_final_points.npy", final_pts.astype(np.float32))

        bmin, bmax = final_pts.min(0), final_pts.max(0)
        data = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "object_names": [self.object_name],
            "grasp_centers": [g[0].tolist() for g in grasps],
            "grasp_yaws": [g[1] for g in grasps],
            "grasp_widths": [g[2] for g in grasps],
            "final_width": grasps[-1][2],
            "mpm": {
                "final_points_path": f"{stem}_final_points.npy",
                "bbox_min": bmin.tolist(), "bbox_max": bmax.tolist(),
                "bbox_size": (bmax - bmin).tolist(), "centroid": final_pts.mean(0).tolist(),
                "num_grasps": len(grasps),
            },
            "snapshots": [
                {"waypoint_index": 0, "gripper": {"position": grasps[0][0].tolist(),
                                                  "width": self.init_width},
                 "objects": {self.object_name: {"position": self.init_pts.mean(0).tolist()}},
                 "screenshot": Path(before).name},
                {"waypoint_index": 1, "gripper": {"position": grasps[-1][0].tolist(),
                                                  "width": grasps[-1][2]},
                 "objects": {self.object_name: {"position": final_pts.mean(0).tolist()}},
                 "screenshot": Path(after).name},
            ],
        }
        path = out_dir / f"{stem}.json"
        path.write_text(json.dumps(data, indent=2))
        return str(path)


class SweepRollout:
    """Single-pusher sweep rollout (the original ``sweep_sand_multi_step.py``).

    A thin box (``SWEEP_BOX``) is added once and moved segment by segment with a linear
    twist velocity (``delta / (num_steps·dt)``) in one continuous MPM sim, pushing a
    coherent pile toward a target region. The scene's ``bg_pcd_path`` (the target) is
    drawn into the renders and drives the measured ``coverage_gate`` (verify.py) — the
    first measured deformable success signal. Accepts both plan schemas via
    ``sweep_segments_from_plan``. The blade's yaw (orientation) is chosen by the plan —
    the prompt guides the VLM to keep its flat face square to the push so the pile is
    swept together into the target.
    """

    def __init__(self, scene_dir, cam: int = 1, *, material_params: dict | None = None,
                 num_steps: int = 100, dt: float = 0.002, grasp_height: float = SWEEP_GRASP_HEIGHT,
                 table_z: float = SWEEP_TABLE_Z, downsample: int | None = 15000,
                 device: str = "cuda:0", seed: int = 0,
                 video: bool = True, video_fps: int = 15, video_stride: int = 5):
        (self.object_name, self.init_T, self.init_pts, self.n_full,
         self.K, self.cam_to_robot, self.image_size) = _load_mpm_scene(scene_dir, cam, downsample, seed)
        from simpact.generator.material import load_material
        self.material_params = dict(material_params) if material_params else load_material(scene_dir, "sweep")
        self.num_steps, self.dt = num_steps, dt
        self.grasp_height, self.table_z, self.device = grasp_height, table_z, device
        self.particle_vol = PARTICLE_VOL * (self.n_full / len(self.init_pts))
        self.video, self.video_fps, self.video_stride = video, video_fps, video_stride
        # the target region (bg_pcd) — for rendering + the coverage gate
        from simpact.utils.layout import find_scene_file
        y = yaml.safe_load(find_scene_file(scene_dir, "scene.yaml").read_text())
        self.target_pts = None
        if y.get("bg_pcd_path"):
            import open3d as o3d
            tp = Path(y["bg_pcd_path"])
            if not tp.is_absolute() or not tp.exists():
                tp = find_scene_file(scene_dir, Path(y["bg_pcd_path"]).name)
            self.target_pts = np.asarray(o3d.io.read_point_cloud(str(tp)).points)

    def _pusher_pose(self, center_xyz, yaw):
        T = self.init_T.copy()
        T[:3, :3] = R.from_euler("z", yaw - float(np.arctan2(self.init_T[1, 0], self.init_T[0, 0]))).as_matrix() @ self.init_T[:3, :3]
        T[:3, 3] = center_xyz
        return T

    def _render(self, pts, path, title, pusher=None, return_array=False):
        """Beans (uniform brown spheres) + the target region (magenta) + optional pusher
        box (depth-composited). Saves ``path`` and/or returns the frame array."""
        return render_deformable(
            pts, self.K, self.cam_to_robot, path, title=title,
            color=(0.42, 0.26, 0.14), extra_points=self.target_pts,
            tool_boxes=([pusher] if pusher is not None else None),
            point_size=9.0, return_array=return_array, img_size=self.image_size)

    def _pusher_box(self, T):
        Rm, t = T[:3, :3], T[:3, 3]
        return ((Rm @ SWEEP_LOCAL) + t, R.from_matrix(Rm).as_quat(), SWEEP_BOX)

    def run(self, plan, index: int, out_dir) -> str:
        import warp as wp
        import torch
        from simpact.simulators.mpm import MPM_Simulator_WARP
        from simpact.simulators.mpm.robot_utils import set_collision_params

        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        wp.init()
        segments = sweep_segments_from_plan(plan, self.init_T, self.grasp_height)

        sim = MPM_Simulator_WARP(10, n_grid=GRID_N, grid_lim=GRID_LIM)
        shift = _grid_shift(self.init_pts)
        shifted = self.init_pts + shift
        x = torch.tensor(shifted, dtype=torch.float32, device=self.device)
        vol = torch.ones(len(shifted)) * self.particle_vol
        sim.load_initial_data_from_torch(x, vol, device=self.device)
        sim.set_parameters_dict(self.material_params, device=self.device)
        sim.finalize_mu_lam_bulk(device=self.device)
        sim.add_surface_collider((0.0, 0.0, self.table_z + shift[2]), (0.0, 0.0, 1.0), "sticky", 0.0)
        sim.add_rotate_box_collider(point=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0),
                                    twist=[0.0] * 6, surface="sticky", friction=0.0,
                                    width=SWEEP_BOX[0], height=SWEEP_BOX[1], length=SWEEP_BOX[2])

        stem = f"rollout_{index:02d}"
        rec = (VideoRecorder(out_dir / f"{stem}.mp4", self.video_fps, img_size=self.image_size)
               if self.video else None)
        first_pusher = None
        for start_xyz, yaw, delta in segments:
            T = self._pusher_pose(start_xyz, yaw)
            if first_pusher is None:
                first_pusher = self._pusher_box(T)
            box_center = (T[:3, :3] @ SWEEP_LOCAL) + T[:3, 3] + shift
            v = delta / (self.num_steps * self.dt)
            set_collision_params(sim.collider_params[-1], box_center,
                                 R.from_matrix(T[:3, :3]).as_quat(),
                                 [0.0, 0.0, 0.0, v[0], v[1], v[2]])

            def frame_fn(pts, frac, start=start_xyz, yaw=yaw, delta=delta):
                pusher = self._pusher_box(self._pusher_pose(start + delta * frac, yaw))
                return self._render(pts, None, "sweep (MPM)", pusher, return_array=True)

            _step_capture(sim, self.num_steps, self.dt, self.device, rec, frame_fn, self.video_stride, shift)
        final_pts = sim.mpm_state.particle_x.numpy() - shift
        if rec is not None:
            rec.save()

        before = self._render(self.init_pts, out_dir / f"{stem}_0.png", "before (beans + target)", first_pusher)
        after = self._render(final_pts, out_dir / f"{stem}_1.png", "after (swept)")
        np.save(out_dir / f"{stem}_final_points.npy", final_pts.astype(np.float32))

        bmin, bmax = final_pts.min(0), final_pts.max(0)
        data = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "object_names": [self.object_name],
            "sweep_segments": [{"start": s.tolist(), "yaw": float(y), "delta": d.tolist()}
                               for s, y, d in segments],
            "mpm": {
                "final_points_path": f"{stem}_final_points.npy",
                "bbox_min": bmin.tolist(), "bbox_max": bmax.tolist(),
                "bbox_size": (bmax - bmin).tolist(), "centroid": final_pts.mean(0).tolist(),
                "num_segments": len(segments),
            },
            "snapshots": [
                {"waypoint_index": 0, "gripper": {"position": segments[0][0].tolist(), "width": 0.0},
                 "objects": {self.object_name: {"position": self.init_pts.mean(0).tolist()}},
                 "screenshot": Path(before).name},
                {"waypoint_index": 1, "gripper": {"position": (segments[-1][0] + segments[-1][2]).tolist(), "width": 0.0},
                 "objects": {self.object_name: {"position": final_pts.mean(0).tolist()}},
                 "screenshot": Path(after).name},
            ],
        }
        path = out_dir / f"{stem}.json"
        path.write_text(json.dumps(data, indent=2))
        return str(path)
