"""ARAP rope rollout — the deformable analogue of ``MuJoCoRollout`` for rope.

Wraps ``simulators.arap.EmbedDeformGraph``: builds the deformation graph from a
trial's segmented cloud, applies a plan (grasp a local blob, translate it to a place
point), solves the ARAP energy, renders before/after from the real camera viewpoint,
and writes the same rollout-JSON envelope the verifier/optimizer read
(``snapshots[*].{screenshot, objects, gripper}`` + ``object_names``) plus a rope
block (``grasp_point``/``place_point``/keypoints) for ``parse_rope_rollout``.

Accepts BOTH plan schemas (see docs/DEFORMABLE_INTEGRATION.md §7): propose primitives
(``PUSH``/``DESCEND``/``GRASP``/``RELEASE``) and the optimizer's ``move``/
``gripper_control`` — grasp/place are recovered by accumulating the in-plane deltas
from the context EE pose up to the grasp/release. Quasi-static: one solve per plan,
no time loop, no GPU stepping (torch/pypose only).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from simpact.simulators.arap import EmbedDeformGraph
from simpact.simulators.arap.pts_utils import connect_points
from simpact.executor.render_deformable import render_deformable, VideoRecorder, IMG_W, IMG_H

GRIP_RADIUS = 0.03
VOXEL = 0.003
ROPE_COLOR = (0.55, 0.40, 0.25)  # uniform rope colour for the sphere-glyph render


def grasp_place_from_plan(plan, init_xy, rope_z):
    """Recover (grasp, place) 3-D points from a plan of either schema.

    Accumulate the in-plane (x,y) deltas of PUSH/MOVE from the context EE start; the
    position at the grasp (GRASP or the closing GRIPPER_CONTROL) is the grasp point,
    the position at the release (RELEASE / opening GRIPPER_CONTROL, else the end) is
    the place point. z is pinned to the rope/table plane.
    """
    pos = np.asarray(init_xy, dtype=float)[:2].copy()
    grasp = place = None
    for a in plan.action_sequence:
        t = a.TYPE.lower()  # propose primitives are UPPER, plan actions lower
        if t in ("push", "move", "flick"):
            pos = pos + [getattr(a, "delta_x", 0.0), getattr(a, "delta_y", 0.0)]
        elif t in ("grasp", "gripper_control"):
            w = float(getattr(a, "width", 0.0))
            if grasp is None and w <= 0.06:      # closing on the rope
                grasp = pos.copy()
            elif grasp is not None and w >= 0.09:  # opening / release
                place = pos.copy()
        elif t == "release" and grasp is not None and place is None:
            place = pos.copy()
    if grasp is None:
        raise ValueError("plan has no grasp (GRASP / gripper_control) action")
    if place is None:
        place = pos.copy()  # never released -> final position is the place
    return (np.array([grasp[0], grasp[1], rope_z]),
            np.array([place[0], place[1], rope_z]))


@dataclass
class RopeRolloutResult:
    grasp: np.ndarray
    place: np.ndarray
    init_kps: np.ndarray
    final_kps: np.ndarray
    frames: list  # (tag, path)

    def displacement(self) -> float:
        return float(np.linalg.norm(self.final_kps.mean(0) - self.init_kps.mean(0)))


class ARAPRollout:
    """Build the rope graph once; roll plans out in it (loop-compatible rollout_fn)."""

    def __init__(self, scene_dir, cam: int = 1, keypoint_voxel: float = 0.05,
                 device: str | None = None, free_end_only: bool = True,
                 video: bool = True, video_fps: int = 15, video_frames: int = 30,
                 min_z: float | None = None):
        # min_z: floor the ARAP solve clamps free-node z to. Default None -> the rope's
        # own minimum z, i.e. NO artificial floor above the rope. The original hardcoded 0.16, but
        # the rope rests ~0.14-0.16, so a 0.16 floor lifts free nodes and that clamp
        # perturbation cascades through the solve into ~1.4 cm of spurious motion (the
        # rest frame != first sim frame). Using rope-min removes that jump; pass an
        # explicit float only to model a genuine table above the rope.
        self.scene = Path(scene_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.video, self.video_fps, self.video_frames = video, video_fps, video_frames
        # free_end_only: the fixed end is anchored (real robot) — only the free end is
        # graspable, so the plan's grasp is overridden to the scene's free_end and the
        # VLM effectively only chooses where to drag it (docs/DEFORMABLE_INTEGRATION §11).
        self.free_end_only = free_end_only
        import yaml
        from simpact.utils.layout import find_scene_file
        y = yaml.safe_load(find_scene_file(self.scene, "scene.yaml").read_text())
        self.fixed_pt = np.array(y["fixed_point"]); self.rope_z = float(self.fixed_pt[2])
        self.free_end = np.array(y["free_end"]); self.free_end[2] = self.rope_z
        from simpact.generator.context import resolve_initial_ee
        self.init_ee = resolve_initial_ee(self.scene)[0].position
        self.cam = cam
        from simpact.real2sim.camera_calibration import load_camera
        cp = load_camera(self.scene, cam)  # embedded per-scene, else scene.yaml profile ref
        self.K, self.cam_to_robot = cp.K, cp.cam_to_robot
        self.image_size = tuple(cp.image_size) if cp.image_size else (IMG_W, IMG_H)

        raw = o3d.io.read_point_cloud(str(find_scene_file(self.scene, "segmented_object.ply")))
        self.raw_pts = np.asarray(raw.points)
        self.colors = (np.asarray(raw.colors) if raw.has_colors()
                       else np.tile([0.6, 0.4, 0.2], (len(self.raw_pts), 1)))
        ds = o3d.geometry.PointCloud(); ds.points = o3d.utility.Vector3dVector(self.raw_pts)
        self.sim_pts = np.asarray(ds.voxel_down_sample(VOXEL).points)
        e1, e2 = connect_points(self.sim_pts, 0.01), connect_points(self.sim_pts, 0.015)
        edges = np.vstack((e1, e2))
        weights = np.hstack((5.0 * np.ones(len(e1)), 100.0 * np.ones(len(e2))))
        self.min_z = float(self.raw_pts[:, 2].min()) if min_z is None else min_z
        self.graph = EmbedDeformGraph(self.sim_pts, edges, edge_weights=weights,
                                      corotate=True, vis_pts=self.raw_pts, rbf_sig=0.3,
                                      rbf_w_max=0.2, dist_max=0.2, min_z=self.min_z,
                                      device=self.device)
        kp = ds.voxel_down_sample(keypoint_voxel)
        self.kp_idx = np.asarray([int(np.argmin(np.linalg.norm(self.sim_pts - p, axis=1)))
                                  for p in np.asarray(kp.points)])
        self.fixed_idx = np.where(np.linalg.norm(self.sim_pts - self.fixed_pt, axis=1) < GRIP_RADIUS)[0]

    def run(self, plan, index: int, out_dir) -> str:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        grasp, place = grasp_place_from_plan(plan, self.init_ee, self.rope_z)
        if self.free_end_only:  # grasp is always the free end; keep the planned drag
            drag = place - grasp
            grasp = self.free_end.copy(); place = grasp + drag
        move_idx = np.where(np.linalg.norm(self.sim_pts - grasp, axis=1) < GRIP_RADIUS)[0]
        if len(move_idx) == 0:  # grasp off the curved rope -> nearest node (see §6)
            push_delta = place - grasp
            n = int(np.argmin(np.linalg.norm(self.sim_pts - grasp, axis=1)))
            grasp = self.sim_pts[n].copy(); place = grasp + push_delta
            move_idx = np.where(np.linalg.norm(self.sim_pts - grasp, axis=1) < GRIP_RADIUS)[0]

        self.graph.reset_state()
        init_kps = self.sim_pts[self.kp_idx].copy()
        handle_idx = np.append(move_idx, self.fixed_idx)
        self.graph.set_handle_idx(handle_idx)
        rest_hp = self.graph.rest_pts_tsr[handle_idx, :].clone()
        disp = torch.tensor(place - grasp, dtype=rest_hp.dtype, device=rest_hp.device)

        # Deform in N incremental drags of the free end (rest -> full), solving ARAP
        # warm-started at each — this is the "full deformation process" the video shows.
        # With video off it collapses to a single full-displacement solve (unchanged).
        stem = f"rollout_{index:02d}"
        rec = (VideoRecorder(out_dir / f"{stem}.mp4", self.video_fps, img_size=self.image_size)
               if self.video else None)
        n = self.video_frames if self.video else 1
        hp = rest_hp.clone()
        for i in range(1, n + 1):
            frac = i / n
            hp = rest_hp.clone(); hp[:len(move_idx), :] += frac * disp
            with torch.no_grad():
                self.graph.solve_global_local(
                    handle_idx, hp, num_iters=(100 if frac == 1.0 else 20),
                    energy_converge_threshold=1e-4)
            if rec is not None:
                vis = np.asarray(self.graph.get_vis_pcd().points)
                rec.add(render_deformable(vis, self.K, self.cam_to_robot, title="rope (ARAP)",
                                          color=ROPE_COLOR, point_size=9.0, return_array=True,
                                          img_size=self.image_size))
        after_vis = np.asarray(self.graph.get_vis_pcd().points)
        final_kps = self.graph.get_curr_pts(handle_idx, hp).detach().cpu().numpy()[self.kp_idx]
        if rec is not None:
            rec.save()

        # before / after stills (solid sphere glyphs, camera-posed — same style as MPM)
        render_deformable(self.raw_pts, self.K, self.cam_to_robot, out_dir / f"{stem}_0.png",
                          title="before (rope)", color=ROPE_COLOR, point_size=9.0, img_size=self.image_size)
        render_deformable(after_vis, self.K, self.cam_to_robot, out_dir / f"{stem}_1.png",
                          title="after (ARAP)", color=ROPE_COLOR, point_size=9.0, img_size=self.image_size)

        data = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "object_names": ["rope"],
            "grasp_point": grasp.tolist(), "place_point": place.tolist(),
            "snapshots": [
                {"waypoint_index": 0, "gripper": {"position": grasp.tolist()},
                 "objects": {"rope": {"position": init_kps.mean(0).tolist()}},
                 "screenshot": f"{stem}_0.png"},
                {"waypoint_index": 1, "gripper": {"position": place.tolist()},
                 "objects": {"rope": {"position": final_kps.mean(0).tolist()}},
                 "screenshot": f"{stem}_1.png"},
            ],
        }
        path = out_dir / f"{stem}.json"
        path.write_text(json.dumps(data, indent=2))
        return str(path)
