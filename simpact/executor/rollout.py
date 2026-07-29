"""MuJoCo action-evaluation rollout (rigid).

Drives the mocap gripper of a ``build_mujoco_scene(with_gripper=True)`` scene through the
waypoints of a proposal, steps physics, and **records the outcome the VLM
optimizer reads** — per-snapshot gripper/object poses + an overhead screenshot —
into legacy-format rollout JSON (consumable by a ``regress_gemini``-style port).
No scoring/ranking happens here (the original had no evaluator).

Ported from the original ``executor/push_6d.py`` (FloatingGripperController + trajectory
follower + RolloutRecorder), built on the unified scene generator and the typed
action schema. CPU-OK and deterministic; offscreen rendering needs a GL context
(``MUJOCO_GL=egl``) and degrades gracefully (JSON still written) if unavailable.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from simpact.actions.primitives import ActionProposal
from simpact.executor.waypoints import Waypoint, proposal_to_waypoints
from simpact.real2sim.scene import build_mujoco_scene

RENDER_H, RENDER_W = 480, 640  # matches the recorded screenshots

# the original pipeline's real Franka gripper home pose (executor/push_6d.py), in the MuJoCo **mocap**
# frame that `FloatingGripperController.set_gripper_pose` drives — i.e. directly
# usable as a rollout's initial gripper pose. The original read this live from the robot
# (`state.O_T_EE`) at capture time, embeds the end-effector pose in `context.txt`,
# and the sim converts EE->mocap via the 0.105 m tool offset
# (`real2sim/convert_gripper_pose`). We pin the recorded home pose so the sim starts
# where the real arm did, instead of an ad-hoc heuristic. orientation is wxyz.
HOME_GRIPPER_POSITION = (0.273422, 0.282776, 0.4057)
HOME_GRIPPER_ORIENTATION = (0.00154509, 0.296818, 0.9547, -0.0211092)  # wxyz


def _slerp(q0, q1, t):
    """SLERP between wxyz quaternions."""
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1; dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        theta = np.arccos(np.clip(dot, -1, 1)) * t
        q2 = q1 - q0 * dot
        q2 /= np.linalg.norm(q2)
        q = q0 * np.cos(theta) + q2 * np.sin(theta)
    return (q / np.linalg.norm(q)).tolist()


@dataclass
class RolloutResult:
    object_names: list[str]
    timestamp: str
    waypoints: list[dict]
    snapshots: list[dict]  # each: waypoint_index, gripper{position,orientation,width}, objects{...}
    frames: list = field(default_factory=list, repr=False)  # per-snapshot RGB arrays or None
    metrics: dict = field(default_factory=dict)
    # full-simulation video frames (strided; a debugging artifact like the MPM/rope
    # mp4s — NOT shown to the VLM, which sees only the before/after PNGs)
    video_frames: list = field(default_factory=list, repr=False)
    video_fps: int = 30

    @property
    def initial_poses(self) -> dict:
        return self.snapshots[0]["objects"]

    @property
    def final_poses(self) -> dict:
        return self.snapshots[-1]["objects"]

    def displacement(self, name: str) -> float:
        a = np.asarray(self.initial_poses[name]["position"])
        b = np.asarray(self.final_poses[name]["position"])
        return float(np.linalg.norm(b - a))

    def to_dict(self, *, proposal_index: Optional[int] = None,
                instruction: Optional[str] = None) -> dict:
        d: dict = {"timestamp": self.timestamp}
        if proposal_index is not None:
            d["proposal_index"] = proposal_index
        if instruction is not None:
            d["instruction"] = instruction
        d["object_names"] = self.object_names
        d["waypoints"] = self.waypoints
        d["snapshots"] = self.snapshots
        return d

    def save(self, out_dir: Union[str, Path], index: int,
             instruction: Optional[str] = None) -> str:
        """Write ``rollout_<index>.json`` + ``rollout_<index>_<wp>.png`` to out_dir.

        Sets each snapshot's ``screenshot`` to the PNG filename (relative to the
        JSON). If a frame is missing (no GL context), ``screenshot`` is null.
        """
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"rollout_{index:02d}"
        for snap, frame in zip(self.snapshots, self.frames):
            if frame is None:
                snap["screenshot"] = None
                continue
            from PIL import Image

            name = f"{stem}_{snap['waypoint_index']}.png"
            Image.fromarray(frame).save(out_dir / name)
            snap["screenshot"] = name
        if self.frames and self.frames[0] is not None:
            from PIL import Image

            # stable aliases (the per-waypoint index of the final frame varies with
            # the plan length; galleries/docs link these instead)
            Image.fromarray(self.frames[0]).save(out_dir / f"{stem}_before.png")
            Image.fromarray(self.frames[-1]).save(out_dir / f"{stem}_after.png")
        d = self.to_dict(proposal_index=index, instruction=instruction)
        if self.video_frames:
            from simpact.executor.render_deformable import VideoRecorder

            rec = VideoRecorder(out_dir / f"{stem}.mp4", fps=self.video_fps,
                                img_size=(RENDER_W, RENDER_H))
            for f in self.video_frames:
                rec.add(f)
            if rec.save():
                d["video"] = f"{stem}.mp4"
        path = out_dir / f"{stem}.json"
        path.write_text(json.dumps(d, indent=2))
        return str(path)


class MuJoCoRollout:
    """Build a gripper scene from objects and roll proposals out in it."""

    def __init__(self, objects: Sequence[dict], *, with_table: bool = True,
                 table_pose=None, xml_path: Optional[Union[str, Path]] = None):
        self.objects = list(objects)
        self.body_names = [str(o["name"]).replace(" ", "_") for o in self.objects]
        xml_path = xml_path or tempfile.mktemp(suffix="_rollout.xml")
        self.xml = build_mujoco_scene(self.objects, xml_path, with_gripper=True,
                               with_table=with_table, table_pose=table_pose)

    def _object_poses(self, mj, model, data) -> dict:
        out = {}
        for name in self.body_names:
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            out[name] = {
                "position": data.xpos[bid].copy().tolist(),
                "orientation": data.xquat[bid].copy().tolist(),
            }
        return out

    def run(
        self,
        proposal: Union[ActionProposal, Sequence],
        initial_position,
        initial_orientation=(0.0, 1.0, 0.0, 0.0),
        *,
        z_min: float = 0.0,
        default_duration: float = 1.5,
        initial_gripper: float = 0.0,
        settle_steps: int = 300,
        render: bool = True,
        render_camera: str = "top",
        video: bool = True,
    ) -> RolloutResult:
        if render:
            os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen GL backend
        import mujoco
        from simpact.real2sim.mujoco_load_gripper import FloatingGripperController

        ctrl = FloatingGripperController(self.xml)
        model, data = ctrl.model, ctrl.data
        dt = float(model.opt.timestep)

        # offscreen renderer (degrade gracefully if no GL); render_camera selects
        # the viewpoint (e.g. the original pipeline's "top_view")
        renderer = None
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, render_camera)
        if render:
            try:
                renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
            except Exception as e:
                print(f"[rollout] offscreen render unavailable ({e!r}); JSON only.")

        def grab():
            if renderer is None:
                return None
            renderer.update_scene(data, camera=cam_id if cam_id >= 0 else -1)
            return renderer.render().copy()

        waypoints = proposal_to_waypoints(
            proposal, initial_position, initial_orientation,
            z_min=z_min, default_duration=default_duration, initial_gripper=initial_gripper,
        )

        snapshots, frames = [], []

        # Full-simulation video: strided frames captured while stepping, so the
        # clip plays at ~real (simulated) time. Same convention as the MPM/rope
        # rollout mp4s — a debugging artifact, never shown to the VLM.
        video_frames: list = []
        vstride = max(1, round(1.0 / (30 * dt)))          # ~30 fps of sim time
        video_fps = max(1, round(1.0 / (vstride * dt)))
        _step_i = [0]

        def vstep():
            """mj_step + strided video capture."""
            mujoco.mj_step(model, data)
            if video and renderer is not None and _step_i[0] % vstride == 0:
                video_frames.append(grab())
            _step_i[0] += 1

        def capture(wp_index):
            gp, gq = ctrl.get_gripper_pose()
            snapshots.append({
                "waypoint_index": wp_index,
                "gripper": {"position": gp.tolist(), "orientation": gq.tolist(),
                            "width": float(ctrl.get_gripper_width())},
                "objects": self._object_poses(mujoco, model, data),
            })
            frames.append(grab())

        w0 = waypoints[0]
        ctrl.set_gripper_pose(np.asarray(w0.position), np.asarray(w0.orientation))
        ctrl.set_gripper_width(w0.gripper_width)
        ctrl.snap_to_mocap()  # start the gripper AT its pose (no weld fly-in that
        #                       would sweep across the scene and knock objects over)
        mujoco.mj_forward(model, data)
        capture(0)  # initial state == "before" image

        for i, (prev, wp) in enumerate(zip(waypoints[:-1], waypoints[1:]), start=1):
            n = max(1, int(wp.duration / dt))
            for k in range(1, n + 1):
                t = k / n
                pos = (1 - t) * np.asarray(prev.position) + t * np.asarray(wp.position)
                quat = _slerp(prev.orientation, wp.orientation, t)
                width = (1 - t) * prev.gripper_width + t * wp.gripper_width
                ctrl.set_gripper_pose(pos, np.asarray(quat))
                ctrl.set_gripper_width(width)
                vstep()
            capture(i)

        for _ in range(settle_steps):
            vstep()
        capture(len(waypoints))  # settled final == "after" image

        if renderer is not None:
            renderer.close()

        result = RolloutResult(
            object_names=self.body_names,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            waypoints=[w.to_dict() for w in waypoints],
            snapshots=snapshots,
            frames=frames,
            video_frames=video_frames,
            video_fps=video_fps,
        )
        result.metrics = {name: result.displacement(name) for name in self.body_names}
        return result
