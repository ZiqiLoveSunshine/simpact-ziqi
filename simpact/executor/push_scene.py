"""Push scene assembly + rollout — the rigid-body ``RolloutLike``.

Builds the MuJoCo scene ONCE from a trial dir holding reconstructed assets
(textured ``{obj}_scaled.obj`` + ``{obj}_6d_cam{cam}.txt`` — e.g. a
perception build output, or ``reproduce_all``'s staged work dir), then exposes
the same ``run(plan, index, out_dir) -> path`` protocol as
``ARAPRollout``/``MPMRollout`` so one driver treats all materials alike.

Scene assembly (faithful to the former ``demo_optimize_loop_push.py``):

* Object world poses are recomputed from the per-camera 6-DoF estimate and the
  **aligned per-scene camera->robot extrinsic** (``transform_to_robot_frame``),
  NOT read from stored ``_mujoco_cam`` files — those predate the committed 0103
  calibration for older trials and are ~15-18 cm off.
  Each object's base is snapped onto the table top so nothing penetrates.
* The gripper starts at the trial's recorded EE pose
  (``context.resolve_initial_ee``; the bundled example for the trial name is the
  fallback), converted EE->mocap by the 0.105 m tool offset matching the real-robot convention; the original pipeline's
  generic Franka home pose is the last resort.
* The fixed world frame (table pose/top, ``top_view`` camera) comes from the original pipeline's
  ``real2sim/generate_xml.py`` rig, matching the reconstructed scene.
"""
import glob
from pathlib import Path

import numpy as np

from simpact.executor.rollout import (
    HOME_GRIPPER_ORIENTATION,
    HOME_GRIPPER_POSITION,
    MuJoCoRollout,
)
from simpact.generator.context import EEPose, resolve_initial_ee
from simpact.real2sim.camera_calibration import load_camera
from simpact.real2sim.convert_gripper_pose import ee_pose_from_matrix, ee_pose_to_matrix
from simpact.real2sim.transform_6d import transform_to_robot_frame
from simpact.utils.config import get_project_root

# Fixed world frame (from real2sim generate_xml.py): table + the top_view camera. Named
# They match the real rig's world frame the reconstructed scene is built in.
RIG_TABLE_POSE = (0.5243, -0.0009, 0.14)
RIG_TABLE_TOP = 0.16  # table center z + half thickness


def _bundled_example(data_dir: Path) -> Path:
    """The committed push example for this trial name (per-scene fallback source)."""
    return get_project_root() / "examples" / "push_real2sim" / data_dir.name


def resolve_init_ee(data_dir: Path, init_ee_file=None):
    """Find the trial's real initial end-effector pose held at capture.

    Resolution: an explicit ``init_ee_file`` (a recorded ``context.txt`` or a 4x4
    matrix file) wins; else ``resolve_initial_ee`` (scene.yaml ``initial_ee_pose``
    -> initial_ee_pose.txt -> context.txt) over the trial dir, then the bundled
    repo example for this trial name. Returns ``(EEPose | None, source_label)``
    — ``None`` falls back to the generic home pose.
    """
    if init_ee_file:
        p = Path(init_ee_file)
        ee = (EEPose.from_context_file(p) if p.name == "context.txt"
              else EEPose.from_matrix(np.loadtxt(p)))
        return ee, str(p)
    for c in (data_dir, _bundled_example(data_dir)):
        try:
            return resolve_initial_ee(c)
        except FileNotFoundError:
            continue
    return None, "none"


def resolve_calib_dir(data_dir: Path) -> Path:
    """Dir that resolves the camera calibration **per scene**: the trial dir if it embeds
    or references calibration, else the committed push example for this trial (whose
    ``scene.yaml`` carries the ``camera: {profile}`` reference). Mirrors ``resolve_init_ee``
    — calibration, like the initial EE pose, is a per-scene asset, not a code default."""
    for c in (data_dir, _bundled_example(data_dir)):
        try:
            load_camera(c, 1)
            return c
        except FileNotFoundError:
            continue
    return data_dir  # let the downstream extrinsic lookup raise a clear calibration error


def build_objects(data_dir: Path, names, cam: int, calib_dir: Path,
                  table_top: float = RIG_TABLE_TOP):
    """Objects from a trial: textured scaled mesh + world pose from the aligned
    extrinsic (`camera->robot @ 6d_cam`), with the base snapped onto the table. The
    extrinsic is resolved per scene from ``calib_dir``; assets resolve through the
    bundled-trial layout (``sim/``) or a flat perception-build dir."""
    import trimesh  # lazy (real2sim extra)
    from simpact.utils.layout import find_scene_file
    objs = []
    for name in names:
        mesh_file = find_scene_file(data_dir, f"{name}_scaled.obj")
        tex = find_scene_file(data_dir, f"{name}_scaled_0.png", required=False)
        pose_cam = np.loadtxt(find_scene_file(data_dir, f"{name}_6d_cam{cam}.txt")).reshape(4, 4)
        pose = transform_to_robot_frame(pose_cam, cam, calib_dir)  # aligned camera->robot
        # snap base to the table top so the object rests (no penetration / float)
        mesh = trimesh.load(mesh_file, force="mesh")
        world_z = (pose[:3, :3] @ np.asarray(mesh.vertices).T).T[:, 2] + pose[2, 3]
        pose[2, 3] += table_top - float(world_z.min())
        o = {"name": name, "mesh_file": str(mesh_file), "pose": pose}
        if tex is not None:
            o["texture"] = str(tex)
        objs.append(o)
    return objs


class PushSceneRollout:
    """Assemble the push MuJoCo scene once; roll out plans via the common protocol.

    Exposes ``object_names`` (discovered or given) and ``context_ee`` (the EE-frame
    pose the VLM context advertises) for the driver's context build and gate wiring.
    """

    def __init__(self, data_dir, cam: int = 1, view: str = "top_view",
                 instruction: str = "", objects: str | None = None,
                 init_ee_file=None, initial_ee: str | None = None,
                 table_pose=RIG_TABLE_POSE, table_top: float = RIG_TABLE_TOP,
                 video: bool = True):
        self.data_dir = data_dir = Path(data_dir)
        self.cam, self.view, self.instruction, self.table_top = cam, view, instruction, table_top
        self.video = video

        if objects:
            names = [s.strip() for s in objects.replace(".", ",").split(",") if s.strip()]
        else:  # all objects with a 6-DoF estimate in the trial (sim/ or flat layout)
            names = sorted({Path(p).name.replace(f"_6d_cam{cam}.txt", "")
                            for pat in (data_dir / f"*_6d_cam{cam}.txt",
                                        data_dir / "sim" / f"*_6d_cam{cam}.txt")
                            for p in glob.glob(str(pat))})
        self.object_names = names
        calib_dir = resolve_calib_dir(data_dir)
        objs = build_objects(data_dir, names, cam, calib_dir, table_top=table_top)
        print(f"scene: {names} (real poses + textures)")

        # Initial gripper pose: the trial's REAL end-effector pose (what the real arm
        # held at capture), else the generic Franka home. The VLM context advertises
        # the EE-frame pose; the sim uses the mocap pose (EE->mocap by the 0.105 m tool
        # offset, matching the real-robot convention).
        init_ee, ee_src = resolve_init_ee(data_dir, init_ee_file)
        if init_ee is not None:
            self.context_ee = init_ee                    # EE frame -> VLM context
            mocap_pos, ee_quat_wxyz = ee_pose_from_matrix(init_ee.to_matrix())  # -> sim mocap
            self._ee_xyz, self._ee_quat_wxyz = list(mocap_pos), list(ee_quat_wxyz)
            print(f"initial gripper: EE pose from {ee_src}")
        else:
            self._ee_xyz = list(HOME_GRIPPER_POSITION)   # generic home (mocap frame)
            self._ee_quat_wxyz = list(HOME_GRIPPER_ORIENTATION)
            self.context_ee = EEPose.from_matrix(
                ee_pose_to_matrix(np.asarray(self._ee_xyz), np.asarray(self._ee_quat_wxyz)))
            print("initial gripper: generic home pose (no recorded EE pose found)")
        if initial_ee:                                   # explicit mocap-position override
            self._ee_xyz = [float(v) for v in initial_ee.split(",")]

        self._roll = MuJoCoRollout(objs, table_pose=table_pose)  # temp xml (not committed)

    def run(self, plan, index: int, out_dir) -> str:
        res = self._roll.run(plan, initial_position=self._ee_xyz,
                             initial_orientation=self._ee_quat_wxyz,
                             z_min=self.table_top, default_duration=1.0, settle_steps=120,
                             render=True, render_camera=self.view, video=self.video)
        return res.save(Path(out_dir), index=index, instruction=self.instruction)
