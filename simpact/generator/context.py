"""Scene-context builder for action proposals — ported from the original ``generate_context.py``.

Builds the ``{ee_pose}`` + ``{object_poses}`` text injected into a per-task
context template (``prompts/contexts/*.txt``), which the LLM proposer then feeds
to the model.

Decoupled from the robot: the original pipeline read the end-effector pose *live* from a Franka
(``franky``). Here the EE pose is a pluggable ``EEPose`` — from an explicit
value, a file (4x4 matrix or ``x y z qx qy qz qw``), the deformable
``scene.yaml``'s ``initial_ee_pose``, or (guarded, optional) a live robot. This
lets context be built offline from recorded trials, with no robot/GPU.

Object poses come from the same recorded-trial artifacts:
* rigid  -> ``{name}_mujoco_cam{cam}.txt`` (4x4 object->world)  [from transform_6d]
* rope   -> ``scene.yaml`` ``fixed_point`` / ``free_end``
* MPM    -> ``scene.yaml`` ``init_mpm_center`` (+ ``bg_pcd_path`` -> target center)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from simpact.generator.templates import load_context_template

# Defaults from the original context generator (the gripper geometry it advertises to the LLM).
DEFAULT_EE_DIMS = (0.025, 0.1, 0.04)
DEFAULT_GRIPPER_MAX_WIDTH = 0.1
# the original pipeline classified object kind by exact name; kept (overridable) for faithful output.
ROPE_NAMES = frozenset({"rope"})
MPM_NAMES = frozenset(
    {"green playdoh", "blue playdoh", "pink playdoh", "coffee beans", "black bean pile", "sand"}
)


@dataclass
class EEPose:
    """End-effector pose: position (m) + quaternion (x, y, z, w)."""

    position: np.ndarray  # (3,)
    quaternion_xyzw: np.ndarray  # (4,)

    @property
    def yaw(self) -> float:
        """Yaw (radians) from the xyzw quaternion (scipy 'xyz' euler), matching the real-robot convention."""
        return float(R.from_quat(self.quaternion_xyzw).as_euler("xyz")[2])

    @classmethod
    def from_xyz_quat(cls, position, quaternion_xyzw) -> "EEPose":
        return cls(
            np.asarray(position, dtype=float).reshape(3),
            np.asarray(quaternion_xyzw, dtype=float).reshape(4),
        )

    @classmethod
    def from_matrix(cls, T) -> "EEPose":
        T = np.asarray(T, dtype=float).reshape(4, 4)
        return cls(T[:3, 3].copy(), R.from_matrix(T[:3, :3]).as_quat())

    def to_matrix(self) -> np.ndarray:
        """4x4 homogeneous end-effector transform (world<-EE)."""
        T = np.eye(4)
        T[:3, :3] = R.from_quat(self.quaternion_xyzw).as_matrix()
        T[:3, 3] = self.position
        return T

    @classmethod
    def from_context_file(cls, path: Union[str, Path]) -> "EEPose":
        """Parse the real end-effector pose from a recorded ``context.txt``.

        Faithful to the original ``executor/push_6d.read_robot_state``: reads the lines
        ``initial robot end effector position (x y z): ...`` and ``... orientation
        (...): ...``. The rigid contexts label the quaternion ``x y z w`` and the
        rope contexts label it ``w x y z`` — both orderings are accepted here.
        """
        text = Path(path).read_text()
        pos_m = re.search(
            r"initial robot end effector position \(x y z\):\s*"
            r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", text)
        quat_m = re.search(
            r"initial robot end effector orientation \((x y z w|w x y z)\):\s*"
            r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", text)
        if not pos_m or not quat_m:
            raise ValueError(
                f"context file {str(path)!r}: could not find the initial EE "
                "position/orientation lines")
        position = [float(pos_m.group(i)) for i in (1, 2, 3)]
        q = [float(quat_m.group(i)) for i in (2, 3, 4, 5)]
        quat_xyzw = q if quat_m.group(1) == "x y z w" else [q[1], q[2], q[3], q[0]]
        return cls.from_xyz_quat(position, quat_xyzw)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "EEPose":
        """Load a 4x4 homogeneous matrix, a 16-vector, or ``x y z qx qy qz qw``."""
        arr = np.loadtxt(path)
        if arr.shape == (4, 4):
            return cls.from_matrix(arr)
        flat = np.asarray(arr).reshape(-1)
        if flat.size == 16:
            return cls.from_matrix(flat.reshape(4, 4))
        if flat.size == 7:
            return cls.from_xyz_quat(flat[:3], flat[3:7])
        raise ValueError(
            f"EE pose file {path!r}: expected a 4x4 matrix or 7-vector "
            f"(x y z qx qy qz qw), got {flat.size} values"
        )

    @classmethod
    def from_robot(cls, host: str, settle_s: float = 1.0) -> "EEPose":
        """Live read from a Franka via ``franky`` (optional dependency)."""
        try:
            from franky import Robot
        except ImportError as e:  # hardware optional — guard
            raise RuntimeError(
                "franky not installed; provide an explicit ee_pose / ee_pose_file "
                "to build context offline."
            ) from e
        import time

        robot = Robot(host)
        time.sleep(settle_s)  # stabilize readings
        st = robot.state
        return cls.from_xyz_quat(st.O_T_EE.translation, st.O_T_EE.quaternion)


def _ee_pose_block(ee: EEPose, ee_dims, gripper_max_width: float) -> str:
    t, q = ee.position, ee.quaternion_xyzw
    return (
        f"initial robot end effector position (x y z): {t[0]:.4f} {t[1]:.4f} {t[2]:.4f}\n"
        f"initial robot end effector orientation (x y z w): "
        f"{q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {q[3]:.4f}\n"
        f"initial robot end effector yaw (radians): {ee.yaw}\n"
        f"robot end effector dimensions (dx dy dz): {ee_dims[0]}, {ee_dims[1]}, {ee_dims[2]}\n"
        f"robot gripper max width: {gripper_max_width}\n"
    )


def _load_scene_yaml(data_dir: Path) -> Optional[dict]:
    from simpact.utils.layout import find_scene_file
    path = find_scene_file(data_dir, "scene.yaml", required=False) or (data_dir / "scene.yaml")
    if not path.exists():
        return None
    import yaml  # lazy (pyyaml is in the real2sim extra)

    return yaml.safe_load(path.read_text())


def _resolve_scene_path(raw: str, data_dir: Path) -> Path:
    """Resolve a scene.yaml path (bundled scenes use relative paths; legacy trials stored absolute
    ``/home/ydu/`` paths that don't exist here) to a real file under ``data_dir``."""
    from simpact.utils.layout import find_scene_file
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    return find_scene_file(data_dir, p.name, required=False) or (data_dir / p.name)


def resolve_initial_ee(scene_dir: Union[str, Path]) -> tuple[EEPose, str]:
    """Resolve the trial's initial end-effector pose from ONE runtime source.

    Resolution order:
      1. ``scene.yaml`` ``initial_ee_pose`` (4x4) — THE runtime source: ``sim/`` is
         self-sufficient for simulation, so every bundled trial carries the pose here.
      2. ``initial_ee_pose.txt`` — the raw capture record (``capture/`` or flat);
         provenance for 1, and the fallback for bundles built before it was embedded.
      3. ``context.txt`` EE lines — legacy external trials, which predate both files.

    Returns ``(EEPose, source_path)``; raises ``FileNotFoundError`` when none resolves.
    """
    from simpact.utils.layout import find_scene_file
    scene_dir = Path(scene_dir)
    scene = _load_scene_yaml(scene_dir)
    if scene and scene.get("initial_ee_pose") is not None:
        yml = find_scene_file(scene_dir, "scene.yaml", required=False) or (scene_dir / "scene.yaml")
        return EEPose.from_matrix(np.asarray(scene["initial_ee_pose"], float)), str(yml)
    p = find_scene_file(scene_dir, "initial_ee_pose.txt", required=False)
    if p is not None:
        return EEPose.from_matrix(np.loadtxt(p)), str(p)
    p = find_scene_file(scene_dir, "context.txt", required=False)
    if p is not None:
        return EEPose.from_context_file(p), str(p)
    raise FileNotFoundError(
        f"no initial EE pose under {scene_dir}: expected scene.yaml 'initial_ee_pose', "
        "an initial_ee_pose.txt, or a recorded-format context.txt")


def _mpm_cloud_center(scene: dict, data_dir: Path) -> np.ndarray:
    """Centroid of the actual MPM particle cloud (``raw_pcd_path``), computed live, so the
    VLM gets the true centre rather than a stale value from scene.yaml. Falls back to the
    optional ``init_mpm_center`` when the cloud file isn't available (e.g. hermetic
    tests). Supports ``.npy`` clouds (dough / beans) and ``.ply`` point clouds."""
    raw = scene.get("raw_pcd_path")
    if raw:
        path = _resolve_scene_path(raw, data_dir)
        if path.exists():
            try:
                if path.suffix == ".npy":
                    return np.load(path).mean(axis=0)
                import open3d as o3d  # lazy (open3d is in the real2sim extra)

                return np.asarray(o3d.io.read_point_cloud(str(path)).points).mean(axis=0)
            except Exception:
                pass  # unreadable cloud -> fall back to the recorded value
    fallback = scene.get("init_mpm_center")  # optional since the cloud is authoritative
    if fallback is not None:
        return np.asarray(fallback, dtype=float)
    raise KeyError(
        "cannot determine MPM centre: the raw_pcd_path cloud is unavailable and no "
        "init_mpm_center fallback is set in scene.yaml")


def _object_pose_block(name: str, data_dir: Path, cam_id: int, rope_names, mpm_names) -> str:
    from simpact.utils.layout import find_scene_file
    mujoco_txt = (find_scene_file(data_dir, f"{name}_mujoco_cam{cam_id}.txt", required=False)
                  or data_dir / f"{name}_mujoco_cam{cam_id}.txt")

    if name in rope_names:
        scene = _require_scene(data_dir, name)
        fixed = np.asarray(scene["fixed_point"], dtype=float)
        free = np.asarray(scene["free_end"], dtype=float)
        return (
            f"{name} free end position (x y z): {free[0]:.4f} {free[1]:.4f} {free[2]:.4f}\n"
            f"{name} fixed end position (x y z): {fixed[0]:.4f} {fixed[1]:.4f} {fixed[2]:.4f}\n"
        )

    if name in mpm_names:
        scene = _require_scene(data_dir, name)
        # Compute the CENTRE from the actual particle cloud, not the hardcoded
        # ``init_mpm_center`` in scene.yaml — so the VLM is told the true current dough /
        # bean centre (the file value can be stale, and this recomputes live each time the
        # context is built). MPM materials are a blob, so report centre of mass (not a
        # rope "free end"). Falls back to ``init_mpm_center`` if the cloud isn't on disk.
        c = _mpm_cloud_center(scene, data_dir)
        block = f"{name} center position (x y z): {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n"
        if scene.get("bg_pcd_path"):
            import open3d as o3d  # lazy (open3d is in the real2sim extra)

            bg_path = _resolve_scene_path(scene["bg_pcd_path"], data_dir)
            bg = np.asarray(o3d.io.read_point_cloud(str(bg_path)).points).mean(axis=0)
            block += f"target center position (x y z): {bg[0]:.4f} {bg[1]:.4f} {bg[2]:.4f}\n"
        return block

    # rigid object: 4x4 object->world pose from transform_6d
    if not mujoco_txt.exists():
        raise FileNotFoundError(
            f"no pose for object {name!r}: expected {mujoco_txt} (rigid) or a "
            f"scene.yaml entry (rope/MPM). Is the name/camera-id correct?"
        )
    pose = np.loadtxt(mujoco_txt).reshape(4, 4)
    pos = pose[:3, 3]
    qx, qy, qz, qw = R.from_matrix(pose[:3, :3]).as_quat()  # xyzw
    return (
        f"{name} position (x y z): {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n"
        f"{name} orientation (w x y z): {qw:.4f} {qx:.4f} {qy:.4f} {qz:.4f}\n"
    )


def _require_scene(data_dir: Path, name: str) -> dict:
    scene = _load_scene_yaml(data_dir)
    if scene is None:
        raise FileNotFoundError(f"object {name!r} needs {data_dir / 'scene.yaml'} (not found)")
    return scene


def build_context(
    object_string: str,
    data_dir: Union[str, Path],
    template: str,
    ee_pose: EEPose,
    cam_id: int = 1,
    rope_names=ROPE_NAMES,
    mpm_names=MPM_NAMES,
    ee_dims=DEFAULT_EE_DIMS,
    gripper_max_width: float = DEFAULT_GRIPPER_MAX_WIDTH,
) -> str:
    """Build the filled context text for ``object_string`` in ``data_dir``.

    ``template`` is a context-template name (e.g. ``"push"``) or path
    (resolved by ``load_context_template``). ``ee_pose`` is required — build it
    with ``EEPose.from_file`` / ``EEPose.from_robot`` / ``EEPose.from_xyz_quat``.
    """
    data_dir = Path(data_dir)
    names = [n.strip() for n in re.split("[.,]", object_string) if n.strip()]
    if not names:
        raise ValueError(f"no object names parsed from {object_string!r}")

    tmpl = load_context_template(template)
    ee_block = _ee_pose_block(ee_pose, ee_dims, gripper_max_width)
    object_block = "\n".join(
        _object_pose_block(n, data_dir, cam_id, rope_names, mpm_names) for n in names
    )
    # explicit replace (not str.format) so any literal braces in templates are safe
    return tmpl.replace("{ee_pose}", ee_block).replace("{object_poses}", object_block)
