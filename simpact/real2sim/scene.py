"""Canonical MuJoCo scene generator for the rigid pipeline.

One builder for both uses:
* **perception output** (``with_gripper=False``) — objects (free-jointed mesh
  bodies) + table + ground + light + a front camera. This is what
  ``scripts/run_rigid_pipeline.py`` writes.
* **action-evaluation rollout** (``with_gripper=True``) — the same scene plus the
  mocap-controlled Franka gripper (``assets/franka_mujoco/franka_gripper.xml``,
  included by absolute path), driven by ``FloatingGripperController``
  (``gripper_mocap`` + ``left_finger``/``right_finger`` actuators).

(Distinct from ``real2sim/generate_xml.py``, the recorded-format generator with
data-dir-relative paths + textures used by the offline ``run_real2sim.py`` driver.)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from simpact.real2sim.paths import get_assets_dir

DEFAULT_GRIPPER_ASSET = "franka_mujoco/franka_gripper.xml"


def _pos_quat_wxyz(pose) -> tuple[np.ndarray, np.ndarray]:
    """(pos xyz, quat wxyz) from a 4x4 object->world matrix."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError(f"object pose must be 4x4; got shape {pose.shape}")
    qx, qy, qz, qw = R.from_matrix(pose[:3, :3]).as_quat()
    return pose[:3, 3], np.array([qw, qx, qy, qz])


def build_mujoco_scene(
    objects: Sequence[dict],
    xml_path: Union[str, Path],
    *,
    with_gripper: bool = False,
    with_table: bool = True,
    table_size: tuple[float, float, float] = (0.6, 0.6, 0.02),
    table_pose: Optional[tuple[float, float, float]] = None,
    gripper_asset: Optional[Union[str, Path]] = None,
    model_name: str = "push_real2sim_scene",
) -> str:
    """Write a MuJoCo scene; return the path.

    ``objects``: dicts with ``name``, ``mesh_file`` (.obj/.stl), ``pose`` (4x4
    object->world), and **either** ``texture`` (path to a 2D texture image, e.g.
    ``{obj}_scaled_0.png`` — rendered realistically via a material) **or** ``rgba``
    (flat colour). ``table_pose`` places the table at a fixed (x,y,z) (e.g. the original pipeline's
    ``(0.5243, -0.0009, 0.14)``); default auto-centres it under the objects.
    ``with_gripper`` adds the mocap Franka gripper for action-evaluation rollouts.
    Includes the original pipeline's ``top_view``/``side_view``/``front_view`` cameras.
    """
    assets, bodies, xs, ys = [], [], [], []
    for o in objects:
        s = str(o["name"]).replace(" ", "_")
        pos, quat = _pos_quat_wxyz(o["pose"])
        assets.append(f'    <mesh name="{s}_mesh" file="{Path(o["mesh_file"]).resolve()}"/>')
        tex = o.get("texture")
        if tex and Path(tex).exists():
            assets.append(f'    <texture name="{s}_tex" type="2d" file="{Path(tex).resolve()}"/>')
            assets.append(f'    <material name="{s}_mat" texture="{s}_tex" shininess="0.3" specular="0.75"/>')
            appearance = f'material="{s}_mat"'
        else:
            r, g, b = o.get("rgba", (0.6, 0.6, 0.6))
            appearance = f'rgba="{r:.3f} {g:.3f} {b:.3f} 1"'
        bodies.append(
            f'    <body name="{s}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
            f'quat="{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}">\n'
            f'      <freejoint/>\n'
            f'      <geom type="mesh" mesh="{s}_mesh" mass="0.3" '
            f'friction="0.3 0.005 0.0001" {appearance}/>\n'
            f'    </body>'
        )
        xs.append(pos[0]); ys.append(pos[1])

    cx = float(np.mean(xs)) if xs else 0.4
    cy = float(np.mean(ys)) if ys else 0.0

    include = ""
    if with_gripper:
        grip = Path(gripper_asset) if gripper_asset else get_assets_dir() / DEFAULT_GRIPPER_ASSET
        grip = grip.resolve()
        if not grip.exists():
            raise FileNotFoundError(f"gripper asset not found: {grip}")
        # absolute path so the gripper's own meshes resolve regardless of scene location
        include = f'  <include file="{grip}"/>\n'

    table = ""
    if with_table:
        tx, ty, tz = table_pose if table_pose is not None else (cx, cy, -0.02)
        table = (
            f'    <body name="table" pos="{tx:.4f} {ty:.4f} {tz:.4f}">\n'
            f'      <geom type="box" size="{table_size[0]} {table_size[1]} {table_size[2]}" '
            f'rgba="0.6 0.6 0.6 1" friction="0.3 0.005 0.0001"/>\n'
            f'    </body>\n'
        )

    xml = f"""<mujoco model="{model_name}">
  <compiler angle="degree" coordinate="local"/>
{include}  <option integrator="implicitfast" timestep="0.002" gravity="0 0 -9.81" cone="pyramidal"/>
  <asset>
{chr(10).join(assets)}
  </asset>
  <worldbody>
    <light diffuse=".7 .7 .7" pos="{cx:.3f} {cy:.3f} 2" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="3 3 0.1" rgba=".9 .9 .9 1" friction="0.3 0.005 0.0001"/>
{table}    <camera name="front" pos="{cx + 0.8:.3f} {cy:.3f} 0.5" xyaxes="0 -1 0 0.3 0 1"/>
    <camera name="top" pos="{cx:.3f} {cy:.3f} 1.2" xyaxes="1 0 0 0 1 0"/>
    <camera name="top_view" pos="0.365 -0.457 1.494" xyaxes="1.000 0.013 0.000 -0.012 0.943 0.334"/>
    <camera name="side_view" pos="0.228 -0.997 0.404" xyaxes="1.000 -0.031 0.000 0.005 0.178 0.984"/>
    <camera name="front_view" pos="1.593 0.044 0.662" xyaxes="0.009 1.000 -0.000 -0.334 0.003 0.943"/>
{chr(10).join(bodies)}
  </worldbody>
</mujoco>
"""
    xml_path = Path(xml_path)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(xml)
    return str(xml_path)
