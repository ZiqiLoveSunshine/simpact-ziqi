"""Tests for the unified scene generator (simpact/real2sim/scene.py)."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
trimesh = pytest.importorskip("trimesh")

from simpact.real2sim.scene import build_mujoco_scene

GRIPPER_BODIES = ("gripper_mocap", "hand", "ee_center_body", "left_finger", "right_finger")


def test_perception_scene_has_no_gripper(tmp_path):
    xml = build_mujoco_scene([], tmp_path / "scene.xml", with_gripper=False)
    m = mujoco.MjModel.from_xml_path(xml)
    assert m.nmocap == 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_mocap") < 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table") >= 0


def test_rollout_scene_includes_controllable_gripper(tmp_path):
    xml = build_mujoco_scene([], tmp_path / "scene.xml", with_gripper=True)
    m = mujoco.MjModel.from_xml_path(xml)
    assert m.nmocap == 1
    for body in GRIPPER_BODIES:
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body) >= 0
    for act in ("left_finger", "right_finger"):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, act) >= 0


def test_objects_get_free_joints_and_simulate(tmp_path):
    mesh = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.05, 0.05, 0.1)).export(mesh)
    pose = np.eye(4)
    pose[:3, 3] = [0.45, 0.0, 0.05]
    obj = {"name": "red box", "mesh_file": str(mesh), "pose": pose, "rgba": (0.8, 0.2, 0.2)}

    for with_gripper in (False, True):
        xml = build_mujoco_scene([obj], tmp_path / f"s_{with_gripper}.xml", with_gripper=with_gripper)
        m = mujoco.MjModel.from_xml_path(xml)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "red_box")
        assert bid >= 0
        d = mujoco.MjData(m)
        if with_gripper:
            d.mocap_pos[0] = [0.4, 0.0, 0.35]
        for _ in range(50):
            mujoco.mj_step(m, d)
        assert np.isfinite(d.xpos[bid]).all()
