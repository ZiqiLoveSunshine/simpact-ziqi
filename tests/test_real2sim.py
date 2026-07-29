"""Phase 4A functional tests for the real2sim geometry/assembly tier.

These exercise real behaviour (CoACD decomposition, MuJoCo XML load, SE(3)
round-trip) and require the ``real2sim`` extra. They skip cleanly if it's absent.
"""
import os

import numpy as np
import pytest

import simpact.real2sim as r2s

pytestmark = pytest.mark.skipif(
    not r2s._REAL2SIM_AVAILABLE,
    reason="real2sim extra not installed",
)


def test_packaged_assets_and_calibration_present():
    assert r2s.get_assets_dir().is_dir()
    assert r2s.get_calibration_dir().is_dir()
    gripper = r2s.get_assets_dir() / "franka_mujoco" / "franka_gripper.xml"
    assert gripper.is_file(), "franka_gripper.xml must ship with the package"


def test_convert_gripper_pose_roundtrip():
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(0)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.5, 1.1]).as_matrix()
    T[:3, 3] = rng.uniform(-1, 1, size=3)

    pos, quat = r2s.ee_pose_from_matrix(T)
    T_back = r2s.ee_pose_to_matrix(pos, quat)
    np.testing.assert_allclose(T_back, T, atol=1e-9)


def test_decompose_mesh_coacd_box(tmp_path):
    import trimesh

    box = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
    mesh_path = tmp_path / "box.obj"
    box.export(mesh_path)

    out_dir = tmp_path / "parts"
    part_paths = r2s.decompose_mesh_coacd(
        str(mesh_path), str(out_dir), threshold=0.1, max_convex_hull=4
    )
    assert part_paths, "CoACD should return at least one convex part"
    for p in part_paths:
        assert os.path.exists(p)


def test_generate_xml_loads_in_mujoco(tmp_path, monkeypatch):
    import mujoco
    import trimesh

    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # a scaled mesh + an identity-ish 4x4 pose for one object named "box"
    trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(data_dir / "box_scaled.obj")
    pose = np.eye(4)
    pose[:3, 3] = [0.5, 0.0, 0.2]
    np.savetxt(data_dir / "box_mujoco_cam0.txt", pose)

    out_xml = tmp_path / "scene.xml"
    r2s.create_mujoco_xml(
        object_string="box.", cam_id=0, data_dir="data", output_xml=str(out_xml)
    )
    assert out_xml.is_file()

    # the generated scene (with packaged franka gripper include) must load
    model = mujoco.MjModel.from_xml_path(str(out_xml))
    assert model.nbody > 1
    body_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    }
    assert "box" in body_names
