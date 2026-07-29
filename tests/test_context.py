"""Tests for the action-proposal context builder (Phase 5, step 2).

Hermetic: synthesize trial dirs in tmp_path; no robot/GPU. One case parses a real
the original ``*_mujoco_cam{id}.txt`` fixture.
"""
from pathlib import Path

import numpy as np
import pytest

from simpact.generator.context import EEPose, build_context

FIXTURES = Path(__file__).parent / "fixtures"


# ---- EEPose ---------------------------------------------------------------- #
def test_eepose_from_matrix_and_xyzquat_agree():
    T = np.eye(4)
    T[:3, 3] = [0.3, -0.1, 0.4]
    ee = EEPose.from_matrix(T)
    assert np.allclose(ee.position, [0.3, -0.1, 0.4])
    assert abs(ee.yaw) < 1e-9  # identity rotation -> zero yaw
    ee2 = EEPose.from_xyz_quat([0.3, -0.1, 0.4], [0, 0, 0, 1])
    assert np.allclose(ee.quaternion_xyzw, ee2.quaternion_xyzw)


def test_eepose_from_file_4x4_and_7vec(tmp_path):
    m = tmp_path / "ee_mat.txt"
    np.savetxt(m, np.eye(4))
    assert np.allclose(EEPose.from_file(m).position, [0, 0, 0])

    v = tmp_path / "ee_vec.txt"
    np.savetxt(v, np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]]))
    ee = EEPose.from_file(v)
    assert np.allclose(ee.position, [0.1, 0.2, 0.3])

    bad = tmp_path / "ee_bad.txt"
    np.savetxt(bad, np.zeros(5))
    with pytest.raises(ValueError, match="4x4 matrix or 7-vector"):
        EEPose.from_file(bad)


def test_eepose_from_context_file(tmp_path):
    # parses the recorded context.txt EE lines ("x y z w" quaternion)
    ctx = tmp_path / "context.txt"
    ctx.write_text(
        "initial robot end effector position (x y z): 0.4859 -0.2317 0.2512\n"
        "initial robot end effector orientation (x y z w): 0.5484 0.5424 0.4448 -0.4552\n"
        "robot gripper max width: 0.1\n")
    ee = EEPose.from_context_file(ctx)
    assert np.allclose(ee.position, [0.4859, -0.2317, 0.2512])
    assert np.allclose(ee.quaternion_xyzw, [0.5484, 0.5424, 0.4448, -0.4552])
    # to_matrix round-trips through from_matrix
    assert np.allclose(EEPose.from_matrix(ee.to_matrix()).position, ee.position)

    with pytest.raises(ValueError, match="initial EE"):
        (tmp_path / "empty.txt").write_text("no pose here")
        EEPose.from_context_file(tmp_path / "empty.txt")


def _ee():
    return EEPose.from_xyz_quat([0.5, 0.0, 0.3], [0, 0, 0, 1])


# ---- rigid object (mujoco_cam txt) ----------------------------------------- #
def test_build_context_rigid(tmp_path):
    np.savetxt(tmp_path / "orange bottle_mujoco_cam1.txt", np.eye(4) * [1, 1, 1, 1])
    pose = np.eye(4)
    pose[:3, 3] = [0.4, 0.1, 0.2]
    np.savetxt(tmp_path / "orange bottle_mujoco_cam1.txt", pose)

    ctx = build_context("orange bottle.", tmp_path, "push", _ee(), cam_id=1)
    assert "{ee_pose}" not in ctx and "{object_poses}" not in ctx  # placeholders filled
    assert "initial robot end effector position (x y z): 0.5000 0.0000 0.3000" in ctx
    assert "orange bottle position (x y z): 0.4000 0.1000 0.2000" in ctx
    assert "orange bottle orientation (w x y z):" in ctx


def test_build_context_parses_real_mujoco_fixture(tmp_path):
    # use a recorded transform_6d output as the object pose
    pose = np.loadtxt(FIXTURES / "pringles_mujoco_cam1.txt").reshape(4, 4)
    np.savetxt(tmp_path / "pringles_mujoco_cam1.txt", pose)
    ctx = build_context("pringles", tmp_path, "push", _ee(), cam_id=1)
    x, y, z = pose[:3, 3]
    assert f"pringles position (x y z): {x:.4f} {y:.4f} {z:.4f}" in ctx


# ---- deformable (scene.yaml) ----------------------------------------------- #
def test_build_context_rope(tmp_path):
    (tmp_path / "scene.yaml").write_text(
        "fixed_point: [0.1, 0.2, 0.05]\nfree_end: [0.4, -0.1, 0.05]\n"
    )
    ctx = build_context("rope", tmp_path, "rope", _ee())
    assert "rope free end position (x y z): 0.4000 -0.1000 0.0500" in ctx
    assert "rope fixed end position (x y z): 0.1000 0.2000 0.0500" in ctx


def test_build_context_mpm(tmp_path):
    (tmp_path / "scene.yaml").write_text("init_mpm_center: [0.3, 0.0, 0.1]\n")
    ctx = build_context("blue playdoh", tmp_path, "dough", _ee())
    assert "blue playdoh center position (x y z): 0.3000 0.0000 0.1000" in ctx


def test_build_context_mpm_center_computed_from_cloud(tmp_path):
    # the centre must come from the actual particle cloud, NOT the (here deliberately
    # wrong) hardcoded init_mpm_center in scene.yaml
    pts = np.array([[0.40, -0.10, 0.14], [0.60, -0.10, 0.14], [0.50, 0.00, 0.16]])
    np.save(tmp_path / "mpm_points.npy", pts)  # mean = (0.5, -0.0667, 0.1467)
    (tmp_path / "scene.yaml").write_text(
        "init_mpm_center: [9.0, 9.0, 9.0]\nraw_pcd_path: mpm_points.npy\n")
    ctx = build_context("blue playdoh", tmp_path, "dough", _ee())
    assert "blue playdoh center position (x y z): 0.5000 -0.0667 0.1467" in ctx
    assert "9.0000" not in ctx  # the stale file value is not used


def test_build_context_mpm_center_falls_back_without_cloud(tmp_path):
    # no cloud on disk -> fall back to the recorded init_mpm_center
    (tmp_path / "scene.yaml").write_text(
        "init_mpm_center: [0.3, 0.0, 0.1]\nraw_pcd_path: missing.npy\n")
    ctx = build_context("blue playdoh", tmp_path, "dough", _ee())
    assert "blue playdoh center position (x y z): 0.3000 0.0000 0.1000" in ctx


def test_build_context_mpm_without_init_mpm_center(tmp_path):
    # minimal schema: no init_mpm_center at all -> centre computed from the cloud
    pts = np.array([[0.40, -0.10, 0.14], [0.60, -0.10, 0.14], [0.50, 0.00, 0.16]])
    np.save(tmp_path / "mpm_points.npy", pts)
    (tmp_path / "scene.yaml").write_text("raw_pcd_path: mpm_points.npy\n")
    ctx = build_context("blue playdoh", tmp_path, "dough", _ee())
    assert "blue playdoh center position (x y z): 0.5000 -0.0667 0.1467" in ctx


def test_build_context_mpm_center_missing_everything_errors(tmp_path):
    # no cloud AND no init_mpm_center fallback -> clear error, not a bare KeyError miss
    (tmp_path / "scene.yaml").write_text("raw_pcd_path: missing.npy\n")
    with pytest.raises(KeyError, match="cannot determine MPM centre"):
        build_context("blue playdoh", tmp_path, "dough", _ee())


def test_build_context_mpm_bg_pcd_relative(tmp_path):
    # sweep scenes carry a relative bg_pcd_path -> must resolve to the scene dir and
    # yield a real target-center line (not NaN from a failed open)
    o3d = pytest.importorskip("open3d")
    pts = np.array([[0.4, 0.05, 0.14], [0.5, 0.07, 0.14]])
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(tmp_path / "target_region.ply"), pc)
    (tmp_path / "scene.yaml").write_text(
        "init_mpm_center: [0.46, -0.11, 0.14]\nbg_pcd_path: target_region.ply\n")
    ctx = build_context("black bean pile", tmp_path, "sweep", _ee())
    assert "target center position (x y z): 0.4500 0.0600 0.1400" in ctx
    assert "nan" not in ctx.lower()


def test_missing_object_pose_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_context("ghost object", tmp_path, "push", _ee(), cam_id=1)


# ---- resolve_initial_ee: one runtime source, ordered fallbacks -------------- #

def _ee_matrix():
    T = np.eye(4)
    T[:3, 3] = [0.4, -0.1, 0.3]
    return T


def test_resolve_initial_ee_prefers_scene_yaml(tmp_path):
    from simpact.generator.context import resolve_initial_ee
    (tmp_path / "sim").mkdir(); (tmp_path / "capture").mkdir()
    T = _ee_matrix()
    (tmp_path / "sim" / "scene.yaml").write_text(
        "initial_ee_pose:\n" + "\n".join(
            "- [" + ", ".join(str(v) for v in row) + "]" for row in T))
    # a DIFFERENT capture record must lose to the runtime source
    np.savetxt(tmp_path / "capture" / "initial_ee_pose.txt", np.eye(4))
    ee, src = resolve_initial_ee(tmp_path)
    assert src.endswith("scene.yaml")
    assert np.allclose(ee.to_matrix(), T)


def test_resolve_initial_ee_txt_fallback(tmp_path):
    from simpact.generator.context import resolve_initial_ee
    np.savetxt(tmp_path / "initial_ee_pose.txt", _ee_matrix())
    ee, src = resolve_initial_ee(tmp_path)
    assert src.endswith("initial_ee_pose.txt")
    assert np.allclose(ee.to_matrix(), _ee_matrix())


def test_resolve_initial_ee_context_fallback(tmp_path):
    from simpact.generator.context import resolve_initial_ee
    (tmp_path / "context.txt").write_text(
        "initial robot end effector position (x y z): 0.4000 -0.1000 0.3000\n"
        "initial robot end effector orientation (x y z w): 0.0000 0.0000 0.0000 1.0000\n")
    ee, src = resolve_initial_ee(tmp_path)
    assert src.endswith("context.txt")
    assert np.allclose(ee.position, [0.4, -0.1, 0.3])


def test_resolve_initial_ee_missing_raises(tmp_path):
    from simpact.generator.context import resolve_initial_ee
    with pytest.raises(FileNotFoundError):
        resolve_initial_ee(tmp_path)
