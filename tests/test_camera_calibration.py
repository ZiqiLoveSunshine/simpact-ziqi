"""Tests for the camera-calibration resolver + registry (assets/calibration/<profile>/).

Covers the resolution order: embedded per-scene files > scene.yaml profile ref >
explicit profile arg > registry > clear error.
"""
import numpy as np
import pytest
import yaml

from simpact.real2sim.camera_calibration import (
    list_profiles,
    load_camera,
    load_profile,
)


def test_registry_profiles_present():
    profs = list_profiles()
    assert "1026" in profs and "0103" in profs


def test_load_profile_shapes_and_image_size():
    p = load_profile("1026", 1)
    assert p.K.shape == (3, 3) and p.cam_to_robot.shape == (4, 4)
    assert p.image_size == (640, 480)
    assert "1026" in p.source
    # 0103 carries both cameras
    assert load_profile("0103", 0).K.shape == (3, 3)
    assert load_profile("0103", 1).cam_to_robot.shape == (4, 4)


def test_load_profile_missing_camera_errors():
    with pytest.raises(FileNotFoundError, match="no cam9"):
        load_profile("1026", 9)


def test_embedded_scene_files_used_first(tmp_path):
    K = np.array([[100.0, 0, 50], [0, 100, 40], [0, 0, 1]])
    T = np.eye(4); T[:3, 3] = [0.1, 0.2, 0.3]
    np.savetxt(tmp_path / "cam1_K.txt", K)
    np.savetxt(tmp_path / "cam1_to_robot.txt", T)
    cp = load_camera(tmp_path, 1)
    assert cp.source.startswith("embedded")
    assert np.allclose(cp.K, K) and np.allclose(cp.cam_to_robot, T)


def test_scene_yaml_profile_reference(tmp_path):
    # no embedded cam files -> resolve via scene.yaml camera:{profile,cam}
    (tmp_path / "scene.yaml").write_text(yaml.dump({"camera": {"profile": "1026", "cam": 1}}))
    cp = load_camera(tmp_path, 1)
    assert cp.source == "registry:1026/cam1"
    assert np.allclose(cp.cam_to_robot, load_profile("1026", 1).cam_to_robot)


def test_embedded_wins_over_scene_yaml_ref(tmp_path):
    np.savetxt(tmp_path / "cam1_K.txt", np.eye(3))
    np.savetxt(tmp_path / "cam1_to_robot.txt", np.eye(4))
    (tmp_path / "scene.yaml").write_text(yaml.dump({"camera": {"profile": "0103", "cam": 1}}))
    assert load_camera(tmp_path, 1).source.startswith("embedded")


def test_explicit_profile_arg(tmp_path):
    cp = load_camera(tmp_path, 0, profile="0103")
    assert cp.source == "registry:0103/cam0"


def test_no_embedded_no_profile_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="no camera calibration"):
        load_camera(tmp_path, 1)
