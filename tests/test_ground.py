"""Tests for VLM rope-endpoint grounding (Phases 2-3), all mocked — no API/GPU.

Covers the camera-consistency guard (the "annotation shares the real camera params"
guarantee), label->3D mapping, VLM role validation, and the full detect->annotate->
map orchestration with a fake ``generate_fn``.
"""
import os

import numpy as np
import open3d as o3d
import pytest
import yaml

from simpact.executor.render_deformable import project
from simpact.generator.ground import (
    LABELS,
    _assert_camera_consistent,
    annotate_tips,
    assign_endpoint_roles,
    ground_rope_endpoints,
)

SCENE = "examples/rope_real2sim/1102_rope_8"


def _has_scene():
    return os.path.exists(f"{SCENE}/sim/segmented_object.ply") and os.path.exists(f"{SCENE}/capture/camera1_rgb.png")


skip_no_scene = pytest.mark.skipif(not _has_scene(), reason=f"missing example scene {SCENE}")


def _load_cam():
    from simpact.real2sim.camera_calibration import load_camera
    cp = load_camera(SCENE, 1)  # resolves via the scene.yaml camera profile ref
    cloud = np.asarray(o3d.io.read_point_cloud(f"{SCENE}/sim/segmented_object.ply").points)
    return cp.K, cp.cam_to_robot, cloud


def _fake_vlm(fixed="A", free="B", valid=True, conf=0.9):
    import json

    def gen(contents):
        # contents = [PIL image, prompt text]; ignore, return a fixed verdict
        assert len(contents) == 2
        return json.dumps({"fixed": fixed, "free": free, "are_valid_tips": valid,
                           "confidence": conf, "reasoning": "mock"})
    return gen


# --- camera-consistency guard -----------------------------------------------

@skip_no_scene
def test_camera_consistency_passes_on_real_scene():
    K, c2r, cloud = _load_cam()
    frac = _assert_camera_consistent((640, 480), K, c2r, cloud)
    assert frac >= 0.8


@skip_no_scene
def test_camera_consistency_rejects_wrong_resolution():
    # real K's principal point (~316,251) lies outside a downscaled 64x48 image
    K, c2r, _ = _load_cam()
    with pytest.raises(ValueError, match="principal point"):
        _assert_camera_consistent((64, 48), K, c2r, None)


@skip_no_scene
def test_camera_consistency_rejects_misaligned_cloud():
    # shift the cloud 10 m away -> it no longer projects into the image
    K, c2r, cloud = _load_cam()
    with pytest.raises(ValueError, match="only .* projects into"):
        _assert_camera_consistent((640, 480), K, c2r, cloud + np.array([10.0, 0, 0]))


# --- annotation (Phase 2) ---------------------------------------------------

@skip_no_scene
def test_annotate_maps_labels_to_xyz_and_pixels():
    K, c2r, cloud = _load_cam()
    tips = np.array([[0.40, 0.28, 0.15], [0.49, -0.16, 0.15]])
    img, mapping = annotate_tips(f"{SCENE}/capture/camera1_rgb.png", tips, K, c2r, cloud=cloud)
    assert img.size == (640, 480)
    assert set(mapping) == set(LABELS)
    # xyz behind each label is exactly the tip passed in, in order A,B
    assert np.allclose(mapping["A"]["xyz"], tips[0])
    assert np.allclose(mapping["B"]["xyz"], tips[1])
    # px matches an independent projection
    uv, _ = project(tips, K, c2r)
    assert np.allclose(mapping["A"]["px"], uv[0], atol=1e-6)
    assert np.allclose(mapping["B"]["px"], uv[1], atol=1e-6)


@skip_no_scene
def test_annotate_wrong_tip_count_raises():
    K, c2r, _ = _load_cam()
    with pytest.raises(ValueError, match="need 2 tips"):
        annotate_tips(f"{SCENE}/capture/camera1_rgb.png", np.zeros((3, 3)), K, c2r)


# --- role assignment (Phase 3) ----------------------------------------------

def _dummy_img():
    from PIL import Image
    return Image.new("RGB", (640, 480))


def test_assign_roles_valid():
    obj = assign_endpoint_roles(_dummy_img(), "ctx", generate_fn=_fake_vlm("A", "B"))
    assert obj["fixed"] == "A" and obj["free"] == "B"


def test_assign_roles_rejects_equal_labels():
    with pytest.raises(ValueError, match="same marker"):
        assign_endpoint_roles(_dummy_img(), "ctx", generate_fn=_fake_vlm("A", "A"))


def test_assign_roles_rejects_bad_label():
    with pytest.raises(ValueError, match="outside"):
        assign_endpoint_roles(_dummy_img(), "ctx", generate_fn=_fake_vlm("A", "Z"))


# --- orchestration (detect -> annotate -> map) ------------------------------

@skip_no_scene
def test_ground_end_to_end_mapping():
    # fixed=A -> fixed_point must equal detector tip_a; free=B -> tip_b
    from simpact.real2sim.detect_rope_endpoints import detect_from_ply
    det = detect_from_ply(f"{SCENE}/sim/segmented_object.ply")
    r = ground_rope_endpoints(SCENE, generate_fn=_fake_vlm("A", "B", conf=0.8),
                              write=False, save_annotated=False)
    assert np.allclose(r.fixed_point, det.tip_a)
    assert np.allclose(r.free_end, det.tip_b)
    # swapping the roles swaps the 3-D points
    r2 = ground_rope_endpoints(SCENE, generate_fn=_fake_vlm("B", "A", conf=0.8),
                               write=False, save_annotated=False)
    assert np.allclose(r2.fixed_point, det.tip_b)
    assert np.allclose(r2.free_end, det.tip_a)
    # combined confidence = detection * vlm
    assert r.confidence == pytest.approx(det.confidence * 0.8)


@skip_no_scene
def test_ground_flags_low_confidence_and_invalid(tmp_path):
    r = ground_rope_endpoints(SCENE, generate_fn=_fake_vlm(valid=False, conf=0.3),
                              write=False, save_annotated=False)
    assert any("not on distinct" in w for w in r.warnings)
    assert any("low VLM role confidence" in w for w in r.warnings)


@skip_no_scene
def test_ground_write_scene_yaml(tmp_path):
    import shutil
    dst = tmp_path / "scene"
    shutil.copytree(SCENE, dst)
    r = ground_rope_endpoints(str(dst), generate_fn=_fake_vlm("A", "B", conf=0.9),
                              write=True, save_annotated=False)
    y = yaml.safe_load((dst / "sim" / "scene.yaml").read_text())
    assert y["endpoint_source"] == "vlm"
    assert np.allclose(y["fixed_point"], r.fixed_point)
    assert np.allclose(y["free_end"], r.free_end)
    assert y["endpoint_confidence"] == pytest.approx(r.confidence)
