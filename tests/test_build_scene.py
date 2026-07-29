"""Tests for the offline scene builder (build_scene.py), fully mocked — no GPU/API.

A synthetic RGB-D bundle + a fake segmenter + a fake VLM exercise the whole
segment -> back-project -> (rope grounding | MPM sampling) -> scene.yaml path,
plus the EE-pose resolution policy (explicit / auto-discover / hard-error /
opt-in home-pose fallback).
"""
import json

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")
import yaml  # noqa: E402

from simpact.real2sim.build_scene import build_scene  # noqa: E402
from simpact.real2sim.perception.base import SegmentationResult  # noqa: E402

W = H = 240
FX = FY = 240.0
CX = CY = 120.0
DEPTH = 0.5


class FakeSegmenter:
    """Return a prompt-keyed boolean mask as a SegmentationResult."""

    def __init__(self, masks_by_prompt):
        self.masks_by_prompt = masks_by_prompt

    def segment(self, image, text_prompt):
        m = self.masks_by_prompt[text_prompt].astype(np.uint8)
        return SegmentationResult(masks=m[None], labels=[text_prompt],
                                  scores=np.array([0.9]), boxes=np.zeros((1, 4)))


def _fake_vlm(fixed="A", free="B"):
    def gen(contents):
        return json.dumps({"fixed": fixed, "free": free, "are_valid_tips": True,
                           "confidence": 0.9, "reasoning": "mock"})
    return gen


def _fake_material():
    """MPM material-ID VLM stub (build_scene queries it for the material: block)."""
    def gen(contents):
        return json.dumps({"softness": "soft", "E": 8000, "nu": 0.42,
                           "yield_stress": 1500, "density": 1200, "confidence": 0.8})
    return gen


def _u_mask():
    """A semicircular 'U' of pixels (a curved rope silhouette)."""
    mask = np.zeros((H, W), bool)
    theta = np.linspace(np.pi, 2 * np.pi, 400)
    u = (120 + 70 * np.cos(theta)).astype(int)
    v = (140 + 70 * np.sin(theta)).astype(int)
    mask[v, u] = True
    return mask


def _box_mask(u0, u1, v0, v1):
    mask = np.zeros((H, W), bool)
    mask[v0:v1, u0:u1] = True
    return mask


def _write_bundle(raw, *, masks_dummy=True, with_ee=True, cam_files=True):
    """Write a synthetic raw bundle: rgb/depth + K + identity extrinsic (+ EE pose)."""
    raw.mkdir(parents=True, exist_ok=True)
    rgb = np.full((H, W, 3), 128, np.uint8)
    np.save(raw / "camera1_rgb.npy", rgb)
    # depth filled everywhere; build_scene zeros it outside the mask before back-projecting
    np.save(raw / "camera1_depth.npy", np.full((H, W), DEPTH, np.float32))
    if cam_files:  # omit to force calibration resolution from a registry --profile
        np.savetxt(raw / "cam1_K.txt", np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]]))
        np.savetxt(raw / "cam1_to_robot.txt", np.eye(4))  # identity -> cloud == camera frame
    if with_ee:
        T = np.eye(4); T[:3, 3] = [0.5, -0.1, 0.35]
        np.savetxt(raw / "initial_ee_pose.txt", T)


# --- rope: grounding + context.txt ------------------------------------------

def test_build_rope_scene(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "scene"
    _write_bundle(raw)
    seg = FakeSegmenter({"rope": _u_mask()})
    res = build_scene(raw, out, "rope", "rope", cam=1,
                      segmenter=seg, generate_fn=_fake_vlm("A", "B"))
    # cloud + assets written
    assert (out / "segmented_object.ply").exists()
    assert (out / "camera1_rgb.png").exists()
    assert (out / "cam1_K.txt").exists() and (out / "cam1_to_robot.txt").exists()
    # scene.yaml carries VLM-grounded endpoints + the runtime EE source (minimal schema)
    y = yaml.safe_load((out / "scene.yaml").read_text())
    assert set(y) <= {"fixed_point", "free_end", "endpoint_source", "endpoint_confidence",
                      "initial_ee_pose"}
    assert np.asarray(y["initial_ee_pose"], float).shape == (4, 4)  # sim/ self-sufficient
    assert y["endpoint_source"] == "vlm"
    assert len(y["fixed_point"]) == 3 and len(y["free_end"]) == 3
    # the two endpoints are distinct (the two ends of the U)
    assert np.linalg.norm(np.array(y["fixed_point"]) - np.array(y["free_end"])) > 0.1
    # context.txt emitted from the resolved EE pose + endpoints
    ctx = (out / "context.txt").read_text()
    assert "initial robot end effector position (x y z): 0.5000 -0.1000 0.3500" in ctx
    assert "rope free end position" in ctx and "rope fixed end position" in ctx


def test_build_rope_role_swap_changes_endpoints(tmp_path):
    raw = tmp_path / "raw"; _write_bundle(raw)
    seg = FakeSegmenter({"rope": _u_mask()})
    a = build_scene(raw, tmp_path / "a", "rope", "rope", segmenter=seg, generate_fn=_fake_vlm("A", "B"))
    b = build_scene(raw, tmp_path / "b", "rope", "rope", segmenter=seg, generate_fn=_fake_vlm("B", "A"))
    ya = yaml.safe_load((a.scene_yaml).read_text())
    yb = yaml.safe_load((b.scene_yaml).read_text())
    # swapping the VLM roles swaps fixed/free
    assert np.allclose(ya["fixed_point"], yb["free_end"])
    assert np.allclose(ya["free_end"], yb["fixed_point"])


# --- MPM: dough + sweep -----------------------------------------------------

def test_build_dough_scene(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "scene"
    _write_bundle(raw)
    seg = FakeSegmenter({"blue playdoh": _box_mask(90, 150, 90, 150)})
    res = build_scene(raw, out, "dough", "blue playdoh", object_name="blue playdoh",
                      table_z=0.45, segmenter=seg, generate_fn=_fake_material())
    assert (out / "mpm_points.npy").exists()
    assert not (out / "context.txt").exists()  # MPM needs no context.txt
    pts = np.load(out / "mpm_points.npy")
    assert pts.ndim == 2 and pts.shape[1] == 3 and len(pts) > 0
    y = yaml.safe_load((out / "scene.yaml").read_text())
    assert y["object_name"] == "blue playdoh"
    assert y["raw_pcd_path"] == "mpm_points.npy"
    assert np.asarray(y["initial_ee_pose"]).shape == (4, 4)
    assert "init_mpm_center" not in y and "init_gripper_pose" not in y
    # VLM-estimated material block written per scene (physics inferred, not hardcoded)
    assert y["material"]["source"] == "vlm"
    assert set(("E", "nu", "yield_stress", "density")) <= set(y["material"])


def test_build_sweep_scene_adds_target_region(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "scene"
    _write_bundle(raw)
    seg = FakeSegmenter({"black bean pile": _box_mask(90, 150, 90, 150),
                         "purple tape cage": _box_mask(30, 70, 30, 70)})
    res = build_scene(raw, out, "sweep", "black bean pile", bg_prompt="purple tape cage",
                      object_name="black bean pile", table_z=0.45, segmenter=seg, generate_fn=_fake_material())
    assert (out / "target_region.ply").exists()
    y = yaml.safe_load((out / "scene.yaml").read_text())
    assert y["bg_pcd_path"] == "target_region.ply"


def test_sweep_without_bg_prompt_errors(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "scene"
    _write_bundle(raw)
    seg = FakeSegmenter({"beans": _box_mask(90, 150, 90, 150)})
    with pytest.raises(ValueError, match="sweep needs"):
        build_scene(raw, out, "sweep", "beans", table_z=0.45, segmenter=seg)


# --- EE-pose resolution policy ----------------------------------------------

def test_ee_pose_auto_discovered(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "scene"
    _write_bundle(raw, with_ee=True)  # writes initial_ee_pose.txt
    seg = FakeSegmenter({"blue playdoh": _box_mask(90, 150, 90, 150)})
    res = build_scene(raw, out, "dough", "blue playdoh", table_z=0.45, segmenter=seg, generate_fn=_fake_material())
    assert res.ee_source.endswith("initial_ee_pose.txt")


def test_ee_pose_explicit_path(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "scene"
    _write_bundle(raw, with_ee=False)
    ee_file = tmp_path / "myee.txt"
    T = np.eye(4); T[:3, 3] = [0.4, 0.0, 0.3]; np.savetxt(ee_file, T)
    seg = FakeSegmenter({"blue playdoh": _box_mask(90, 150, 90, 150)})
    res = build_scene(raw, out, "dough", "blue playdoh", table_z=0.45,
                      ee_pose_path=str(ee_file), segmenter=seg, generate_fn=_fake_material())
    y = yaml.safe_load((out / "scene.yaml").read_text())
    assert np.allclose(np.asarray(y["initial_ee_pose"])[:3, 3], [0.4, 0.0, 0.3])


def test_ee_pose_missing_hard_errors(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "scene"
    _write_bundle(raw, with_ee=False)  # no EE pose anywhere
    seg = FakeSegmenter({"blue playdoh": _box_mask(90, 150, 90, 150)})
    with pytest.raises(FileNotFoundError, match="no recorded EE pose"):
        build_scene(raw, out, "dough", "blue playdoh", table_z=0.45, segmenter=seg)


def test_build_reference_mode_vs_embed(tmp_path):
    seg = FakeSegmenter({"blue playdoh": _box_mask(90, 150, 90, 150)})
    # reference mode: --profile, raw bundle has NO cam files -> scene.yaml camera ref,
    # no embedded cam files in the output
    raw = tmp_path / "raw"; _write_bundle(raw, cam_files=False)
    r = build_scene(raw, tmp_path / "ref", "dough", "blue playdoh", table_z=0.45,
                    segmenter=seg, profile="1026", generate_fn=_fake_material())
    y = yaml.safe_load((r.scene_yaml).read_text())
    assert y["camera"] == {"profile": "1026", "cam": 1}
    assert not (tmp_path / "ref" / "cam1_K.txt").exists()
    assert not (tmp_path / "ref" / "cam1_to_robot.txt").exists()
    # embed mode: --embed-calibration -> cam files written, no camera ref
    r2 = build_scene(raw, tmp_path / "emb", "dough", "blue playdoh", table_z=0.45,
                     segmenter=seg, profile="1026", embed_calibration=True, generate_fn=_fake_material())
    assert (tmp_path / "emb" / "cam1_K.txt").exists()
    assert (tmp_path / "emb" / "cam1_to_robot.txt").exists()
    assert "camera" not in yaml.safe_load((r2.scene_yaml).read_text())


def test_calibration_profile_fallback(tmp_path):
    # a raw bundle with NO cam files -> build_scene pulls K + extrinsic from the registry
    from simpact.real2sim.build_scene import _load_extrinsic, _load_intrinsics
    from simpact.real2sim.camera_calibration import load_profile
    T = _load_extrinsic(tmp_path, 1, profile="0103")
    intr = _load_intrinsics(tmp_path, 1, profile="0103")
    assert T.shape == (4, 4)
    assert np.allclose(T, load_profile("0103", 1).cam_to_robot)
    assert intr["fx"] == pytest.approx(load_profile("0103", 1).K[0, 0])
    # no cam files and no profile -> clear error
    with pytest.raises(FileNotFoundError):
        _load_extrinsic(tmp_path, 1, profile=None)


def test_ee_pose_home_fallback_opt_in(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "scene"
    _write_bundle(raw, with_ee=False)
    seg = FakeSegmenter({"blue playdoh": _box_mask(90, 150, 90, 150)})
    res = build_scene(raw, out, "dough", "blue playdoh", table_z=0.45,
                      segmenter=seg, allow_home_pose=True, generate_fn=_fake_material())
    assert "FALLBACK" in res.ee_source
    assert any("home pose" in w for w in res.warnings)
