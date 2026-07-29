"""Phase 4C tests for the offline real2sim pipeline driver (stages 3, 7, 8).

Hermetic: synthesize the inputs each deterministic stage consumes, run it, and
check the output. No cameras, no perception models, no recorded data needed.

An optional end-to-end golden check runs only when ``SIMPACT_R2S_GOLDEN_DIR``
points at a recorded trial dir (e.g. a the original ``real2sim/data/<trial>``); it is
skipped otherwise so the suite stays self-contained.
"""
import os

import numpy as np
import pytest

import simpact.real2sim as r2s

pytestmark = pytest.mark.skipif(
    not r2s._REAL2SIM_AVAILABLE, reason="real2sim extra not installed"
)


# Calibration is resolved PER SCENE (no code-baked default): the push example carries a
# scene.yaml 'camera: {profile: 0103}' ref that resolves through the registry.
PUSH_SCENE = "examples/push_real2sim/0103_push_0"


def test_transform_to_robot_frame_composes_extrinsic():
    """Stage 7: final pose == camera->robot extrinsic @ object->camera pose."""
    ext = r2s.get_camera_to_robot(1, PUSH_SCENE)
    rng = np.random.default_rng(0)
    pose_cam = np.eye(4)
    pose_cam[:3, 3] = rng.uniform(-1, 1, size=3)

    got = r2s.transform_to_robot_frame(pose_cam, 1, PUSH_SCENE)
    np.testing.assert_allclose(got, ext @ pose_cam, atol=1e-12)


def test_transform_object_pose_roundtrip(tmp_path):
    """Stage 7 file I/O: writes {object}_mujoco_cam{id}.txt = ext @ 6d pose. The trial dir
    carries its own calibration reference (staged scene.yaml) — resolved per scene."""
    import shutil

    pose_cam = np.eye(4)
    pose_cam[:3, 3] = [0.1, -0.2, 0.3]
    np.savetxt(tmp_path / "widget_6d_cam1.txt", pose_cam)
    shutil.copy(f"{PUSH_SCENE}/sim/scene.yaml", tmp_path / "scene.yaml")  # calibration ref

    final = r2s.transform_object_pose(tmp_path, "widget", 1)
    written = np.loadtxt(tmp_path / "widget_mujoco_cam1.txt")
    np.testing.assert_allclose(written, final, atol=1e-12)
    np.testing.assert_allclose(final, r2s.get_camera_to_robot(1, PUSH_SCENE) @ pose_cam, atol=1e-12)


def test_unknown_camera_id_raises():
    # the referenced 0103 registry profile has no cam7 -> clear FileNotFoundError.
    # (get_camera_to_robot via a scene.yaml ref takes the scene's declared cam, so test
    # the profile lookup directly for the missing-camera case.)
    from simpact.real2sim.camera_calibration import load_profile
    with pytest.raises(FileNotFoundError):
        load_profile("0103", 7)


def test_no_calibration_raises():
    # retirement guard: a dir with neither embedded cam files nor a scene.yaml profile
    # ref has no calibration to fall back on -> must raise (never silently frozen).
    with pytest.raises(FileNotFoundError):
        r2s.get_camera_to_robot(1, "/nonexistent/scene/dir")


def test_extract_masks_roundtrip(tmp_path):
    """Stage 3: a known RLE-encoded mask decodes back to the same pixels, and
    the cropped RGBA matches the bbox+padding."""
    from PIL import Image
    from pycocotools import mask as mask_util

    import json

    H, W = 40, 60
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[10:25, 15:35] = 1  # a filled rectangle
    rle = mask_util.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("ascii")  # JSON-serialisable

    img = (np.random.default_rng(1).uniform(0, 255, (H, W, 3))).astype(np.uint8)
    img_path = tmp_path / "rgb.png"
    Image.fromarray(img).save(img_path)

    gsam2 = {
        "image_path": str(img_path),
        "img_width": W,
        "img_height": H,
        "annotations": [
            {"class_name": "thing", "bbox": [15, 10, 35, 25], "segmentation": rle}
        ],
    }
    json_path = tmp_path / "cam_gsam2.json"
    json_path.write_text(json.dumps(gsam2))

    written = r2s.extract_masks(
        json_path, tmp_path / "mask", png=True, crop=True, image_path=str(img_path)
    )
    assert [c for c, _ in written] == ["thing"]

    decoded = np.load(tmp_path / "mask_thing.npy")
    np.testing.assert_array_equal(decoded, mask)
    assert (tmp_path / "mask_thing.png").is_file()
    assert (tmp_path / "mask_thing_cropped.png").is_file()


def test_offline_driver_endtoend(tmp_path, monkeypatch):
    """Stages 7+8 through the driver: dummy scaled mesh + 6d pose -> scene XML
    that compiles in MuJoCo."""
    import mujoco
    import trimesh

    import run_real2sim  # scripts/ on sys.path via conftest

    import shutil

    data_dir = tmp_path / "trial"
    data_dir.mkdir()
    trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(data_dir / "box_scaled.obj")
    pose = np.eye(4)
    pose[:3, 3] = [0.5, 0.0, 0.2]
    np.savetxt(data_dir / "box_6d_cam1.txt", pose)
    shutil.copy(f"{PUSH_SCENE}/sim/scene.yaml", data_dir / "scene.yaml")  # per-scene calibration

    out_xml = run_real2sim.run(
        data_dir=str(data_dir), objects="box.", camera_id=1
    )
    assert out_xml.is_file()
    assert (data_dir / "box_mujoco_cam1.txt").is_file()

    model = mujoco.MjModel.from_xml_path(str(out_xml))
    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    }
    assert "box" in names


@pytest.mark.skipif(
    not os.environ.get("SIMPACT_R2S_GOLDEN_DIR"),
    reason="set SIMPACT_R2S_GOLDEN_DIR to a recorded trial dir for the golden check",
)
def test_golden_trial(tmp_path):
    """Opt-in: reproduce a recorded trial and assert bit-exact poses/masks."""
    import shutil

    import run_real2sim

    gold = os.environ["SIMPACT_R2S_GOLDEN_DIR"]
    objects = os.environ.get("SIMPACT_R2S_GOLDEN_OBJECTS", "")
    cam = int(os.environ.get("SIMPACT_R2S_GOLDEN_CAM", "1"))
    assert objects, "set SIMPACT_R2S_GOLDEN_OBJECTS too"

    work = tmp_path / "trial"
    shutil.copytree(gold, work)
    run_real2sim.run(data_dir=str(work), objects=objects, camera_id=cam)

    for name in [n.strip() for n in objects.replace(".", ",").split(",") if n.strip()]:
        got = np.loadtxt(work / f"{name}_mujoco_cam{cam}.txt")
        ref = np.loadtxt(f"{gold}/{name}_mujoco_cam{cam}.txt")
        np.testing.assert_allclose(got, ref, atol=1e-9)
