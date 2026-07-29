"""Regression tests for the bundled-trial layout + minimal scene.yaml schema.

Each bundled example trial separates its files by role (simpact/utils/layout.py):
    <trial>/capture/  raw recording (RGB-D, initial_ee_pose.txt)
    <trial>/sim/      simulation assets (scene.yaml, point clouds)
    <trial>/runs/     recorded planning outputs (propose.json, refined_plan.json, ...)

Asserts that layout, that scene.yaml carries ONLY the load-bearing fields (removed
fields — init_gripper_pose everywhere; rope raw_pcd_path; init_mpm_center — stay
gone), and that the rollout/context loaders still consume the bundles.
See docs/DEFORMABLE_INTEGRATION.md field audit.
"""
import os

import numpy as np
import pytest
import yaml

ROPE_SCENES = ["examples/rope_real2sim/1102_rope_8",
               "examples/rope_real2sim/1102_rope_11"]
DOUGH = "examples/dough_real2sim/1104_sand_6"
SWEEP = "examples/sweep_real2sim/0118_sweep_0"
PUSH = "examples/push_real2sim/0103_push_0"
ALL_TRIALS = ROPE_SCENES + [DOUGH, SWEEP, PUSH]


def _load(scene):
    return yaml.safe_load(open(f"{scene}/sim/scene.yaml"))


def _has(scene):
    return os.path.exists(f"{scene}/sim/scene.yaml")


# --- trial layout: capture / sim / runs, nothing loose at the root -----------

@pytest.mark.parametrize("trial", ALL_TRIALS)
def test_trial_layout_roles(trial):
    """Files live under their role subdir — raw capture in capture/, sim assets in
    sim/, recorded outputs in runs/ — with nothing loose at the trial root (a local
    gitignored build/ from the perception pipeline is allowed)."""
    if not os.path.isdir(trial):
        pytest.skip(f"missing {trial}")
    entries = set(os.listdir(trial))
    assert entries <= {"capture", "sim", "runs", "build"}, \
        f"{trial} has loose entries: {entries - {'capture', 'sim', 'runs', 'build'}}"
    assert {"capture", "sim", "runs"} <= entries
    assert os.path.exists(f"{trial}/sim/scene.yaml")


# --- dead fields are gone ----------------------------------------------------

@pytest.mark.parametrize("scene", ROPE_SCENES + [DOUGH, SWEEP])
def test_no_dead_init_gripper_pose(scene):
    if not _has(scene):
        pytest.skip(f"missing {scene}")
    assert "init_gripper_pose" not in _load(scene), "init_gripper_pose is dead — remove it"


@pytest.mark.parametrize("scene", ROPE_SCENES)
def test_rope_schema_minimal(scene):
    if not _has(scene):
        pytest.skip(f"missing {scene}")
    y = _load(scene)
    assert set(y) <= {"fixed_point", "free_end", "endpoint_source", "endpoint_confidence",
                      "camera", "initial_ee_pose"}
    assert "fixed_point" in y and "free_end" in y
    # the old hardcoded /home/ydu raw_pcd_path must be gone (rope reads the ply directly)
    assert "raw_pcd_path" not in y
    # calibration by registry reference, not embedded cam files
    assert "camera" in y and y["camera"].get("profile")
    assert not os.path.exists(f"{scene}/cam1_K.txt")
    assert not os.path.exists(f"{scene}/sim/cam1_K.txt")


@pytest.mark.parametrize("scene", [DOUGH, SWEEP])
def test_mpm_schema_minimal(scene):
    if not _has(scene):
        pytest.skip(f"missing {scene}")
    y = _load(scene)
    assert "raw_pcd_path" in y and "initial_ee_pose" in y
    assert "init_mpm_center" not in y  # centre is computed live from the cloud
    assert np.asarray(y["initial_ee_pose"], float).shape == (4, 4)


def test_sweep_keeps_bg_pcd_path():
    if not _has(SWEEP):
        pytest.skip("missing sweep scene")
    assert "bg_pcd_path" in _load(SWEEP)


@pytest.mark.parametrize("scene", ALL_TRIALS)
def test_scene_yaml_is_the_runtime_ee_source(scene):
    """Every bundled sim/scene.yaml embeds initial_ee_pose (4x4): sim/ is
    self-sufficient — simulation never reads from capture/."""
    if not _has(scene):
        pytest.skip(f"missing {scene}")
    T = np.asarray(_load(scene)["initial_ee_pose"], float)
    assert T.shape == (4, 4)


@pytest.mark.parametrize("scene", ALL_TRIALS)
def test_example_calibration_via_registry(scene):
    """Examples reference the calibration registry (no embedded cam files) and resolve."""
    if not _has(scene):
        pytest.skip(f"missing {scene}")
    from simpact.real2sim.camera_calibration import load_camera
    cp = load_camera(scene, 1)
    assert cp.source.startswith("registry:"), f"{scene} should resolve via registry, got {cp.source}"
    assert cp.K.shape == (3, 3) and cp.cam_to_robot.shape == (4, 4)


@pytest.mark.parametrize("scene,cloud", [
    (ROPE_SCENES[0], "segmented_object.ply"), (ROPE_SCENES[1], "segmented_object.ply"),
    (DOUGH, "mpm_points.npy"), (SWEEP, "beans_mpm_points.npy"),
])
def test_example_cloud_projects_in_bounds(scene, cloud):
    """Geometric verification of the registry loading: each scene's own cloud must
    project into its 640x480 RGB through the RESOLVED (K, extrinsic). A wrong profile
    assignment (e.g. rope pointing at 0103) would push the cloud out of frame."""
    path = f"{scene}/sim/{cloud}"
    if not os.path.exists(path):
        pytest.skip(f"missing {path}")
    import open3d as o3d
    from simpact.real2sim.camera_calibration import load_camera
    from simpact.executor.render_deformable import project
    cp = load_camera(scene, 1)
    pts = (np.load(path) if cloud.endswith(".npy")
           else np.asarray(o3d.io.read_point_cloud(path).points))
    uv, front = project(pts, cp.K, cp.cam_to_robot)
    inb = front & (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
    assert inb.mean() > 0.95, f"{scene}: only {inb.mean()*100:.0f}% of cloud in-bounds — wrong calibration?"


# --- no hardcoded /home/* paths anywhere in the bundled scene.yamls ----------

@pytest.mark.parametrize("scene", ROPE_SCENES + [DOUGH, SWEEP])
def test_no_absolute_home_paths(scene):
    if not _has(scene):
        pytest.skip(f"missing {scene}")
    raw = open(f"{scene}/sim/scene.yaml").read()
    assert "/home/" not in raw, "scene.yaml still bakes an absolute /home path"


# --- loaders still consume the minimal files --------------------------------

def test_mpm_loader_reads_minimal_dough():
    if not _has(DOUGH):
        pytest.skip("missing dough scene")
    from simpact.executor.mpm_rollout import _load_mpm_scene
    object_name, init_T, pts, n_full, K, c2r, image_size = _load_mpm_scene(
        DOUGH, cam=1, downsample=None, seed=0)
    # loader returns the init EE transform + the particle cloud from raw_pcd_path
    assert init_T.shape == (4, 4)
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert object_name == "blue playdoh"
    assert tuple(image_size) == (640, 480)  # from the calibration profile


@pytest.mark.parametrize("scene", ALL_TRIALS)
def test_capture_ee_record_matches_runtime_source(scene):
    """capture/initial_ee_pose.txt is the raw provenance record; the runtime source
    (sim/scene.yaml initial_ee_pose) must equal it — and where a legacy-format
    context.txt exists, its EE lines must agree too."""
    from simpact.generator.context import EEPose
    p = f"{scene}/capture/initial_ee_pose.txt"
    if not os.path.exists(p):
        pytest.skip(f"missing {p}")
    T = np.loadtxt(p)
    assert T.shape == (4, 4)
    assert np.allclose(T, np.asarray(_load(scene)["initial_ee_pose"], float), atol=1e-6)
    ctx = f"{scene}/capture/context.txt"
    if os.path.exists(ctx):
        assert np.allclose(T, EEPose.from_context_file(ctx).to_matrix(), atol=1e-6)


def test_push_sim_carries_golden_reconstruction():
    """push sim/ is self-sufficient like every other task: the golden perception
    reconstruction (textured mesh + camera-frame 6-DoF + derived robot-frame pose
    per object) is committed, so planning needs no perception models."""
    import glob
    if not _has(PUSH):
        pytest.skip("missing push trial")
    poses = sorted(glob.glob(f"{PUSH}/sim/*_6d_cam1.txt"))
    assert len(poses) >= 2, "expected the two carton reconstructions"
    for p in poses:
        name = os.path.basename(p).replace("_6d_cam1.txt", "")
        assert os.path.exists(f"{PUSH}/sim/{name}_scaled.obj")
        assert os.path.exists(f"{PUSH}/sim/{name}_mujoco_cam1.txt")


def test_push_rollout_builds_from_committed_sim():
    """The push MuJoCo scene assembles straight from the committed trial (object
    discovery, aligned poses, EE from scene.yaml) — no perception build needed."""
    if not _has(PUSH):
        pytest.skip("missing push trial")
    from simpact.executor.push_scene import PushSceneRollout
    roll = PushSceneRollout(PUSH)
    assert len(roll.object_names) == 2
    assert roll.context_ee.position.shape == (3,)


def test_rope_rollout_is_sim_only(tmp_path):
    """A rollout must be constructible from sim/ alone — copy a trial, DELETE
    capture/, and build the ARAP rollout (EE pose + endpoints + cloud all resolve
    from sim/). Guards the 'simulation never reads capture/' contract."""
    import shutil
    if not os.path.exists(f"{ROPE_SCENES[0]}/sim/segmented_object.ply"):
        pytest.skip("missing rope cloud")
    dst = tmp_path / "trial"
    shutil.copytree(ROPE_SCENES[0], dst)
    shutil.rmtree(dst / "capture")
    from simpact.executor.rope_rollout import ARAPRollout
    roll = ARAPRollout(dst, video=False, device="cpu")
    assert roll.init_ee.shape == (3,)


def test_runs_dirs_have_uniform_propose_and_refined():
    """Each trial's runs/ carries the initial proposals (propose.json) AND the final
    refined plan (refined_plan.json) — the uniform per-run artifact contract — and both
    load as a ProposalSet. propose.json lives with the outputs, NOT in capture/ or sim/."""
    import glob
    from simpact.actions import ProposalSet
    runs_dirs = sorted(glob.glob("examples/*/*/runs"))
    if not runs_dirs:
        pytest.skip("no example runs dirs")
    for d in runs_dirs:
        assert os.path.exists(f"{d}/propose.json"), f"{d} missing propose.json"
        assert os.path.exists(f"{d}/refined_plan.json"), f"{d} missing refined_plan.json"
        assert len(ProposalSet.from_json(f"{d}/propose.json").action_proposals) > 0
    # and propose.json must NOT be in the observation/asset dirs
    assert not glob.glob("examples/*/*/capture/propose.json") \
        and not glob.glob("examples/*/*/sim/propose.json"), \
        "propose.json is an output artifact — it belongs in runs/, not capture/ or sim/"


def test_rope_rollout_reads_minimal_scene():
    if not os.path.exists(f"{ROPE_SCENES[0]}/sim/segmented_object.ply"):
        pytest.skip("missing rope cloud")
    from simpact.executor.rope_rollout import ARAPRollout
    roll = ARAPRollout(ROPE_SCENES[0], video=False, device="cpu")
    assert roll.fixed_pt.shape == (3,) and roll.free_end.shape == (3,)
