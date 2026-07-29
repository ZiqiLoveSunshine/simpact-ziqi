"""Tests for the MPM dough/sand rollout driver (Phase 5 deformable slice).

Hermetic parts (plan->transform, rollout parser, headless render) run without a GPU;
the full ``MPMRollout.run`` physics test is gated on CUDA + warp being importable.
"""
import json
import os

import numpy as np
import pytest

from simpact.actions import ActionProposal, Descend, GripperControl, Grasp, Move, Push, Release, Rotate
from simpact.executor.mpm_rollout import (
    grasps_from_plan,
    sweep_segments_from_plan,
    SWEEP_GRASP_HEIGHT,
)
from simpact.executor.render_deformable import project, render_deformable
from simpact.generator.regress import parse_mpm_rollout
from simpact.generator.verify import coverage_gate

SCENE = "examples/dough_real2sim/1104_sand_6"


def _has_cuda_warp():
    try:
        import torch
        import warp  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


# --- plan -> ordered grasp list (multi-grasp is the only dough path) ---------

def test_grasps_from_plan_move_pairs():
    init = np.eye(4); init[0, 3] = 0.5; init[1, 3] = -0.04
    # two squeezes: each move (cumulative) then a closing gripper_control
    plan = ActionProposal([
        Move(-0.02, 0.0, 0.0, 0.0, 0.0, 0.0), GripperControl(0.03),
        Move(0.04, 0.0, 0.0, 0.0, 0.0, 0.2), GripperControl(0.025)])
    grasps = grasps_from_plan(plan, init, init_width=0.15)
    assert len(grasps) == 2
    (c0, y0, w0), (c1, y1, w1) = grasps
    assert c0[0] == pytest.approx(0.48) and c1[0] == pytest.approx(0.52)  # cumulative
    assert w0 == pytest.approx(0.03) and w1 == pytest.approx(0.025)
    assert y1 == pytest.approx(0.2, abs=1e-6)  # yaw accumulates


def test_grasps_from_plan_propose_prims_and_release():
    # PUSH/ROTATE/GRASP squeezes; a trailing RELEASE (open) is not a squeeze
    plan = ActionProposal([Push(0.1, 0.0), Grasp(0.03), Push(0.05, 0.0), Grasp(0.02), Release()])
    grasps = grasps_from_plan(plan, np.eye(4), init_width=0.15)
    assert len(grasps) == 2
    assert grasps[0][0][0] == pytest.approx(0.1) and grasps[1][0][0] == pytest.approx(0.15)


def test_grasps_from_plan_requires_a_squeeze():
    with pytest.raises(ValueError):
        grasps_from_plan(ActionProposal([Move(0.1, 0.0, 0.0, 0.0, 0.0, 0.0)]), np.eye(4), 0.15)


# --- sweep: plan -> pusher segments ------------------------------------------

def test_sweep_segments_descend_then_push():
    init = np.eye(4); init[0, 3] = 0.47; init[1, 3] = -0.25; init[2, 3] = 0.44
    plan = ActionProposal([Descend(0.3), Push(0.0, 0.32)])  # descend clamps to grasp_height
    segs = sweep_segments_from_plan(plan, init)
    assert len(segs) == 2
    (s0, y0, d0), (s1, y1, d1) = segs
    assert s0[2] == pytest.approx(0.44)          # descend starts at init height
    assert d0[2] == pytest.approx(SWEEP_GRASP_HEIGHT - 0.44)  # clamped to the floor
    assert s1[2] == pytest.approx(SWEEP_GRASP_HEIGHT)  # push happens at table level
    assert d1[1] == pytest.approx(0.32) and d1[2] == 0.0


def test_sweep_segments_push_floors_to_grasp_height():
    # a bare PUSH (no descend) still sweeps at table level, not 44 cm up
    init = np.eye(4); init[2, 3] = 0.44
    segs = sweep_segments_from_plan(ActionProposal([Push(0.0, 0.2)]), init)
    assert segs[0][0][2] == pytest.approx(SWEEP_GRASP_HEIGHT)


def test_sweep_segments_requires_a_move():
    with pytest.raises(ValueError):
        sweep_segments_from_plan(ActionProposal([Rotate(0.5)]), np.eye(4))


# --- coverage gate (the first measured deformable gate) ----------------------

def _write_sweep_rollout(dirpath, final_xy):
    dirpath.mkdir(parents=True, exist_ok=True)
    pts = np.zeros((len(final_xy), 3)); pts[:, :2] = final_xy
    np.save(dirpath / "rollout_00_final_points.npy", pts.astype(np.float32))
    (dirpath / "rollout_00.json").write_text(json.dumps(
        {"mpm": {"final_points_path": "rollout_00_final_points.npy"}}))
    return dirpath / "rollout_00.json"


def test_coverage_gate_measures_fraction_inside(tmp_path):
    # target is the unit square [0,1]^2; 3 of 4 particles inside -> 75%
    target = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    gate = coverage_gate(target, min_coverage=0.5)
    rj = _write_sweep_rollout(tmp_path, [[0.5, 0.5], [0.2, 0.2], [0.8, 0.8], [5.0, 5.0]])
    passed, detail = gate(rj)
    assert passed and "75%" in detail


def test_coverage_gate_fails_below_threshold(tmp_path):
    target = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    gate = coverage_gate(target, min_coverage=0.5)
    rj = _write_sweep_rollout(tmp_path, [[0.5, 0.5], [9, 9], [9, 9], [9, 9]])  # 25%
    passed, detail = gate(rj)
    assert not passed and "25%" in detail


# --- rollout parser ----------------------------------------------------------

def _write_mpm_rollout(dirpath, verdict=None):
    from PIL import Image
    dirpath.mkdir(parents=True, exist_ok=True)
    png = "rollout_00_1.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(dirpath / png)
    data = {
        "timestamp": "20260706_000000", "object_names": ["blue playdoh"],
        "grasp_centers": [[0.5, -0.04, 0.16]], "grasp_yaws": [0.0], "grasp_widths": [0.03],
        "mpm": {"final_points_path": "rollout_00_final_points.npy",
                "bbox_min": [0.45, -0.06, 0.13], "bbox_max": [0.55, -0.02, 0.17],
                "bbox_size": [0.10, 0.04, 0.04], "centroid": [0.50, -0.04, 0.15]},
        "snapshots": [
            {"waypoint_index": 0, "gripper": {"position": [0.5, -0.04, 0.16], "width": 0.08},
             "objects": {"blue playdoh": {"position": [0.5, -0.04, 0.16]}}, "screenshot": "rollout_00_0.png"},
            {"waypoint_index": 1, "gripper": {"position": [0.5, -0.04, 0.16], "width": 0.03},
             "objects": {"blue playdoh": {"position": [0.50, -0.04, 0.15]}}, "screenshot": png},
        ],
    }
    if verdict is not None:
        data["verdict"] = verdict
    p = dirpath / "rollout_00.json"
    p.write_text(json.dumps(data))
    return p


def test_parse_mpm_rollout(tmp_path):
    # a single squeeze is N=1 in the unified (plural) envelope
    out = parse_mpm_rollout(_write_mpm_rollout(tmp_path))
    assert "Number of squeezes: 1" in out["text"]
    assert "squeeze 0: center [0.5, -0.04, 0.16], yaw 0.0, width 0.03" in out["text"]
    assert "Final dough bounding-box size (dx,dy,dz): [0.1, 0.04, 0.04]" in out["text"]
    assert "Final dough centroid (x,y,z): [0.5, -0.04, 0.15]" in out["text"]
    assert out["after_image"].endswith("rollout_00_1.png")


def test_parse_mpm_rollout_multi_step(tmp_path):
    from PIL import Image
    tmp_path.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / "rollout_00_1.png")
    data = {
        "timestamp": "t", "object_names": ["blue playdoh"],
        "grasp_centers": [[0.49, -0.04, 0.16], [0.53, -0.04, 0.16]],
        "grasp_yaws": [0.0, 0.2], "grasp_widths": [0.03, 0.025],
        "mpm": {"bbox_size": [0.1, 0.03, 0.03], "centroid": [0.51, -0.04, 0.15]},
        "snapshots": [{"screenshot": "rollout_00_1.png", "objects": {}}],
    }
    p = tmp_path / "rollout_00.json"; p.write_text(json.dumps(data))
    out = parse_mpm_rollout(p)
    assert "Number of squeezes: 2" in out["text"]
    assert "squeeze 0: center [0.49, -0.04, 0.16], yaw 0.0, width 0.03" in out["text"]
    assert "squeeze 1: center [0.53, -0.04, 0.16], yaw 0.2, width 0.025" in out["text"]


def test_parse_mpm_rollout_sweep(tmp_path):
    from PIL import Image
    tmp_path.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / "rollout_00_1.png")
    data = {
        "timestamp": "t", "object_names": ["black bean pile"],
        "sweep_segments": [{"start": [0.47, -0.25, 0.217], "yaw": 1.57, "delta": [0.0, 0.32, 0.0]}],
        "mpm": {"bbox_size": [0.1, 0.1, 0.02], "centroid": [0.46, 0.06, 0.14]},
        "snapshots": [{"screenshot": "rollout_00_1.png", "objects": {}}],
    }
    p = tmp_path / "rollout_00.json"; p.write_text(json.dumps(data))
    out = parse_mpm_rollout(p)
    assert "Number of sweep segments: 1" in out["text"]
    assert "push delta [0.0, 0.32, 0.0]" in out["text"]


def test_parse_mpm_rollout_surfaces_verdict(tmp_path):
    v = {"success": False, "reason": "too thick", "remaining": "close to 0.02"}
    out = parse_mpm_rollout(_write_mpm_rollout(tmp_path, verdict=v))
    assert "VERIFIER OUTCOME: FAILURE — too thick" in out["text"]
    assert "STILL NEEDED: close to 0.02" in out["text"]


# --- headless render ---------------------------------------------------------

def test_render_deformable_writes_png(tmp_path):
    pts = np.random.default_rng(0).normal(0.5, 0.02, (200, 3))
    K = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    ext = np.eye(4); ext[2, 3] = -0.7  # camera 0.7 m in front along z
    p = render_deformable(pts, K, ext, tmp_path / "r.png",
                          tool_boxes=[([0.5, 0.5, 0.5], [0, 0, 0, 1], [0.06, 0.02, 0.1])])
    assert os.path.exists(p) and os.path.getsize(p) > 0


def test_render_deformable_returns_array():
    from simpact.executor.render_deformable import render_deformable as rd
    K = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    ext = np.eye(4); ext[2, 3] = -0.7
    frame = rd(np.random.rand(100, 3) * 0.1 + 0.45, K, ext, return_array=True)
    assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8


def test_video_recorder_writes_mp4(tmp_path):
    from simpact.executor.render_deformable import VideoRecorder
    rec = VideoRecorder(tmp_path / "v.mp4", fps=10, img_size=(320, 240))
    for i in range(6):
        rec.add(np.full((240, 320, 3), i * 30, np.uint8))
    p = rec.save()
    assert p and os.path.exists(p) and os.path.getsize(p) > 0


def test_video_recorder_empty_is_noop(tmp_path):
    from simpact.executor.render_deformable import VideoRecorder
    assert VideoRecorder(tmp_path / "none.mp4").save() is None


def test_project_shapes():
    pts = np.array([[0.0, 0.0, 0.7]])
    K = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    uv, front = project(pts, K, np.eye(4))
    assert uv.shape == (1, 2) and front.shape == (1,)


# --- full physics rollout (GPU-gated) ----------------------------------------

@pytest.mark.skipif(not _has_cuda_warp(), reason="MPM rollout needs CUDA + warp")
def test_mpm_rollout_single_grasp(tmp_path):
    # the unified MPMRollout handles N=1 (a single squeeze is the degenerate case)
    from simpact.executor.mpm_rollout import MPMRollout
    roll = MPMRollout(SCENE, num_steps=60, downsample=4000)
    cx, cy = roll.init_pts.mean(0)[:2]
    ex, ey = roll.init_T[0, 3], roll.init_T[1, 3]
    plan = ActionProposal([Move(cx - ex, cy - ey, 0.0, 0.0, 0.0, 0.0), GripperControl(0.03)])
    path = roll.run(plan, 0, tmp_path)
    data = json.loads(open(path).read())
    assert data["object_names"] == ["blue playdoh"]
    assert data["mpm"]["num_grasps"] == 1 and len(data["grasp_centers"]) == 1
    for f in ("rollout_00_0.png", "rollout_00_1.png", "rollout_00_final_points.npy", "rollout_00.mp4"):
        assert (tmp_path / f).exists()
    final = np.load(tmp_path / "rollout_00_final_points.npy")
    assert len(final) == 4000 and not np.allclose(final.mean(0), roll.init_pts.mean(0))


@pytest.mark.skipif(not _has_cuda_warp(), reason="MPM rollout needs CUDA + warp")
def test_mpm_rollout_multi_grasp(tmp_path):
    from simpact.executor.mpm_rollout import MPMRollout
    roll = MPMRollout(SCENE, num_steps=50, downsample=4000)
    cx, cy = roll.init_pts.mean(0)[:2]
    ex, ey = roll.init_T[0, 3], roll.init_T[1, 3]
    # two squeezes centred on the dough, offset in x — one continuous sim
    plan = ActionProposal([
        Move(cx - ex - 0.02, cy - ey, 0.0, 0.0, 0.0, 0.0), GripperControl(0.03),
        Move(0.04, 0.0, 0.0, 0.0, 0.0, 0.0), GripperControl(0.03)])
    path = roll.run(plan, 0, tmp_path)
    data = json.loads(open(path).read())
    assert data["mpm"]["num_grasps"] == 2 and len(data["grasp_centers"]) == 2
    for f in ("rollout_00_0.png", "rollout_00_1.png", "rollout_00_final_points.npy", "rollout_00.mp4"):
        assert (tmp_path / f).exists()
    final = np.load(tmp_path / "rollout_00_final_points.npy")
    assert not np.allclose(final.mean(0), roll.init_pts.mean(0))


SWEEP_SCENE = "examples/sweep_real2sim/0118_sweep_0"


@pytest.mark.skipif(not _has_cuda_warp(), reason="MPM rollout needs CUDA + warp")
def test_sweep_rollout_end_to_end_and_gate(tmp_path):
    from simpact.executor.mpm_rollout import SweepRollout
    roll = SweepRollout(SWEEP_SCENE, num_steps=80, downsample=6000)
    # descend to the table, then sweep the pile +y into the target region
    plan = ActionProposal([Descend(0.25), Push(0.0, 0.32)])
    path = roll.run(plan, 0, tmp_path)
    data = json.loads(open(path).read())
    assert data["mpm"]["num_segments"] == 2
    for f in ("rollout_00_0.png", "rollout_00_1.png", "rollout_00_final_points.npy", "rollout_00.mp4"):
        assert (tmp_path / f).exists()
    # the pile moved toward the target (+y) and the measured gate reads high coverage
    final = np.load(tmp_path / "rollout_00_final_points.npy")
    assert final[:, 1].mean() > roll.init_pts[:, 1].mean() + 0.05
    passed, detail = coverage_gate(f"{SWEEP_SCENE}/sim/target_region.ply", min_coverage=0.5)(path)
    assert passed, detail


# --- material-ID outcome A/B: VLM-estimated vs old default (GPU-gated) --------
# The pre-VLM recipes removed in 798fe47 (identical solver config, only the physics
# params differ from the committed VLM material). Overriding just these keys on the
# scene's VLM material reproduces the old default arm exactly.
_OLD_DEFAULT = {
    "dough": {"E": 5000.0, "nu": 0.4, "yield_stress": 1000.0, "density": 1200.0},
    "sweep": {"E": 10000.0, "nu": 0.4, "yield_stress": 4000.0, "density": 600.0},
}
DOUGH_PLAN = "examples/dough_real2sim/1104_sand_6/runs/refined_plan.json"
SWEEP_PLAN = "examples/sweep_real2sim/0118_sweep_0/runs/refined_plan.json"


def _final_plan(path):
    """The single selected action (best_proposalset == ProposalSet([best_plan]))."""
    from simpact.actions import ProposalSet
    return ProposalSet.from_json(path).action_proposals[0]


def _bbox_size(pts):
    return pts.max(0) - pts.min(0)


@pytest.mark.parametrize("task,scene,plan_path,Roller,num_steps", [
    ("dough", SCENE, DOUGH_PLAN, "MPMRollout", 120),
    ("sweep", SWEEP_SCENE, SWEEP_PLAN, "SweepRollout", 80),
])
@pytest.mark.skipif(not _has_cuda_warp(), reason="MPM rollout needs CUDA + warp")
def test_material_id_changes_outcome(task, scene, plan_path, Roller, num_steps, tmp_path):
    """Material-ID must be load-bearing: running the committed final plan with the
    VLM-estimated material vs the old hand-tuned default (same plan, same seed, same
    downsampled cloud — the ONLY difference is the physics params) must yield a
    measurably different final shape. If it did not, VLM material-ID would be a silent
    no-op. This is the outcome-closure A/B for §15.1 in regression form (no sweep)."""
    if not os.path.exists(plan_path):
        pytest.skip(f"missing {plan_path}")
    import importlib

    from simpact.generator.material import load_material

    Roll = getattr(importlib.import_module("simpact.executor.mpm_rollout"), Roller)
    plan = _final_plan(plan_path)
    vlm = load_material(scene, task)
    default = {**vlm, **_OLD_DEFAULT[task]}
    assert vlm["E"] != default["E"], "VLM material must differ from the old default"

    def outcome(mat, sub):
        roll = Roll(scene, material_params=mat, downsample=4000, num_steps=num_steps,
                    video=False, seed=0)
        roll.run(plan, 0, tmp_path / sub)
        return np.load(tmp_path / sub / "rollout_00_final_points.npy")

    a = outcome(vlm, "vlm")
    b = outcome(default, "default")
    # aggregate bbox-size delta is robust to per-particle atomic-add jitter (which is
    # << 1 mm), while the material effect (§15.1: ~25% height) is several mm.
    dbbox = np.abs(_bbox_size(a) - _bbox_size(b))
    assert dbbox.max() > 1e-3, (
        f"{task}: material-ID had no measurable effect on the outcome "
        f"(Δbbox_m={dbbox.tolist()}, VLM E={vlm['E']} vs default E={default['E']})")
