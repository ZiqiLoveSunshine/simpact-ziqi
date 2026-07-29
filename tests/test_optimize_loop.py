"""Tests for the closed-loop optimizer (stubbed regress/verify/rollout, no VLM/sim)."""
import json

from simpact.generator.optimize_loop import OptimizationLoop
from simpact.generator.regress import RegressOptimizer
from simpact.generator.verify import TaskVerifier
from simpact.actions import ActionProposal, Move


def _write_rollout(path, index, moved=0.0):
    data = {
        "timestamp": f"T{index}", "object_names": ["carton"], "waypoints": [],
        "snapshots": [
            {"waypoint_index": 0, "gripper": {"position": [0, 0, 0.4],
             "orientation": [0, 1, 0, 0], "width": 0.0},
             "objects": {"carton": {"position": [0.0, 0, 0.2], "orientation": [1, 0, 0, 0]}},
             "screenshot": None},
            {"waypoint_index": 1, "gripper": {"position": [0.1, 0, 0.4],
             "orientation": [0, 1, 0, 0], "width": 0.0},
             "objects": {"carton": {"position": [moved, 0, 0.2], "orientation": [1, 0, 0, 0]}},
             "screenshot": None},
        ],
    }
    path.write_text(json.dumps(data))
    return path


def _candidate():
    return ActionProposal([Move(0.05, 0.0, 0.0, 0.0, 0.0, 0.0)], description="cand")


def _refined_plan_json():
    return json.dumps({"action_proposals": [{"description": "refined",
        "action_sequence": [{"type": "move", "delta_x": -0.08, "delta_y": 0.0, "delta_z": 0.0,
                             "delta_roll": 0.0, "delta_pitch": 0.0, "delta_yaw": 0.0}]}]})


def _loop(tmp_path, verify_seq, *, max_iters=5):
    """Build a loop whose verifier returns success per the verify_seq list."""
    calls = {"verify": 0, "regress": 0, "rollout": 0}

    def verify_stub(contents):
        i = calls["verify"]; calls["verify"] += 1
        ok = verify_seq[i] if i < len(verify_seq) else False
        return json.dumps({"success": ok, "confidence": 0.9 if ok else 0.7,
                           "reason": "ok" if ok else "no", "remaining": "" if ok else "push more"})

    def regress_stub(contents):
        calls["regress"] += 1
        return _refined_plan_json()

    def rollout_fn(plan, index):
        calls["rollout"] += 1
        return _write_rollout(tmp_path / f"rollout_{index:02d}.json", index, moved=0.0)

    loop = OptimizationLoop(
        RegressOptimizer(generate_fn=regress_stub),
        TaskVerifier(generate_fn=verify_stub),
        rollout_fn=rollout_fn, max_iters=max_iters, accumulate_dir=tmp_path,
    )
    return loop, calls


def test_candidate_success_short_circuits(tmp_path):
    # first candidate already solves it -> no regress, no extra rollout
    rp = _write_rollout(tmp_path / "rollout_00.json", 0, moved=0.10)
    loop, calls = _loop(tmp_path, verify_seq=[True])
    res = loop.run([(_candidate(), rp)], "push")
    assert res.success and res.iterations == 0
    assert calls["regress"] == 0 and calls["rollout"] == 0
    # verdict was attached to the candidate rollout (memory)
    assert json.loads(rp.read_text())["verdict"]["success"] is True


def test_refined_succeeds_after_iterations(tmp_path):
    rp = _write_rollout(tmp_path / "rollout_00.json", 0, moved=0.0)
    # candidate fails, first refined fails, second refined succeeds
    loop, calls = _loop(tmp_path, verify_seq=[False, False, True])
    res = loop.run([(_candidate(), rp)], "push")
    assert res.success and res.iterations == 2
    assert calls["regress"] == 2 and calls["rollout"] == 2
    # the failed refined attempt was accumulated with its verdict
    failed = json.loads((tmp_path / "rollout_01.json").read_text())
    assert failed["verdict"]["success"] is False and failed["verdict"]["remaining"]


def test_never_succeeds_returns_best_effort(tmp_path):
    rp = _write_rollout(tmp_path / "rollout_00.json", 0, moved=0.0)
    loop, calls = _loop(tmp_path, verify_seq=[False] * 10, max_iters=3)
    res = loop.run([(_candidate(), rp)], "push")
    assert not res.success and res.iterations == 3
    assert calls["regress"] == 3 and res.best_plan.description == "refined"


def test_max_iters_zero_is_no_regress(tmp_path):
    rp = _write_rollout(tmp_path / "rollout_00.json", 0, moved=0.0)
    loop, calls = _loop(tmp_path, verify_seq=[False], max_iters=0)
    res = loop.run([(_candidate(), rp)], "push")
    assert not res.success and res.iterations == 0 and calls["regress"] == 0
    assert res.best_plan.description == "cand"  # falls back to the candidate
