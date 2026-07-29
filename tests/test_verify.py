"""Tests for the VLM task-completion verifier (stubbed generate_fn, no real VLM)."""
import json

from simpact.generator.verify import (
    TaskVerifier, Verdict, alignment_gate, rollout_displacements)


def _write_rollout(path, before_xy, after_xy):
    data = {
        "timestamp": "T0",
        "object_names": ["carton"],
        "waypoints": [],
        "snapshots": [
            {"waypoint_index": 0, "gripper": {"position": [0, 0, 0.4],
             "orientation": [0, 1, 0, 0], "width": 0.0},
             "objects": {"carton": {"position": [*before_xy, 0.2], "orientation": [1, 0, 0, 0]}},
             "screenshot": None},
            {"waypoint_index": 1, "gripper": {"position": [0.1, 0, 0.4],
             "orientation": [0, 1, 0, 0], "width": 0.0},
             "objects": {"carton": {"position": [*after_xy, 0.2], "orientation": [1, 0, 0, 0]}},
             "screenshot": None},
        ],
    }
    path.write_text(json.dumps(data))
    return path


def test_rollout_displacements(tmp_path):
    rp = _write_rollout(tmp_path / "rollout_00.json", (0.0, 0.0), (0.10, 0.0))
    assert rollout_displacements(rp) == {"carton": 10.0}  # 0.10 m -> 10 cm


def test_verifier_parses_success_verdict(tmp_path):
    rp = _write_rollout(tmp_path / "rollout_00.json", (0.0, 0.0), (0.10, 0.0))
    stub = lambda contents: json.dumps(
        {"success": True, "confidence": 0.9, "reason": "carton reached target", "remaining": ""})
    v = TaskVerifier(generate_fn=stub).verify(rp, "push the carton right")
    assert isinstance(v, Verdict) and v.success and v.confidence == 0.9


def test_verifier_failure_verdict_keeps_feedback(tmp_path):
    rp = _write_rollout(tmp_path / "rollout_00.json", (0.0, 0.0), (0.003, 0.0))
    stub = lambda contents: "```json\n" + json.dumps(
        {"success": False, "confidence": 0.8, "reason": "barely moved",
         "remaining": "push further in -x"}) + "\n```"
    v = TaskVerifier(generate_fn=stub).verify(rp, "push the carton right")
    assert not v.success and v.remaining == "push further in -x"


def _write_two_object_rollout(path, white_final_y, blue_final_y):
    def snap(wi, wy, by):
        return {"waypoint_index": wi, "gripper": {"position": [0, 0, 0.4],
                "orientation": [0, 1, 0, 0], "width": 0.0},
                "objects": {
                    "white_carton": {"position": [0.5, wy, 0.2], "orientation": [1, 0, 0, 0]},
                    "blue_carton": {"position": [0.6, by, 0.2], "orientation": [1, 0, 0, 0]}},
                "screenshot": None}
    data = {"timestamp": "T0", "object_names": ["white_carton", "blue_carton"],
            "waypoints": [], "snapshots": [snap(0, -0.14, 0.03), snap(1, white_final_y, blue_final_y)]}
    path.write_text(json.dumps(data))
    return path


def test_alignment_gate_measures_axis_offset(tmp_path):
    gate = alignment_gate("white carton", "blue carton", axis="y", tol_m=0.02)
    # aligned: |Δy| = 1 cm <= 2 cm
    ok, detail = gate(_write_two_object_rollout(tmp_path / "a.json", 0.025, 0.035))
    assert ok and "1.0 cm" in detail
    # not aligned: |Δy| = 11 cm > 2 cm
    ok, detail = gate(_write_two_object_rollout(tmp_path / "b.json", -0.075, 0.035))
    assert not ok and "11.0 cm" in detail


def test_gate_ands_with_vlm_verdict(tmp_path):
    # VLM says valid (success), but the measured gate must veto when not aligned
    vlm_ok = lambda contents: json.dumps({"success": True, "confidence": 0.9,
                                          "reason": "clean", "remaining": ""})
    gate = alignment_gate("white carton", "blue carton", axis="y", tol_m=0.02)
    v = TaskVerifier(generate_fn=vlm_ok, success_gate=gate)

    aligned = _write_two_object_rollout(tmp_path / "aligned.json", 0.03, 0.035)
    assert v.verify(aligned, "align them").success is True          # VLM ok AND aligned

    misaligned = _write_two_object_rollout(tmp_path / "off.json", -0.10, 0.035)
    verdict = v.verify(misaligned, "align them")
    assert verdict.success is False                                  # gate vetoes
    assert "alignment" in verdict.remaining.lower()

    # conversely, gate passes but VLM fails (toppled) -> still failure
    vlm_bad = lambda contents: json.dumps({"success": False, "confidence": 0.9,
                                           "reason": "toppled", "remaining": "lower contact"})
    v2 = TaskVerifier(generate_fn=vlm_bad, success_gate=gate)
    assert v2.verify(aligned, "align them").success is False


def test_verifier_includes_motion_evidence(tmp_path):
    # the measured displacement must be passed to the model as evidence
    rp = _write_rollout(tmp_path / "rollout_00.json", (0.0, 0.0), (0.07, 0.0))
    seen = {}

    def stub(contents):
        seen["text"] = " ".join(c for c in contents if isinstance(c, str))
        return json.dumps({"success": False, "confidence": 0.5, "reason": "x", "remaining": "y"})

    TaskVerifier(generate_fn=stub).verify(rp, "push")
    assert "carton" in seen["text"] and "7.0" in seen["text"]  # displacement evidence present
