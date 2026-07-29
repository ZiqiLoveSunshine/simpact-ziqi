"""Tests for the VLM action-sequence optimizer (regress, step 3).

The VLM call is a stub ``generate_fn(contents) -> str`` (no API key/network); a
real Gemini call is env-gated.
"""
import json

import numpy as np
import pytest

from simpact.actions import GripperControl, Move, ProposalSet
from simpact.generator.regress import (
    RegressOptimizer,
    load_regress_template,
    parse_rigid_rollout,
)

REGRESS_RESPONSE = """```json
{"action_proposals": [{"description": "refined: push lower and further",
  "action_sequence": [
    {"type": "move", "delta_x": 0.2, "delta_y": 0.0, "delta_z": 0.0,
     "delta_roll": 0.0, "delta_pitch": 0.0, "delta_yaw": 0.0, "reasoning": "push"},
    {"type": "gripper_control", "width": 0.0, "reasoning": "stay closed"}]}]}
```"""


def _write_rollout(dirpath, index=0):
    """A minimal rigid rollout JSON (+ a real after-image png)."""
    from PIL import Image

    dirpath.mkdir(parents=True, exist_ok=True)
    png = f"rollout_{index:02d}_2.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(dirpath / png)
    data = {
        "timestamp": "20260623_000000",
        "proposal_index": index,
        "object_names": ["box"],
        "waypoints": [],
        "snapshots": [
            {"waypoint_index": 0, "gripper": {"position": [0.30, 0.0, 0.30],
             "orientation": [0, 1, 0, 0], "width": 0.0}, "objects": {}, "screenshot": None},
            {"waypoint_index": 1, "gripper": {"position": [0.40, 0.0, 0.30],
             "orientation": [0, 1, 0, 0], "width": 0.02}, "objects": {}, "screenshot": png},
        ],
    }
    (dirpath / f"rollout_{index:02d}.json").write_text(json.dumps(data))
    return dirpath / f"rollout_{index:02d}.json"


def test_template_has_move_gripper_format():
    t = load_regress_template("push")
    assert '"type": "move"' in t and '"type": "gripper_control"' in t
    assert "{instruction}" not in t  # regress template has no format placeholders


def test_parse_rigid_rollout(tmp_path):
    rj = _write_rollout(tmp_path)
    out = parse_rigid_rollout(rj)
    assert "Initial end effector's position (x,y,z): [0.3, 0.0, 0.3]" in out["text"]
    assert "move(delta_x=0.1000, delta_y=0.0000, delta_z=0.0000" in out["text"]
    assert "gripper_control(width=0.0400)" in out["text"]  # 2 * 0.02
    assert out["after_image"].endswith("rollout_00_2.png")


def test_optimize_with_stub_returns_move_plan(tmp_path):
    _write_rollout(tmp_path, 0)
    _write_rollout(tmp_path, 1)
    seen = {}

    def capture(contents):
        seen["contents"] = contents
        return REGRESS_RESPONSE

    opt = RegressOptimizer(generate_fn=capture)
    plan = opt.optimize(tmp_path, "push the box right", context="ctx")
    seq = plan.action_proposals[0].action_sequence
    assert [a.TYPE for a in seq] == ["move", "gripper_control"]
    assert isinstance(seq[0], Move) and isinstance(seq[1], GripperControl)

    contents = seen["contents"]
    from PIL import Image
    # both rollouts' after-images are attached
    assert sum(isinstance(c, Image.Image) for c in contents) == 2
    # the instruction + a rollout action trace are in the text contents
    text = "\n".join(c for c in contents if isinstance(c, str))
    assert "push the box right" in text and "move(delta_x=0.1000" in text


def test_optimize_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="no rollouts"):
        RegressOptimizer(generate_fn=lambda c: REGRESS_RESPONSE).optimize(tmp_path, "x")


@pytest.mark.requires_api
def test_real_gemini_regress(tmp_path):
    _write_rollout(tmp_path, 0)
    plan = RegressOptimizer().optimize(tmp_path, "push the box forward", context="N/A")
    assert len(plan.action_proposals) == 1
