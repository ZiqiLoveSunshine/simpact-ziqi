"""Tests for the action primitive schema + random sampler (Phase 5, step 1)."""
import json
from pathlib import Path

import pytest

from simpact.actions import (
    ActionProposal,
    Grasp,
    ProposalSet,
    Push,
    Release,
    Rotate,
    primitive_from_dict,
)
from simpact.actions.primitives import PRIMITIVE_TYPES
from simpact.generator.sampling import DEFAULT_RANGES, RandomProposer

FIXTURES = Path(__file__).parent / "fixtures"
RECORDED_PROPOSALS = [FIXTURES / "recorded_proposal_sand.json", FIXTURES / "recorded_proposal_rope.json"]


@pytest.mark.parametrize("path", RECORDED_PROPOSALS, ids=lambda p: p.name)
def test_roundtrip_real_pag_proposal(path):
    """Real recorded proposals parse and re-serialize to the same structure."""
    original = json.loads(path.read_text())
    ps = ProposalSet.from_dict(original)
    assert json.loads(ps.to_json()) == original  # structural round-trip


def test_real_proposals_use_grasp_width_and_parse():
    """The recorded proposals use the 'grasp_width' key; it maps to Grasp.width."""
    ps = ProposalSet.from_json(FIXTURES / "recorded_proposal_sand.json")
    grasps = [a for p in ps.action_proposals for a in p.action_sequence if isinstance(a, Grasp)]
    assert grasps, "fixture should contain GRASP actions"
    assert all(0.0 <= g.width <= 0.1 for g in grasps)


def test_grasp_key_alias_normalization():
    """GRASP parses from either 'grasp_width' or 'width'; serializes to 'grasp_width'."""
    g1 = primitive_from_dict({"type": "GRASP", "grasp_width": 0.04})
    g2 = primitive_from_dict({"type": "GRASP", "width": 0.04})  # random-sampler / template style
    assert isinstance(g1, Grasp) and isinstance(g2, Grasp)
    assert g1.width == g2.width == 0.04
    assert g1.to_dict() == {"type": "GRASP", "grasp_width": 0.04}
    assert "width" not in g1.to_dict() and "grasp_width" in g1.to_dict()


def test_all_primitive_types_roundtrip():
    seq = [
        Push(0.1, -0.2, reasoning="r"),
        Rotate(1.57),
        Grasp(0.03),
        Release(),
    ]
    ps = ProposalSet([ActionProposal(seq, description="d")])
    assert ProposalSet.from_dict(ps.to_dict()).to_dict() == ps.to_dict()
    assert set(PRIMITIVE_TYPES) == {"PUSH", "LIFT", "DESCEND", "GRASP", "RELEASE", "ROTATE", "ROLL", "FLICK"}


def test_unknown_primitive_type_raises():
    with pytest.raises(ValueError):
        primitive_from_dict({"type": "TELEPORT", "delta_x": 1.0})


def test_random_sampler_is_seed_deterministic():
    a = RandomProposer(seed=0).sample(n=5, action_types=["PUSH", "ROTATE", "DESCEND"], min_len=2, max_len=6)
    b = RandomProposer(seed=0).sample(n=5, action_types=["PUSH", "ROTATE", "DESCEND"], min_len=2, max_len=6)
    c = RandomProposer(seed=1).sample(n=5, action_types=["PUSH", "ROTATE", "DESCEND"], min_len=2, max_len=6)
    assert a.to_dict() == b.to_dict()       # same seed -> identical
    assert a.to_dict() != c.to_dict()       # different seed -> different


def test_random_samples_are_valid_and_in_range():
    types = ["PUSH", "LIFT", "DESCEND", "GRASP", "ROTATE", "FLICK"]
    ps = RandomProposer(seed=7).sample(n=30, action_types=types, min_len=1, max_len=8)
    assert ps.validate(allowed_types=set(types), ranges=DEFAULT_RANGES) == []


def test_validate_flags_disallowed_type_and_out_of_range():
    ps = ProposalSet([ActionProposal([Push(9.0, 0.0), Grasp(0.05)])])
    errs = ps.validate(allowed_types={"GRASP"}, ranges=DEFAULT_RANGES)
    assert any("type not allowed" in e for e in errs)   # PUSH disallowed
    assert any("delta_x=9.0 out of" in e for e in errs)  # out of [-0.5, 0.5]


# ---- optimizer-output ("regress") plan actions ----------------------------- #
def test_move_and_gripper_control_roundtrip():
    from simpact.actions import GripperControl, Move, PLAN_ACTION_TYPES

    plan = ProposalSet([ActionProposal(
        [Move(0.1, -0.05, 0.0, 0.0, 0.0, 0.2, reasoning="approach"),
         GripperControl(0.04, reasoning="open")],
        description="refined plan")])
    assert ProposalSet.from_dict(plan.to_dict()).to_dict() == plan.to_dict()
    assert set(PLAN_ACTION_TYPES) == {"move", "gripper_control"}


def test_move_gripper_parse_from_regress_template_json():
    # exact shape from regress_template.txt
    data = {"action_proposals": [{"description": "d", "action_sequence": [
        {"type": "move", "delta_x": 0.1, "delta_y": 0.0, "delta_z": -0.05,
         "delta_roll": 0.0, "delta_pitch": 0.0, "delta_yaw": 1.57, "reasoning": "r"},
        {"type": "gripper_control", "width": 0.0, "reasoning": "close"}]}]}
    ps = ProposalSet.from_dict(data)
    seq = ps.action_proposals[0].action_sequence
    assert [a.TYPE for a in seq] == ["move", "gripper_control"]
    assert seq[0].delta_yaw == 1.57 and seq[1].width == 0.0
    assert ps.to_dict() == data  # round-trips byte-for-byte


def test_legacy_regress_file_still_roundtrips_as_primitives():
    # recorded legacy regress outputs predate move/gripper_control and use the
    # primitive format (PUSH/GRASP) -> must still parse via the same schema
    import json

    path = FIXTURES / "recorded_regress_legacy.json"
    original = json.loads(path.read_text())
    assert json.loads(ProposalSet.from_json(path).to_json()) == original


def test_propose_primitive_types_unchanged():
    # adding plan actions must not change the propose-stage primitive set
    assert set(PRIMITIVE_TYPES) == {"PUSH", "LIFT", "DESCEND", "GRASP", "RELEASE", "ROTATE", "ROLL", "FLICK"}
