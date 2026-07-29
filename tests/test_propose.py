"""Tests for the VLM action proposer (Phase 5, step 3).

The VLM call is injected as a stub ``generate_fn(contents) -> str`` so
parsing/validation/retry are tested with no API key or network. A real Gemini
call is env-gated.
"""

import numpy as np
import pytest

from simpact.actions import Grasp, ProposalSet
from simpact.generator.propose import VLMProposer, load_proposal_template

# A model response in the exact shape the original model output uses (note: fenced + grasp_width).
VLM_RESPONSE = """```json
{"action_proposals": [
  {"description": "push then grasp",
   "action_sequence": [
     {"type": "PUSH", "delta_x": 0.1, "delta_y": 0.0, "reasoning": "approach"},
     {"type": "GRASP", "grasp_width": 0.03, "reasoning": "close"}]}]}
```"""


@pytest.fixture
def scene_image(tmp_path):
    p = tmp_path / "camera1_rgb.npy"
    np.save(p, np.zeros((8, 8, 3), dtype=np.uint8))
    return p


def test_templates_load_and_have_placeholders():
    t = load_proposal_template("primitive")
    for ph in ("{instruction}", "{image_path}", "{context}", "{motion_plan}"):
        assert ph in t
    assert "grasp_width" in t  # fixed key (was 'width' in the original pipeline output-format block)


def test_proposer_parses_stub_response(scene_image):
    prop = VLMProposer(generate_fn=lambda contents: VLM_RESPONSE)
    ps = prop.propose("push the bottle", scene_image, context="ctx")
    assert isinstance(ps, ProposalSet) and len(ps.action_proposals) == 1
    seq = ps.action_proposals[0].action_sequence
    assert [a.TYPE for a in seq] == ["PUSH", "GRASP"]
    assert isinstance(seq[1], Grasp) and seq[1].width == 0.03  # grasp_width -> width


def test_proposer_sends_image_and_prompt_as_contents(scene_image):
    from PIL import Image

    seen = {}

    def capture(contents):
        seen["contents"] = contents
        return VLM_RESPONSE

    VLMProposer(generate_fn=capture).propose("ALIGN THE CARTON", scene_image, context="CTXSTR")
    contents = seen["contents"]
    assert isinstance(contents[0], Image.Image)  # scene image first
    prompt = contents[-1]
    assert "ALIGN THE CARTON" in prompt and scene_image.name in prompt and "CTXSTR" in prompt


def test_proposer_retries_then_succeeds(scene_image):
    calls = {"n": 0}

    def flaky(contents):
        calls["n"] += 1
        return "not json" if calls["n"] == 1 else VLM_RESPONSE

    ps = VLMProposer(generate_fn=flaky).propose("x", scene_image, retries=1)
    assert calls["n"] == 2 and len(ps.action_proposals) == 1


def test_proposer_raises_after_exhausting_retries(scene_image):
    with pytest.raises(ValueError, match="failed after"):
        VLMProposer(generate_fn=lambda contents: "garbage").propose("x", scene_image, retries=1)


def test_proposer_validation_rejects_out_of_task_primitive(scene_image):
    with pytest.raises(ValueError, match="failed after"):
        VLMProposer(generate_fn=lambda contents: VLM_RESPONSE).propose(
            "x", scene_image, allowed_types={"PUSH"}, retries=0
        )


@pytest.mark.requires_api
def test_real_gemini_proposal(scene_image):
    ps = VLMProposer().propose("push the object forward", scene_image, context="N/A")
    assert len(ps.action_proposals) >= 1
