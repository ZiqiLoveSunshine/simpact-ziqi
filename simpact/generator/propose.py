"""VLM action proposer — ported from the original ``propose_gemini_primitive.py``.

Vision-language: builds a prompt from a proposal template
(``prompts/proposals/*.txt``) + the scene image + context + instruction, calls a
VLM, and returns a validated ``ProposalSet``. The model call + parsing/retry are
the shared helpers in ``simpact.generator.vlm`` (also used by the optimizer);
the default backend is the secure Gemini client (no hardcoded keys).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from simpact.actions import ProposalSet
from simpact.generator.vlm import GenerateFn, gemini_generate, generate_proposalset, load_image
from simpact.utils.config import get_project_root


def get_proposals_dir() -> Path:
    return get_project_root() / "prompts" / "proposals"


def load_proposal_template(name_or_path: str) -> str:
    """Resolve a proposal template by name (e.g. ``"primitive"``) or path."""
    p = Path(name_or_path)
    if p.suffix == ".txt" and p.exists():
        return p.read_text()
    candidate = get_proposals_dir() / f"{Path(name_or_path).stem}.txt"
    if candidate.exists():
        return candidate.read_text()
    if p.exists():
        return p.read_text()
    raise FileNotFoundError(
        f"proposal template {name_or_path!r} not found (looked at {candidate} and as a path). "
        f"Available: {[t.stem for t in get_proposals_dir().glob('*.txt')]}"
    )


# kept for backwards-compatible imports
load_scene_image = load_image


class VLMProposer:
    """Propose action-primitive sequences from an image + context + instruction."""

    def __init__(
        self,
        prompt_template: str = "primitive",
        generate_fn: Optional[GenerateFn] = None,
        model_id: Optional[str] = None,
    ):
        self.prompt_template = prompt_template
        self.generate_fn = generate_fn or (lambda contents: gemini_generate(contents, model_id))

    def build_prompt(
        self,
        instruction: str,
        image_path: Union[str, Path],
        context: Optional[str] = None,
        motion_plan: Optional[str] = None,
    ) -> str:
        tmpl = load_proposal_template(self.prompt_template)
        # template uses ( ... ) for JSON examples (not braces), so .format is safe
        return tmpl.format(
            instruction=instruction,
            image_path=Path(image_path).name,
            context=context if context is not None else "N/A",
            motion_plan=motion_plan if motion_plan is not None else "N/A",
        )

    def propose(
        self,
        instruction: str,
        image_path: Union[str, Path],
        context: Optional[str] = None,
        motion_plan: Optional[str] = None,
        retries: int = 1,
        allowed_types: Optional[set] = None,
        ranges: Optional[dict] = None,
    ) -> ProposalSet:
        """Query the VLM and return a parsed (and optionally validated) ProposalSet."""
        prompt = self.build_prompt(instruction, image_path, context, motion_plan)
        image = load_image(image_path)
        # VLM contents: the scene image followed by the prompt text
        return generate_proposalset(
            self.generate_fn, [image, prompt],
            retries=retries, allowed_types=allowed_types, ranges=ranges,
        )


# Backwards-compatible alias (the proposer is vision-language; prefer VLMProposer).
LLMProposer = VLMProposer
