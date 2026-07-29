"""Shared VLM call + structured-output parsing.

Used by both the **VLM proposer** (action sampling) and the **VLM optimizer**
(regress). A VLM call takes an interleaved ``contents`` list of text and images
(``[str | PIL.Image, ...]``) — the proposer sends one scene image, the optimizer
sends several rollout after-images at once. Provider-agnostic: the model call is a
pluggable ``generate_fn(contents) -> str``; the default uses the secure Gemini
client (``GOOGLE_API_KEY`` from env). Output is parsed to a ``ProposalSet`` with
markdown-fence stripping and validate-with-retry.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional, Union

from simpact.actions import ProposalSet

# A VLM backend: interleaved contents (text + PIL images) -> raw model text.
GenerateFn = Callable[[list], str]


def load_image(path: Union[str, Path]):
    """Load an image as a PIL Image (supports .png/.jpg and .npy)."""
    from PIL import Image

    p = Path(path)
    if p.suffix == ".npy":
        import numpy as np

        return Image.fromarray(np.load(p))
    return Image.open(p)


def strip_json_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)


def gemini_generate(contents: list, model_id: Optional[str] = None,
                    timeout_s: float = 120.0) -> str:
    """Default VLM backend: Gemini via the secure client (key from env).

    A per-request ``timeout_s`` bounds each attempt so transient network/TLS flakiness
    surfaces as a fast error instead of the client's retry looping for minutes.
    """
    from google.genai import types

    from simpact.generator.client import get_gemini_client, get_model_id

    client = get_gemini_client()
    resp = client.models.generate_content(
        model=model_id or get_model_id(), contents=contents,
        config=types.GenerateContentConfig(http_options=types.HttpOptions(
            timeout=int(timeout_s * 1000))))  # google-genai timeout is in ms
    return resp.text


def generate_json(
    generate_fn: GenerateFn,
    contents: list,
    *,
    retries: int = 1,
    required_keys: Optional[set] = None,
) -> dict:
    """Call the VLM, parse JSON -> dict, validate required keys; retry on bad output.

    The structured-output sibling of ``generate_proposalset`` for callers that need
    a free-form dict (e.g. the task verifier's ``{success, reason, ...}`` verdict)
    rather than a ``ProposalSet``.
    """
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        text = generate_fn(contents)
        try:
            obj = json.loads(strip_json_fences(text))
            if not isinstance(obj, dict):
                raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
        except Exception as e:
            last_err = e
            continue
        if required_keys is not None:
            missing = required_keys - obj.keys()
            if missing:
                last_err = ValueError(f"missing keys: {sorted(missing)}")
                continue
        return obj
    raise ValueError(f"VLM JSON output failed after {retries + 1} attempt(s): {last_err}")


def generate_proposalset(
    generate_fn: GenerateFn,
    contents: list,
    *,
    retries: int = 1,
    allowed_types: Optional[set] = None,
    ranges: Optional[dict] = None,
) -> ProposalSet:
    """Call the VLM, parse JSON -> ProposalSet, validate; retry on bad output."""
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        text = generate_fn(contents)
        try:
            ps = ProposalSet.from_dict(json.loads(strip_json_fences(text)))
        except Exception as e:  # malformed JSON / wrong shape
            last_err = e
            continue
        if allowed_types is not None or ranges is not None:
            errs = ps.validate(allowed_types=allowed_types, ranges=ranges)
            if errs:
                last_err = ValueError("; ".join(errs[:5]))
                continue
        return ps
    raise ValueError(f"VLM structured output failed after {retries + 1} attempt(s): {last_err}")
