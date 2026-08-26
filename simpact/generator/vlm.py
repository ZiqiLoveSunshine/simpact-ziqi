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


# Fields each primitive carries on the wire (``Primitive.to_dict``), used to build
# a decode-time JSON schema. Mirrors simpact/actions/primitives.py.
_PRIMITIVE_FIELDS: dict[str, tuple[str, ...]] = {
    "PUSH": ("delta_x", "delta_y"),
    "LIFT": ("delta_z",),
    "DESCEND": ("delta_z",),
    "GRASP": ("grasp_width",),
    "RELEASE": (),
    "ROTATE": ("delta_yaw",),
    "ROLL": ("delta_roll",),
    "FLICK": ("delta_x", "delta_y", "delta_z"),
    # optimizer-output ("regress") plan actions — lowercase tags, same container
    "move": ("delta_x", "delta_y", "delta_z", "delta_roll", "delta_pitch", "delta_yaw"),
    "gripper_control": ("width",),
}


def proposalset_schema(allowed_types: Optional[set] = None,
                       max_proposals: int = 3, max_actions: int = 10) -> dict:
    """JSON schema for a ``ProposalSet``, restricted to ``allowed_types``.

    Handed to a local server as ``response_format={"type": "json_schema", ...}``, this
    makes the task's primitive whitelist a **decoding constraint** rather than a
    request the prompt makes politely. Small models reliably ignore the latter — they
    reach for GRASP/RELEASE on a push task — and every such sample is a wasted rollout.
    The constraint is the same one ``ProposalSet.validate`` already enforces after the
    fact, so this only moves the check earlier; it never widens what is accepted.

    ``max_proposals``/``max_actions`` bound the arrays. Without an upper bound a small
    model happily emits proposals until it hits ``max_tokens`` and the JSON is truncated
    mid-object — the cap simply matches the prompt, which asks for 3 plans.
    """
    types = sorted(allowed_types) if allowed_types else sorted(_PRIMITIVE_FIELDS)
    variants = []
    for t in types:
        fields = _PRIMITIVE_FIELDS.get(t, ())
        props: dict = {"type": {"const": t}}
        props.update({f: {"type": "number"} for f in fields})
        props["reasoning"] = {"type": "string"}
        variants.append({"type": "object", "properties": props,
                         "required": ["type", *fields, "reasoning"]})
    return {
        "type": "object",
        "properties": {"action_proposals": {
            "type": "array", "minItems": 1, "maxItems": max_proposals,
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "action_sequence": {
                        "type": "array", "minItems": 1, "maxItems": max_actions,
                        "items": variants[0] if len(variants) == 1 else {"anyOf": variants},
                    },
                },
                "required": ["description", "action_sequence"],
            },
        }},
        "required": ["action_proposals"],
    }


def _accepts_schema(fn) -> bool:
    """Whether a ``GenerateFn`` can take a decode-time ``schema`` (local backends can)."""
    import inspect

    try:
        return "schema" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _encode_image(img, fmt: str = "PNG") -> str:
    """PIL image -> base64 string (no data-URI prefix)."""
    import base64
    import io

    buf = io.BytesIO()
    if fmt == "JPEG" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def openai_generate(contents: list, model_id: Optional[str] = None,
                    base_url: Optional[str] = None, timeout_s: float = 600.0,
                    schema: Optional[dict] = None) -> str:
    """VLM backend for any OpenAI-compatible ``/chat/completions`` server.

    This is the **local-model** path: llama.cpp's ``llama-server``, vLLM, Ollama and
    friends all speak this wire format, so the same propose/verify/regress loop runs
    against a self-hosted VLM with no API key. The interleaved ``contents`` list is
    flattened into a single user message whose parts are ``text`` and ``image_url``
    (base64 data URI) — the shared subset those servers implement.

    Every simpact VLM call site parses JSON, so ``response_format={"type":
    "json_object"}`` is requested by default: on llama.cpp that constrains decoding
    with a JSON grammar, which is what makes small local models usable here (no
    markdown fences, no prose preamble). Set ``SIMPACT_VLM_JSON_MODE=0`` to disable
    it for a server that rejects the field.

    Config comes from the environment (see ``.env.example``): ``SIMPACT_VLM_BASE_URL``,
    ``SIMPACT_VLM_MODEL``, ``SIMPACT_VLM_API_KEY``, ``SIMPACT_VLM_TEMPERATURE``,
    ``SIMPACT_VLM_MAX_TOKENS``, ``SIMPACT_VLM_IMAGE_FORMAT``.
    """
    import json as _json
    import os
    import urllib.error
    import urllib.request

    base = (base_url or os.environ.get("SIMPACT_VLM_BASE_URL")
            or "http://127.0.0.1:8080/v1").rstrip("/")
    model = model_id or os.environ.get("SIMPACT_VLM_MODEL") or "local-vlm"
    fmt = os.environ.get("SIMPACT_VLM_IMAGE_FORMAT", "PNG").upper()
    mime = "jpeg" if fmt == "JPEG" else "png"

    parts: list[dict] = []
    for c in contents:
        if isinstance(c, str):
            if c.strip():
                parts.append({"type": "text", "text": c})
        else:  # PIL image
            parts.append({"type": "image_url", "image_url": {
                "url": f"data:image/{mime};base64,{_encode_image(c, fmt)}"}})

    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": parts}],
        "temperature": float(os.environ.get("SIMPACT_VLM_TEMPERATURE", 0.7)),
        "max_tokens": int(os.environ.get("SIMPACT_VLM_MAX_TOKENS", 8192)),
        "stream": False,
    }
    if schema is not None and os.environ.get("SIMPACT_VLM_SCHEMA_MODE", "1") != "0":
        # strongest form: llama.cpp compiles the schema to a GBNF grammar, so the
        # sampler cannot emit a disallowed primitive at all
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "simpact_output", "schema": schema, "strict": True}}
    elif os.environ.get("SIMPACT_VLM_JSON_MODE", "1") != "0":
        payload["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=_json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('SIMPACT_VLM_API_KEY', 'no-key')}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            resp = _json.loads(r.read())
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"local VLM request to {base}/chat/completions failed: {e}. "
            "Is the server running? See docs/LOCAL_VLM.md"
        ) from e
    return resp["choices"][0]["message"]["content"]


def default_generate(contents: list, model_id: Optional[str] = None,
                     schema: Optional[dict] = None) -> str:
    """Dispatch to the backend named by ``SIMPACT_VLM_BACKEND`` (default ``gemini``).

    The single seam every VLM caller (proposer, verifier, optimizer, grounding,
    material-ID) resolves its default through, so pointing the whole closed loop at a
    self-hosted model is one env var and no code change.
    """
    import os

    backend = os.environ.get("SIMPACT_VLM_BACKEND", "gemini").strip().lower()
    if backend in ("openai", "local", "llama", "llamacpp", "vllm", "ollama"):
        return openai_generate(contents, model_id, schema=schema)
    if backend == "gemini":
        return gemini_generate(contents, model_id)  # schema-free; Gemini follows the prompt
    raise ValueError(
        f"unknown SIMPACT_VLM_BACKEND={backend!r} (expected 'gemini' or 'openai')")


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
    """Call the VLM, parse JSON -> ProposalSet, validate; retry on bad output.

    When the backend supports decode-time schemas (the local/OpenAI path) the
    ``allowed_types`` whitelist is also compiled into a grammar, so a disallowed
    primitive cannot be sampled in the first place. Backends without that support
    (Gemini) are called unchanged and still validated after the fact.
    """
    schema = (proposalset_schema(allowed_types)
              if allowed_types and _accepts_schema(generate_fn) else None)
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        text = generate_fn(contents, schema=schema) if schema else generate_fn(contents)
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
