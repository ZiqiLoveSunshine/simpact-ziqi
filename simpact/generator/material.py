"""VLM material-ID (real2sim): estimate MPM object physical parameters per scene.

The OBJECT physical properties (``E``, ``nu``, ``yield_stress``, ``density``) are
estimated by a VLM from the scene image at build time (real2sim) and written into
``scene.yaml`` as a ``material:`` block — inferred from the observation, NOT hand-assigned
(docs/DEFORMABLE_INTEGRATION.md §15). This aligns physics with geometry: both are derived
from the scene, per scene.

The remaining MPM parameters (constitutive model, gravity, numerical damping, plasticity
``friction_angle``) are SIMULATION setup, not object physics, and stay as per-class
defaults here — they are not VLM-estimated.

Absolute SI prediction is unreliable, so the prompt grounds the VLM in the simulator's
effective-parameter regime via reference bands (``assets/materials/bands.yaml``) and asks
for a softness classification; the estimate is then CLAMPED to those bands as a safety net.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from simpact.generator.vlm import GenerateFn, default_generate, generate_json, load_image
from simpact.utils.config import get_materials_dir

# Object physical properties the VLM estimates per scene.
PHYS_KEYS = ("E", "nu", "yield_stress", "density")

# Non-physical MPM simulation setup per material class (NOT VLM-estimated): the
# constitutive model, gravity, numerical damping, and the plasticity friction angle.
SOLVER_CONFIG = {
    "dough": {"material": "plasticine", "friction_angle": 35.0,
              "g": [0.0, 0.0, -10.0], "grid_v_damping_scale": 0.9},
    "sweep": {"material": "plasticine", "friction_angle": 0.0,
              "g": [0.0, 0.0, -9.81], "grid_v_damping_scale": 0.95},
}

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "material" / "mpm_params.txt"


def load_bands() -> dict:
    return yaml.safe_load((get_materials_dir() / "bands.yaml").read_text())


def _band_range(bands: dict, key: str):
    """Overall plausible [lo, hi] for a parameter across all its softness sub-bands."""
    sub = bands[key].values()
    return min(v[0] for v in sub), max(v[1] for v in sub)


def _render_bands(bands: dict) -> str:
    lines = []
    for key in PHYS_KEYS:
        parts = "; ".join(f"{name} {lo}-{hi}" for name, (lo, hi) in bands[key].items())
        lines.append(f"- {key}: {parts}")
    return "\n".join(lines)


def estimate_material(image, object_name: str, material_class: str = "dough", *,
                      generate_fn: GenerateFn = default_generate, retries: int = 1) -> dict:
    """VLM-estimate the object's physical params (clamped to the reference bands).

    Returns ``{E, nu, yield_stress, density, softness, confidence, source: "vlm"}`` — the
    object-physics half of the MPM material dict (merge with SOLVER_CONFIG via load_material).
    """
    bands = load_bands()
    prompt = PROMPT_PATH.read_text().format(name=object_name, bands=_render_bands(bands))
    img = load_image(image) if isinstance(image, (str, Path)) else image
    obj = generate_json(generate_fn, [img, prompt], retries=retries, required_keys=set(PHYS_KEYS))

    def num(x):
        if isinstance(x, (int, float)):
            return float(x)
        import re
        m = re.search(r"[-+]?[\d.]+(?:[eE][-+]?\d+)?", str(x))
        if not m:
            raise ValueError(f"non-numeric material value: {x!r}")
        return float(m.group())

    out = {}
    for k in PHYS_KEYS:
        lo, hi = _band_range(bands, k)
        out[k] = float(min(max(num(obj[k]), lo), hi))  # clamp to plausible range
    out["softness"] = str(obj.get("softness", ""))
    out["confidence"] = float(obj.get("confidence", 0.0))
    out["source"] = "vlm"
    return out


def load_material(scene_dir, material_class: str = "dough") -> dict:
    """Full MPM ``material_params``: the VLM-estimated object physics from the scene's
    ``material:`` block merged with the per-class solver config.

    Errors if the scene has no complete ``material:`` block — physical params are
    VLM-estimated at build time (build_scene), never a hand-assigned default. Pass
    ``material_params`` explicitly to the rollout to bypass (e.g. sensitivity sweeps).
    """
    from simpact.utils.layout import find_scene_file
    y = yaml.safe_load(find_scene_file(scene_dir, "scene.yaml").read_text()) or {}
    mat = y.get("material")
    if not mat or not all(k in mat for k in PHYS_KEYS):
        raise KeyError(
            f"{Path(scene_dir)/'scene.yaml'} has no complete 'material:' block "
            f"(need {list(PHYS_KEYS)}). Physical params are VLM-estimated at build time; "
            "run estimate_material (build_scene) or pass material_params to the rollout.")
    if material_class not in SOLVER_CONFIG:
        raise KeyError(f"unknown material_class {material_class!r} (known: {list(SOLVER_CONFIG)})")
    phys = {k: float(mat[k]) for k in PHYS_KEYS}
    return {**SOLVER_CONFIG[material_class], **phys}


def write_material(scene_dir, params: dict) -> None:
    """Stamp an estimated ``material:`` block into the scene's scene.yaml (provenance kept)."""
    from simpact.utils.layout import find_scene_file
    yml = find_scene_file(scene_dir, "scene.yaml", required=False) or (Path(scene_dir) / "scene.yaml")
    y = yaml.safe_load(yml.read_text()) if yml.exists() else {}
    y["material"] = {k: params[k] for k in (*PHYS_KEYS, "softness", "confidence", "source")
                     if k in params}
    yml.write_text(yaml.dump(y, sort_keys=False))
