"""VLM task-completion verifier.

The piece the original pipeline's open-loop ``regress`` lacks: after a plan is rolled out, a VLM
looks at the rollout and decides whether the **instruction was accomplished**.
This turns the optimizer into a closed loop (see ``optimize_loop``): a failed
attempt's verdict becomes feedback the next ``regress`` reads.

The judgment is VLM-first but corroborated by the **measured per-object
displacement** already recorded in the rollout JSON (the camera angle can deceive;
the numbers cannot). A task whose success is a precise geometric condition (e.g.
"the two cartons end horizontally aligned") should not be left to the VLM's eyeball
at all: pass a **measured ``success_gate``** (see ``alignment_gate``) — the final
verdict is then ``VLM_valid AND gate_passed``, so the VLM only judges validity
(upright / collateral / no-catastrophe) while the gate decides goal completion.
Provider-agnostic ``generate_fn`` like the proposer/optimizer, so it is
stub-injectable in tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np

from simpact.generator.vlm import GenerateFn, gemini_generate, generate_json, load_image
from simpact.utils.config import get_project_root


def get_verify_dir() -> Path:
    return get_project_root() / "prompts" / "verify"


def load_verify_template(name_or_path: str) -> str:
    p = Path(name_or_path)
    if p.suffix == ".txt" and p.exists():
        return p.read_text()
    candidate = get_verify_dir() / f"{Path(name_or_path).stem}.txt"
    if candidate.exists():
        return candidate.read_text()
    if p.exists():
        return p.read_text()
    raise FileNotFoundError(f"verify template {name_or_path!r} not found (looked at {candidate}).")


@dataclass
class Verdict:
    """A task-completion judgment over one rollout."""

    success: bool
    confidence: float = 0.0
    reason: str = ""
    remaining: str = ""  # actionable feedback when not successful (regress reads it)

    def to_dict(self) -> dict:
        return {"success": self.success, "confidence": self.confidence,
                "reason": self.reason, "remaining": self.remaining}

    @classmethod
    def from_dict(cls, d: dict) -> "Verdict":
        return cls(
            success=bool(d.get("success", False)),
            confidence=float(d.get("confidence", 0.0) or 0.0),
            reason=str(d.get("reason", "")),
            remaining=str(d.get("remaining", "")),
        )


def rollout_displacements(rollout_json: Union[str, Path]) -> dict:
    """Net per-object displacement (cm) between the first and last snapshot."""
    data = json.loads(Path(rollout_json).read_text())
    snaps = data.get("snapshots", [])
    if len(snaps) < 2:
        return {}
    first, last = snaps[0]["objects"], snaps[-1]["objects"]
    out = {}
    for name in data.get("object_names", list(first)):
        a = np.asarray(first[name]["position"]); b = np.asarray(last[name]["position"])
        out[name] = round(float(np.linalg.norm(b - a)) * 100, 1)
    return out


def _screenshot(snap, base: Path) -> Optional[Path]:
    s = snap.get("screenshot")
    return base / s if s else None


# A measured success gate: given a rollout JSON path, return (passed, one-line detail).
SuccessGate = Callable[[Union[str, Path]], Tuple[bool, str]]
_AXES = {"x": 0, "y": 1, "z": 2}


def _final_objects(rollout_json: Union[str, Path]) -> dict:
    return json.loads(Path(rollout_json).read_text())["snapshots"][-1]["objects"]


def _match_object(objects: dict, name: str) -> str:
    """Resolve an object key in a rollout's ``objects`` dict (rollout keys use
    underscores; callers may pass the spaced name or a substring)."""
    if name in objects:
        return name
    under = name.replace(" ", "_")
    if under in objects:
        return under
    hits = [k for k in objects if under in k or k in under]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f"object {name!r} not found among {list(objects)}")


def alignment_gate(name_a: str, name_b: str, *, axis: str = "y",
                   tol_m: float = 0.02) -> SuccessGate:
    """Measured gate: two objects end aligned along ``axis`` within ``tol_m`` metres.

    "Horizontal alignment" for the push task = the two cartons share the same
    front-back depth (``axis="y"``), i.e. ``|y_a - y_b| <= tol_m`` in the final
    settled snapshot. Deterministic and read straight from the rollout positions, so
    it is immune to the top-down camera's foreshortening (unlike a VLM eyeball).
    """
    ai = _AXES[axis]

    def gate(rollout_json: Union[str, Path]) -> Tuple[bool, str]:
        objs = _final_objects(rollout_json)
        pa = objs[_match_object(objs, name_a)]["position"]
        pb = objs[_match_object(objs, name_b)]["position"]
        off = abs(float(pa[ai]) - float(pb[ai]))
        passed = off <= tol_m
        detail = (f"{axis}-alignment |Δ{axis}| = {off * 100:.1f} cm "
                  f"({'≤' if passed else '>'} {tol_m * 100:.0f} cm tol)")
        return passed, detail

    return gate


def coverage_gate(target, *, min_coverage: float = 0.5) -> SuccessGate:
    """Measured gate: a fraction ``min_coverage`` of the final MPM particles ends up
    inside the target region (the sweep task's ``bg_pcd``).

    ``target`` is a target-region point cloud (``.ply`` path or ``(N,3)`` array). The
    region's ``(x,y)`` convex hull is the goal footprint; coverage = fraction of the
    rollout's final particles (from ``mpm.final_points_path``) whose ``(x,y)`` falls
    inside it. This is the **first measured deformable success signal** — a real
    closed-loop metric, unlike the VLM-only shape judgment (docs/DEFORMABLE_INTEGRATION
    §5/§12). Read straight from the particle cloud in robot frame, so it is immune to
    the render viewpoint.
    """
    from matplotlib.path import Path as MplPath
    from scipy.spatial import ConvexHull

    if isinstance(target, (str, Path)):
        import open3d as o3d
        target = np.asarray(o3d.io.read_point_cloud(str(target)).points)
    xy = np.asarray(target)[:, :2]
    poly = MplPath(xy[ConvexHull(xy).vertices])

    def gate(rollout_json: Union[str, Path]) -> Tuple[bool, str]:
        path = Path(rollout_json)
        mpm = json.loads(path.read_text()).get("mpm", {})
        fp = path.parent / mpm["final_points_path"]
        pts = np.load(fp)[:, :2]
        cov = float(poly.contains_points(pts).mean())
        passed = cov >= min_coverage
        detail = (f"coverage {cov * 100:.0f}% of material inside target "
                  f"({'≥' if passed else '<'} {min_coverage * 100:.0f}% goal)")
        return passed, detail

    return gate


class TaskVerifier:
    """Decide whether a rollout accomplished the instruction (VLM + motion evidence)."""

    def __init__(
        self,
        prompt_template: str = "push",
        generate_fn: Optional[GenerateFn] = None,
        model_id: Optional[str] = None,
        success_gate: Optional[SuccessGate] = None,
    ):
        self.prompt_template = prompt_template
        self.generate_fn = generate_fn or (lambda contents: gemini_generate(contents, model_id))
        # measured goal check ANDed with the VLM's validity verdict (e.g. alignment)
        self.success_gate = success_gate

    def verify(
        self,
        rollout_json: Union[str, Path],
        instruction: str,
        context: str = "",
        *,
        action_trace: str = "",
        retries: int = 1,
    ) -> Verdict:
        """Return a ``Verdict`` for one rollout. ``action_trace`` is the optional
        parsed action text (e.g. from ``parse_rigid_rollout``)."""
        path = Path(rollout_json)
        data = json.loads(path.read_text())
        snaps = data.get("snapshots", [])
        before = _screenshot(snaps[0], path.parent) if snaps else None
        after = next((_screenshot(s, path.parent) for s in reversed(snaps)
                      if s.get("screenshot")), None)
        disp = rollout_displacements(path)

        contents: list = [
            load_verify_template(self.prompt_template),
            "\n### High-Level Instruction:\n", instruction,
            "\n### Real-World Context:\n", context,
        ]
        if before and before.exists():
            contents += ["\n### Before (initial scene):\n", load_image(before)]
        if after and after.exists():
            contents += ["\n### After (final settled scene):\n", load_image(after)]
        if action_trace:
            contents += ["\n### Action trace executed:\n", action_trace]
        contents += [
            "\n### Measured evidence — net object displacement (cm):\n",
            json.dumps(disp),
            f"\n\nDecide strictly whether this accomplished: **{instruction}**\n",
        ]
        obj = generate_json(self.generate_fn, contents, retries=retries,
                            required_keys={"success"})
        verdict = Verdict.from_dict(obj)

        # AND the VLM's validity verdict with the measured goal gate (if any).
        if self.success_gate is not None:
            passed, detail = self.success_gate(path)
            verdict.reason = f"[{detail}] {verdict.reason}".strip()
            if not passed:
                if verdict.remaining:
                    verdict.remaining = f"{detail}; {verdict.remaining}"
                else:
                    verdict.remaining = f"not aligned yet: {detail}"
            verdict.success = bool(verdict.success and passed)
        return verdict
