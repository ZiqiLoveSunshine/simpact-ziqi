"""VLM action-sequence optimizer ("regress") — ported from the original ``regress_gemini.py``.

Reads the physics rollouts of N proposals (recorded-format JSON + overhead
screenshots from ``MuJoCoRollout``) and asks a VLM to propose **one optimized
plan** in the universal 6-DoF ``move``/``gripper_control`` format. There is no
cost/ranking — the model reasons over the rollouts' action traces + after-images.
Built on the shared ``simpact.generator.vlm`` helper (multi-image contents).

Rigid task only for now (consumes ``MuJoCoRollout`` JSON); rope/sand variants add
their own rollout parsers + templates later.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Union

from scipy.spatial.transform import Rotation as R

from simpact.actions import ProposalSet
from simpact.generator.vlm import GenerateFn, default_generate, generate_proposalset, load_image
from simpact.utils.config import get_project_root


def get_regress_dir() -> Path:
    return get_project_root() / "prompts" / "regress"


def load_regress_template(name_or_path: str) -> str:
    p = Path(name_or_path)
    if p.suffix == ".txt" and p.exists():
        return p.read_text()
    candidate = get_regress_dir() / f"{Path(name_or_path).stem}.txt"
    if candidate.exists():
        return candidate.read_text()
    if p.exists():
        return p.read_text()
    raise FileNotFoundError(f"regress template {name_or_path!r} not found (looked at {candidate}).")


def _yaw_from_wxyz(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    return float(R.from_quat([x, y, z, w]).as_euler("xyz")[2])


def parse_rigid_rollout(rollout_json: Union[str, Path]) -> dict:
    """Summarize a rigid MuJoCoRollout JSON as (text trace, after-image path).

    Reproduces the original ``regress_gemini.parse_rollout_to_text``: initial EE pose +
    gripper width (x2 for the total opening), then per-waypoint ``move`` deltas
    (and ``gripper_control`` when the width changes). ``snapshots[0]`` is the
    initial state; the last snapshot with a screenshot is the "after" image.
    """
    path = Path(rollout_json)
    data = json.loads(path.read_text())
    snaps = data.get("snapshots", [])
    text = f"Rollout serial no.: {data.get('timestamp', 'UNKNOWN')}\n"

    # If a verifier judged this rollout (closed-loop memory), surface its verdict so
    # the optimizer learns which prior attempts failed and exactly why.
    v = data.get("verdict")
    if v is not None:
        outcome = "SUCCESS" if v.get("success") else "FAILURE"
        text += f"- VERIFIER OUTCOME: {outcome}"
        if v.get("reason"):
            text += f" — {v['reason']}"
        if not v.get("success") and v.get("remaining"):
            text += f"\n- STILL NEEDED: {v['remaining']}"
        text += "\n"

    prev_pos = prev_yaw = prev_grip = None
    for i, s in enumerate(snaps):
        g = s["gripper"]
        pos, width = g["position"], g["width"]
        yaw = _yaw_from_wxyz(g["orientation"])
        if i == 0:
            text += f"- Initial end effector's position (x,y,z): {pos}\n"
            text += f"- Initial end effector's yaw (radians): {yaw}\n"
            text += f"- Initial gripper width: {width * 2.0}\n"
        else:
            dx, dy, dz = (pos[0] - prev_pos[0], pos[1] - prev_pos[1], pos[2] - prev_pos[2])
            text += f"\n--- Waypoint {s.get('waypoint_index', i) - 1} to {s.get('waypoint_index', i)} ---\n"
            text += (f"- ACTION TAKEN: move(delta_x={dx:.4f}, delta_y={dy:.4f}, "
                     f"delta_z={dz:.4f}, delta_yaw={yaw - prev_yaw:.4f})\n")
            if width - prev_grip > 1e-6:
                text += f"- ACTION TAKEN: gripper_control(width={2 * width:.4f})\n"
        prev_pos, prev_yaw, prev_grip = pos, yaw, width

    after = next((s["screenshot"] for s in reversed(snaps) if s.get("screenshot")), None)
    after_image = str(path.parent / after) if after else None
    return {"text": text, "after_image": after_image}


def parse_rope_rollout(rollout_json: Union[str, Path]) -> dict:
    """Summarize an ARAP rope rollout JSON as (text, after-image path).

    Faithful to the original ``regress_gemini_rope.parse_rollout_to_text``: grasp point, place
    point, and their delta, plus (closed-loop) the verifier verdict. The "after" image
    is the final rope-shape screenshot.
    """
    path = Path(rollout_json)
    data = json.loads(path.read_text())
    g = [round(v, 4) for v in data["grasp_point"]]
    p = [round(v, 4) for v in data["place_point"]]
    delta = [round(p[i] - g[i], 4) for i in range(3)]
    text = f"Rollout serial no.: {data.get('timestamp', 'UNKNOWN')}\n"
    v = data.get("verdict")
    if v is not None:
        text += f"- VERIFIER OUTCOME: {'SUCCESS' if v.get('success') else 'FAILURE'}"
        if v.get("reason"):
            text += f" — {v['reason']}"
        if not v.get("success") and v.get("remaining"):
            text += f"\n- STILL NEEDED: {v['remaining']}"
        text += "\n"
    text += (f"- Grasp point on the rope (x,y,z): {g}\n"
             f"- Place point the grasp was dragged to (x,y,z): {p}\n"
             f"- Delta (dx,dy,dz): {delta}\n")
    snaps = data.get("snapshots", [])
    after = next((s["screenshot"] for s in reversed(snaps) if s.get("screenshot")), None)
    return {"text": text, "after_image": str(path.parent / after) if after else None}


def parse_mpm_rollout(rollout_json: Union[str, Path]) -> dict:
    """Summarize an MPM dough/sand rollout JSON as (text, after-image path).

    Deformable analogue of ``parse_rope_rollout``: the grasp center + the width the
    jaws closed to, the resulting particle-cloud bounding box + centroid (the shape
    evidence, since MPM has no keypoints), and (closed-loop) the verifier verdict. The
    "after" image is the final dough-shape screenshot.
    """
    path = Path(rollout_json)
    data = json.loads(path.read_text())
    m = data.get("mpm", {})
    rnd = lambda v: [round(float(x), 4) for x in v]
    text = f"Rollout serial no.: {data.get('timestamp', 'UNKNOWN')}\n"
    v = data.get("verdict")
    if v is not None:
        text += f"- VERIFIER OUTCOME: {'SUCCESS' if v.get('success') else 'FAILURE'}"
        if v.get("reason"):
            text += f" — {v['reason']}"
        if not v.get("success") and v.get("remaining"):
            text += f"\n- STILL NEEDED: {v['remaining']}"
        text += "\n"
    if "sweep_segments" in data:  # sweep: a sequence of pusher moves
        segs = data["sweep_segments"]
        text += f"- Number of sweep segments: {len(segs)}\n"
        for i, s in enumerate(segs):
            text += (f"  segment {i}: start {rnd(s['start'])}, yaw {round(float(s['yaw']), 3)}, "
                     f"push delta {rnd(s['delta'])}\n")
    elif "grasp_centers" in data:  # dough: a sequence of 1..N squeezes
        text += f"- Number of squeezes: {len(data['grasp_centers'])}\n"
        for i, (c, w) in enumerate(zip(data["grasp_centers"], data.get("grasp_widths", []))):
            yaw = data.get("grasp_yaws", [0.0] * len(data["grasp_centers"]))[i]
            text += (f"  squeeze {i}: center {rnd(c)}, yaw {round(float(yaw), 3)}, "
                     f"width {round(float(w), 4)}\n")
    if m:
        text += (f"- Final dough centroid (x,y,z): {rnd(m.get('centroid', []))}\n"
                 f"- Final dough bounding-box size (dx,dy,dz): {rnd(m.get('bbox_size', []))}\n")
    snaps = data.get("snapshots", [])
    after = next((s["screenshot"] for s in reversed(snaps) if s.get("screenshot")), None)
    return {"text": text, "after_image": str(path.parent / after) if after else None}


def _strip_gripper_geometry(context: str) -> str:
    """Drop the EE-dimensions block from the context (as the original pipeline's regress does)."""
    a = context.find("robot gripper max width")
    b = context.find("--- World Coordinate System ---")
    if a != -1 and b != -1:
        line_end = context.find("\n", a)
        return context[: line_end + 1] + context[b - 1:]
    return context


class RegressOptimizer:
    """Optimize an action plan from physics rollouts via a VLM (no scoring)."""

    def __init__(
        self,
        prompt_template: str = "push",
        generate_fn: Optional[GenerateFn] = None,
        model_id: Optional[str] = None,
        parse_rollout=parse_rigid_rollout,
    ):
        self.prompt_template = prompt_template
        self.parse_rollout = parse_rollout
        self.generate_fn = generate_fn or (lambda contents, schema=None: default_generate(contents, model_id, schema))

    @staticmethod
    def _resolve_rollouts(rollouts) -> list[Path]:
        if isinstance(rollouts, (str, Path)) and Path(rollouts).is_dir():
            return sorted(Path(rollouts).glob("rollout_*.json"))
        return [Path(r) for r in rollouts]

    def optimize(
        self,
        rollouts: Union[str, Path, Sequence[Union[str, Path]]],
        instruction: str,
        context: str = "",
        *,
        retries: int = 1,
        allowed_types: Optional[set] = None,
    ) -> ProposalSet:
        """Return one optimized plan (ProposalSet of move/gripper_control)."""
        rollout_paths = self._resolve_rollouts(rollouts)
        if not rollout_paths:
            raise ValueError(f"no rollouts found in {rollouts!r}")

        contents: list = [
            load_regress_template(self.prompt_template),
            "--- Task Context ---",
            "\n### High-Level Instruction:\n", instruction,
            "\n### Real-World Context:\n", _strip_gripper_geometry(context),
            "\n### Simulation Rollouts:\n",
        ]
        images = []
        for rj in rollout_paths:
            parsed = self.parse_rollout(rj)
            contents.append(parsed["text"])
            if parsed["after_image"] and Path(parsed["after_image"]).exists():
                images.append(load_image(parsed["after_image"]))
        contents += images
        contents.append(
            f"Your final answer should **ALWAYS** consider the task instruction: **{instruction}**\n"
        )
        return generate_proposalset(
            self.generate_fn, contents, retries=retries, allowed_types=allowed_types
        )
