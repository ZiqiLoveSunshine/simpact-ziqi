"""VLM grounding of rope endpoints (Phases 2-3 of the endpoint-grounding plan).

Phase 1 (``real2sim.detect_rope_endpoints``) finds the rope's two tips in 3-D but not
their *roles*. This module:
  * **Phase 2 — annotate**: projects the two 3-D tips onto the REAL camera image with
    that camera's own ``K`` + ``cam_to_robot`` and draws labeled markers "A"/"B". A
    camera-consistency guard (``_assert_camera_consistent``) refuses to annotate unless
    the intrinsics/extrinsic and the RGB are the same camera, so the markers the VLM
    sees land on the true rope ends in the real image.
  * **Phase 3 — assign roles**: sends the annotated image + task context to the VLM and
    gets a structured verdict of which marker is the *fixed* (anchored) end vs the
    *free* (graspable) end.
  * **orchestration** (``ground_rope_endpoints``): detect -> annotate -> VLM -> map
    labels back to the 3-D tips -> optionally write ``scene.yaml`` (``fixed_point`` /
    ``free_end``) with provenance, the same schema the manual picker produced.

See docs/DEFORMABLE_INTEGRATION.md §13. Zero downstream changes: context.py /
rope_rollout.py keep reading ``scene.yaml`` unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from simpact.executor.render_deformable import project
from simpact.generator.vlm import GenerateFn, gemini_generate, generate_json, load_image
from simpact.real2sim.detect_rope_endpoints import EndpointResult, detect_rope_endpoints

LABELS = ("A", "B")
ROLE_KEYS = {"fixed", "free", "are_valid_tips", "confidence", "reasoning"}
PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "grounding" / "rope_endpoints.txt"
MIN_INBOUNDS_FRAC = 0.8  # this fraction of the cloud must project into the RGB


def _assert_camera_consistent(img_size, K, cam_to_robot, cloud=None,
                              min_inbounds: float = MIN_INBOUNDS_FRAC) -> float:
    """Refuse to annotate unless (K, cam_to_robot) and the RGB are the same camera.

    Checks the principal point falls inside the image and — if a reference cloud is
    given — that most of it projects in-front-and-in-bounds. A resolution / intrinsics
    mismatch (wrong camera, resized RGB) would silently place the markers off the real
    rope; this turns that into a loud error. Returns the in-bounds fraction.
    """
    W, H = img_size
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if not (0 < cx < W and 0 < cy < H):
        raise ValueError(
            f"camera mismatch: K principal point ({cx:.0f},{cy:.0f}) is outside the "
            f"{W}x{H} image — K does not match this RGB resolution")
    if cloud is None or len(cloud) == 0:
        return 1.0
    uv, front = project(np.asarray(cloud, float), K, cam_to_robot)
    inb = front & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    frac = float(inb.mean())
    if frac < min_inbounds:
        raise ValueError(
            f"camera mismatch: only {frac:.0%} of the rope cloud projects into the "
            f"{W}x{H} image via this (K, cam_to_robot) — annotation would not align "
            f"with the real image (need >= {min_inbounds:.0%})")
    return frac


def annotate_tips(image, tips: np.ndarray, K: np.ndarray, cam_to_robot: np.ndarray, *,
                  labels=LABELS, cloud=None, radius: int = 12):
    """Draw labeled markers at the projected tips on the real RGB (Phase 2).

    Returns ``(annotated_PIL_image, mapping)`` where ``mapping[label]`` is
    ``{"px": (u, v), "xyz": tip}`` — the 3-D coordinate is kept behind each label so
    the VLM's A/B answer maps straight back to an exact 3-D point.
    """
    from PIL import ImageDraw, ImageFont

    img = (load_image(image) if isinstance(image, (str, Path)) else image).convert("RGB")
    tips = np.asarray(tips, float)
    if len(tips) != len(labels):
        raise ValueError(f"need {len(labels)} tips, got {len(tips)}")
    _assert_camera_consistent(img.size, K, cam_to_robot, cloud)
    uv, front = project(tips, K, cam_to_robot)

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    mapping = {}
    for i, lab in enumerate(labels):
        u, v = float(uv[i, 0]), float(uv[i, 1])
        draw.ellipse([u - radius, v - radius, u + radius, v + radius],
                     outline=(0, 255, 0), width=4)
        draw.line([u - radius, v, u + radius, v], fill=(0, 255, 0), width=2)
        draw.line([u, v - radius, u, v + radius], fill=(0, 255, 0), width=2)
        draw.text((u + radius + 3, v - radius - 3), lab, fill=(0, 255, 0), font=font,
                  stroke_width=2, stroke_fill=(0, 0, 0))
        mapping[lab] = {"px": (u, v), "xyz": tips[i], "front": bool(front[i])}
    return img, mapping


def assign_endpoint_roles(annotated_image, context: str, *,
                          generate_fn: GenerateFn = gemini_generate,
                          labels=LABELS, prompt_path: Path = PROMPT_PATH,
                          retries: int = 1) -> dict:
    """Ask the VLM which marker is the fixed vs free end (Phase 3).

    Returns the validated verdict dict (keys in ``ROLE_KEYS``). Raises if the model's
    fixed/free are missing, equal, or not among ``labels``.
    """
    prompt = prompt_path.read_text()
    contents = [annotated_image, f"{prompt}\n\n--- TASK CONTEXT ---\n{context}"]
    obj = generate_json(generate_fn, contents, retries=retries, required_keys=ROLE_KEYS)
    fixed, free = obj.get("fixed"), obj.get("free")
    if fixed not in labels or free not in labels:
        raise ValueError(f"VLM returned labels outside {labels}: fixed={fixed!r} free={free!r}")
    if fixed == free:
        raise ValueError(f"VLM assigned the same marker to both roles: {fixed!r}")
    return obj


@dataclass
class GroundingResult:
    fixed_point: np.ndarray          # 3-D anchor (VLM-chosen)
    free_end: np.ndarray             # 3-D graspable tip (VLM-chosen)
    detection: EndpointResult        # Phase-1 geometry result
    roles: dict                      # raw VLM verdict
    mapping: dict                    # label -> {px, xyz, front}
    annotated_image_path: Optional[str] = None
    warnings: list = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Combined trust: geometry detection x VLM role confidence."""
        return float(self.detection.confidence) * float(self.roles.get("confidence", 0.0))


def ground_rope_endpoints(scene_dir, *, cam: int = 1,
                          generate_fn: GenerateFn = gemini_generate,
                          write: bool = False, save_annotated: bool = True
                          ) -> GroundingResult:
    """Full detect -> annotate -> VLM -> map pipeline for one rope scene.

    With ``write=True`` updates ``scene.yaml`` (``fixed_point``/``free_end`` +
    ``endpoint_source: vlm`` + ``endpoint_confidence``) so the rest of the pipeline
    consumes it unchanged. Never overwrites without ``write``.
    """
    import open3d as o3d

    from simpact.utils.layout import find_scene_file
    scene = Path(scene_dir)
    cloud = np.asarray(o3d.io.read_point_cloud(str(find_scene_file(scene, "segmented_object.ply"))).points)
    det = detect_rope_endpoints(cloud)

    from simpact.real2sim.camera_calibration import load_camera
    cp = load_camera(scene, cam)  # embedded per-scene, else scene.yaml profile ref
    K, cam_to_robot = cp.K, cp.cam_to_robot
    rgb = find_scene_file(scene, f"camera{cam}_rgb.png", required=False)
    if rgb is None:
        raise FileNotFoundError(f"grounding needs the real RGB image: {scene}/camera{cam}_rgb.png")

    annotated, mapping = annotate_tips(rgb, det.tips, K, cam_to_robot, cloud=cloud)
    ann_path = None
    if save_annotated:
        ann_path = str(scene / f"endpoint_annotated_cam{cam}.png")
        annotated.save(ann_path)

    _ctx = find_scene_file(scene, "context.txt", required=False)
    context = _ctx.read_text() if _ctx else ""
    roles = assign_endpoint_roles(annotated, context, generate_fn=generate_fn)

    fixed_xyz = np.asarray(mapping[roles["fixed"]]["xyz"], float)
    free_xyz = np.asarray(mapping[roles["free"]]["xyz"], float)

    warnings = []
    if not roles.get("are_valid_tips", True):
        warnings.append("VLM flagged the markers as not on distinct rope ends")
    if det.method != "geodesic":
        warnings.append(f"geometry fell back to {det.method}")
    if float(roles.get("confidence", 0.0)) < 0.5:
        warnings.append(f"low VLM role confidence ({roles.get('confidence')})")

    result = GroundingResult(fixed_xyz, free_xyz, det, roles, mapping, ann_path, warnings)

    if write:
        import yaml
        yml = find_scene_file(scene, "scene.yaml", required=False) or (scene / "scene.yaml")
        y = yaml.safe_load(yml.read_text()) if yml.exists() else {}
        y["fixed_point"] = fixed_xyz.tolist()
        y["free_end"] = free_xyz.tolist()
        y["endpoint_source"] = "vlm"
        y["endpoint_confidence"] = result.confidence
        yml.write_text(yaml.dump(y, indent=2))
    return result


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="VLM-ground a rope's fixed/free endpoints.")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--write", action="store_true", help="write results to scene.yaml")
    args = ap.parse_args()
    r = ground_rope_endpoints(args.scene, cam=args.cam, write=args.write)
    print(f"fixed({r.roles['fixed']}) = {np.round(r.fixed_point, 4).tolist()}")
    print(f"free ({r.roles['free']}) = {np.round(r.free_end, 4).tolist()}")
    print(f"detection={r.detection.method}/{r.detection.confidence:.2f}  "
          f"vlm_conf={r.roles.get('confidence')}  combined={r.confidence:.2f}")
    print(f"reasoning: {r.roles.get('reasoning')}")
    if r.warnings:
        print("warnings: " + "; ".join(r.warnings))
    if r.annotated_image_path:
        print(f"annotated image -> {r.annotated_image_path}")


if __name__ == "__main__":
    _main()
