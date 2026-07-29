"""Offline scene builder: a raw RGB-D bundle -> a simpact scene dir + minimal scene.yaml.

Mirrors the original pipeline's real2sim procedure (segment -> mask -> back-project masked depth of ONE
selected camera -> write scene.yaml) but runs OFFLINE (docs/DEFORMABLE_INTEGRATION.md
§13/§14). Four deliberate deviations from the original pipeline's live pipeline:
  * no live get_stream capture — consume a recorded bundle (camera{cam}_rgb.{npy,png} +
    camera{cam}_depth.npy + intrinsics + cam{cam}_to_robot.txt);
  * rope pick_points (human) -> VLM endpoint grounding (simpact.generator.ground);
  * live robot.state.O_T_EE -> a RECORDED EE-pose file (see _resolve_ee_pose);
  * emit the minimal scene.yaml schema (no dead init_gripper_pose / init_mpm_center).

The segmenter and the VLM are injectable so the whole thing is unit-testable with fakes
(no GPU / API / external repo). A live run needs the Grounded-SAM-2 env + a raw bundle.

The EE pose is the one input NOT derivable from RGB-D (robot proprioception). The original pipeline's
capture never persisted it (get_stream saves only RGB-D) and read it live; offline we
load it from a recorded file. Missing -> hard error unless allow_home_pose is set.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from simpact.generator.context import EEPose
from simpact.generator.ground import ground_rope_endpoints
from simpact.generator.vlm import GenerateFn, gemini_generate
from simpact.real2sim.estimate_scale import (
    create_point_cloud_from_rgbd,
    load_intrinsics_from_file,
)
from simpact.real2sim.prepare_dough_asset import sample_between_surface_and_table_columns

ROPE_MATERIALS = {"rope"}
MPM_MATERIALS = {"dough", "sweep"}
# Gripper geometry is a fixed property of THIS gripper, not a per-scene observation.
GRIPPER_DIMS = (0.025, 0.1, 0.04)  # dx dy dz (m) — matches the bundled context.txt

# World-frame block + generic rope rules for context.txt (per-task text, not observed).
_WORLD_BLOCK = (
    "--- World Coordinate System ---\n"
    "+x: out of screen, or downwards in image\n"
    "+y: rightward\n"
    "+z: upward\n"
)


@dataclass
class BuildResult:
    scene_dir: Path
    material: str
    cloud_path: Path
    scene_yaml: Path
    context_txt: Optional[Path] = None
    ee_source: str = ""
    warnings: list = field(default_factory=list)


# --- IO helpers --------------------------------------------------------------

def _load_rgb(path: Path) -> np.ndarray:
    """Load an RGB image (HxWx3, uint8) from .npy or .png/.jpg."""
    if path.suffix == ".npy":
        return np.load(path).astype(np.uint8)
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _K_to_intr(K: np.ndarray) -> dict:
    return {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2])}


def _load_intrinsics(raw: Path, cam: int, profile: Optional[str] = None) -> dict:
    """RGB intrinsics dict {fx,fy,cx,cy}: a legacy cam{cam}_intrinsics.txt or 3x3
    cam{cam}_K.txt in the raw bundle, else the registry ``profile``."""
    fi = raw / f"cam{cam}_intrinsics.txt"
    if fi.exists():
        return load_intrinsics_from_file(fi)[0]
    fk = raw / f"cam{cam}_K.txt"
    if fk.exists():
        return _K_to_intr(np.loadtxt(fk).reshape(3, 3))
    if profile:
        from simpact.real2sim.camera_calibration import load_profile
        return _K_to_intr(load_profile(profile, cam).K)
    raise FileNotFoundError(f"no intrinsics for cam{cam} in {raw} (and no --profile)")


def _load_extrinsic(raw: Path, cam: int, profile: Optional[str] = None) -> np.ndarray:
    """4x4 camera->robot: raw cam{cam}_to_robot.txt, else the registry ``profile``."""
    f = raw / f"cam{cam}_to_robot.txt"
    if f.exists():
        return np.loadtxt(f).reshape(4, 4)
    if profile:
        from simpact.real2sim.camera_calibration import load_profile
        return load_profile(profile, cam).cam_to_robot
    raise FileNotFoundError(f"no cam{cam}_to_robot.txt in {raw} (and no --profile)")


def _K_from_intrinsics(intr: dict) -> np.ndarray:
    return np.array([[intr["fx"], 0, intr["cx"]],
                     [0, intr["fy"], intr["cy"]],
                     [0, 0, 1.0]])


def _segment_mask(segmenter, rgb: np.ndarray, prompt: str) -> np.ndarray:
    """Union of all masks the segmenter returns for ``prompt`` -> bool (H,W)."""
    res = segmenter.segment(rgb, prompt)
    if res.masks is None or len(res.masks) == 0:
        raise ValueError(f"segmenter found nothing for prompt {prompt!r}")
    return np.any(np.asarray(res.masks).astype(bool), axis=0)


def _object_cloud(rgb, depth, mask, intr, cam_to_robot):
    """Back-project the masked depth to a robot-frame Open3D cloud (original real2sim procedure)."""
    depth = depth.astype(np.float32).copy()
    depth[~mask] = 0.0  # keep only the object's depth, matching the original pipeline
    pcd = create_point_cloud_from_rgbd(rgb, depth, intr)
    pcd.transform(np.asarray(cam_to_robot, dtype=float).reshape(4, 4))
    return pcd


# --- EE-pose resolution (the one non-RGB-D input) ----------------------------

def _resolve_ee_pose(raw: Path, ee_pose_path: Optional[str],
                     allow_home_pose: bool) -> tuple[EEPose, str]:
    """Load the initial EE pose from a recorded file (never derived from RGB-D).

    Resolution order: explicit ``ee_pose_path`` (a pose file or a context.txt) ->
    auto-discovered files in the raw bundle -> hard error (or the generic home pose only
    if ``allow_home_pose``). Returns (pose, human-readable source).
    """
    def _load_any(p: Path) -> EEPose:
        return EEPose.from_context_file(p) if p.name == "context.txt" \
            else EEPose.from_file(p)

    if ee_pose_path:
        p = Path(ee_pose_path)
        if not p.exists():
            raise FileNotFoundError(f"--ee-pose {p} not found")
        return _load_any(p), str(p)

    for name in ("initial_ee_pose.txt", "robot_state.txt", "ee_pose.txt", "context.txt"):
        p = raw / name
        if p.exists():
            return _load_any(p), str(p)

    if allow_home_pose:
        from simpact.executor.rollout import (
            HOME_GRIPPER_ORIENTATION,
            HOME_GRIPPER_POSITION,
        )
        w, x, y, z = HOME_GRIPPER_ORIENTATION  # stored wxyz
        return EEPose.from_xyz_quat(HOME_GRIPPER_POSITION, [x, y, z, w]), \
            "generic home pose (FALLBACK)"

    raise FileNotFoundError(
        "no recorded EE pose found (looked for --ee-pose and initial_ee_pose.txt / "
        "robot_state.txt / context.txt in the bundle). The EE pose is robot "
        "proprioception, not in the RGB-D — record it at capture, or pass "
        "--allow-home-pose to use the generic home pose as a stand-in.")


def save_ee_pose_from_robot(host: str, out_path) -> Path:
    """Capture-time helper: persist the live robot's O_T_EE next to the RGB-D so future
    bundles are self-contained (the EE-save that the original pipeline's get_stream never did). Writes a
    4x4 matrix to ``out_path``. franky is optional/hardware — guarded."""
    try:
        from franky import Robot
    except ImportError as e:  # hardware optional
        raise RuntimeError("franky not installed; cannot capture the live EE pose") from e
    T = np.asarray(Robot(host).state.O_T_EE.matrix, dtype=float).reshape(4, 4)
    out_path = Path(out_path)
    np.savetxt(out_path, T)
    return out_path


# --- context.txt (rope) ------------------------------------------------------

def _write_context_txt(path: Path, ee: EEPose, fixed_pt, free_end) -> None:
    """Emit a rope context.txt in the original pipeline's format from the resolved EE pose + endpoints.

    Only the EE pose and endpoints are observed data; gripper dims and the world block
    are fixed config. The per-task 'Critical Rules' live in the context TEMPLATE
    (generator/contexts), so they are intentionally not duplicated here."""
    q = ee.quaternion_xyzw
    wxyz = (q[3], q[0], q[1], q[2])
    p = ee.position
    lines = [
        "",
        f"initial robot end effector position (x y z): {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}",
        f"initial robot end effector orientation (w x y z): "
        f"{wxyz[0]:.4f} {wxyz[1]:.4f} {wxyz[2]:.4f} {wxyz[3]:.4f}",
        f"initial robot end effector yaw (radians): {ee.yaw}",
        f"robot end effector dimensions (dx dy dz): {GRIPPER_DIMS[0]}, "
        f"{GRIPPER_DIMS[1]}, {GRIPPER_DIMS[2]}",
        "",
        f"rope free end position (x y z): {free_end[0]:.4f} {free_end[1]:.4f} {free_end[2]:.4f}",
        f"rope fixed end position (x y z): {fixed_pt[0]:.4f} {fixed_pt[1]:.4f} {fixed_pt[2]:.4f}",
        "",
        _WORLD_BLOCK,
    ]
    path.write_text("\n".join(lines))


# --- the builder -------------------------------------------------------------

def build_scene(raw_dir, out_dir, material: str, object_prompt: str, *,
                cam: int = 1, bg_prompt: Optional[str] = None,
                object_name: Optional[str] = None,
                ee_pose_path: Optional[str] = None, table_z: Optional[float] = None,
                segmenter=None, generate_fn: GenerateFn = gemini_generate,
                allow_home_pose: bool = False, debug: bool = False,
                profile: Optional[str] = None, embed_calibration: bool = False) -> BuildResult:
    """Build a simpact scene dir from a recorded RGB-D bundle (offline, no robot).

    ``segmenter`` defaults to Grounded-SAM-2 (lazy). Rope endpoints come from the VLM
    grounding pipeline; MPM clouds from column volume-sampling. Emits the minimal
    scene.yaml (+ context.txt for rope).
    """
    import open3d as o3d
    import yaml

    material = material.lower()
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: list = []

    # --- load the recorded observation + calibration ------------------------
    rgb_path = next((raw / f"camera{cam}_rgb{s}" for s in (".npy", ".png", ".jpg")
                     if (raw / f"camera{cam}_rgb{s}").exists()), None)
    if rgb_path is None:
        raise FileNotFoundError(f"no camera{cam}_rgb.(npy|png|jpg) in {raw}")
    rgb = _load_rgb(rgb_path)
    depth = np.load(raw / f"camera{cam}_depth.npy")
    intr = _load_intrinsics(raw, cam, profile)
    cam_to_robot = _load_extrinsic(raw, cam, profile)
    ee, ee_source = _resolve_ee_pose(raw, ee_pose_path, allow_home_pose)
    if "FALLBACK" in ee_source:
        warnings.append("using generic home pose (no recorded EE pose found)")

    if segmenter is None:  # lazy: heavy import only for a live run
        from simpact.real2sim.perception.grounded_sam2 import GroundedSAM2Segmenter
        segmenter = GroundedSAM2Segmenter()

    # --- shared: emit the scene bundle ---------------------------------------
    # Per-capture data (rgb/depth/ee) is always embedded. Calibration is a REGISTRY
    # REFERENCE by default (``camera: {profile}`` in scene.yaml, when --profile is used)
    # and only materialized into cam files when embedding a portable, registry-independent
    # bundle (--embed-calibration, or when calibration came from raw cam files).
    ref_mode = profile is not None and not embed_calibration
    from PIL import Image
    Image.fromarray(rgb).save(out / f"camera{cam}_rgb.png")
    np.save(out / f"camera{cam}_depth.npy", depth.astype(np.float32))
    np.savetxt(out / "initial_ee_pose.txt", ee.to_matrix())  # recorded EE pose (4x4)
    if ref_mode:
        import yaml as _yaml  # seed scene.yaml with the ref (rope grounding reads it back)
        (out / "scene.yaml").write_text(_yaml.dump({"camera": {"profile": profile, "cam": cam}}))
    else:  # embed the resolved calibration -> portable bundle
        np.savetxt(out / f"cam{cam}_K.txt", _K_from_intrinsics(intr))
        np.savetxt(out / f"cam{cam}_to_robot.txt", cam_to_robot)

    # --- segment + back-project the object ----------------------------------
    mask = _segment_mask(segmenter, rgb, object_prompt)
    obj_pcd = _object_cloud(rgb, depth, mask, intr, cam_to_robot)
    obj_pts = np.asarray(obj_pcd.points)
    if len(obj_pts) == 0:
        raise ValueError("object cloud is empty after masking/back-projection")

    if material in ROPE_MATERIALS:
        o3d.io.write_point_cloud(str(out / "segmented_object.ply"), obj_pcd)
        # VLM grounding: detect tips -> annotate on the real RGB -> assign fixed/free,
        # and write fixed_point/free_end (+ provenance) into scene.yaml.
        # save_annotated only in debug mode — the A/B overlay is a grounding debug
        # artifact, not part of the canonical scene bundle.
        gr = ground_rope_endpoints(out, cam=cam, generate_fn=generate_fn, write=True,
                                   save_annotated=debug)
        # embed the runtime EE source — sim consumers read only scene.yaml
        import yaml as _yaml
        _yml = out / "scene.yaml"
        _y = _yaml.safe_load(_yml.read_text()) or {}
        _y["initial_ee_pose"] = ee.to_matrix().tolist()
        _yml.write_text(_yaml.dump(_y, indent=2))
        ctx = out / "context.txt"
        _write_context_txt(ctx, ee, gr.fixed_point, gr.free_end)
        warnings += gr.warnings
        return BuildResult(out, material, out / "segmented_object.ply",
                           out / "scene.yaml", ctx, ee_source, warnings)

    if material in MPM_MATERIALS:
        if material == "sweep" and not bg_prompt:  # fail fast, before the material VLM call
            raise ValueError("sweep needs --bg (the target-region prompt)")
        tz = float(obj_pts[:, 2].min()) if table_z is None else float(table_z)
        mpm_pts = sample_between_surface_and_table_columns(obj_pts, tz)
        np.save(out / "mpm_points.npy", mpm_pts)
        scene_dict = {
            "object_name": object_name or object_prompt,
            "raw_pcd_path": "mpm_points.npy",
            "initial_ee_pose": ee.to_matrix().tolist(),
        }
        # VLM material-ID: estimate the object's physical params from the scene image and
        # write them per-scene (real2sim infers physics like it infers geometry, §15).
        from simpact.generator.material import estimate_material
        from PIL import Image
        scene_dict["material"] = estimate_material(
            Image.fromarray(rgb), object_name or object_prompt, material,
            generate_fn=generate_fn)
        if ref_mode:  # calibration by registry reference (else cam files were embedded)
            scene_dict["camera"] = {"profile": profile, "cam": cam}
        if material == "sweep":
            bg_mask = _segment_mask(segmenter, rgb, bg_prompt)
            bg_pcd = _object_cloud(rgb, depth, bg_mask, intr, cam_to_robot)
            o3d.io.write_point_cloud(str(out / "target_region.ply"), bg_pcd)
            scene_dict["bg_pcd_path"] = "target_region.ply"
        (out / "scene.yaml").write_text(yaml.dump(scene_dict, indent=2))
        return BuildResult(out, material, out / "mpm_points.npy",
                           out / "scene.yaml", None, ee_source, warnings)

    raise ValueError(f"unknown material {material!r} (rope|dough|sweep)")


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build a simpact scene from a raw RGB-D bundle.")
    ap.add_argument("--raw-dir", required=True, help="recorded bundle (rgb/depth/calib)")
    ap.add_argument("--out-dir", required=True, help="output scene dir")
    ap.add_argument("--material", required=True, choices=["rope", "dough", "sweep"])
    ap.add_argument("--object", required=True, help='segmentation prompt, e.g. "rope"')
    ap.add_argument("--bg", default=None, help="sweep target-region prompt")
    ap.add_argument("--object-name", default=None, help="scene.yaml label (MPM)")
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--ee-pose", default=None, help="recorded EE pose file or context.txt")
    ap.add_argument("--table-z", type=float, default=None, help="MPM table height (else cloud min z)")
    ap.add_argument("--allow-home-pose", action="store_true",
                    help="fall back to generic home pose if no recorded EE pose (warns)")
    ap.add_argument("--debug", action="store_true",
                    help="also write the rope grounding annotated A/B overlay image")
    ap.add_argument("--profile", default=None,
                    help="calibration registry profile (e.g. 1026, 0103); with this the "
                         "scene references the registry (camera: {profile}) instead of "
                         "embedding cam files")
    ap.add_argument("--embed-calibration", action="store_true",
                    help="materialize cam{cam}_K/to_robot into the scene (portable, "
                         "registry-independent bundle) instead of a profile reference")
    a = ap.parse_args()
    r = build_scene(a.raw_dir, a.out_dir, a.material, a.object, cam=a.cam,
                    bg_prompt=a.bg, object_name=a.object_name, ee_pose_path=a.ee_pose,
                    table_z=a.table_z, allow_home_pose=a.allow_home_pose, debug=a.debug,
                    profile=a.profile, embed_calibration=a.embed_calibration)
    print(f"built {r.material} scene -> {r.scene_dir}")
    print(f"  cloud: {r.cloud_path.name}   scene.yaml: {r.scene_yaml.name}"
          + (f"   context.txt: {r.context_txt.name}" if r.context_txt else ""))
    print(f"  EE pose from: {r.ee_source}")
    if r.warnings:
        print("  warnings: " + "; ".join(r.warnings))


if __name__ == "__main__":
    _main()
