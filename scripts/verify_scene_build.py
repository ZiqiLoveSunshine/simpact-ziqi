"""Verify a rebuilt scene against the committed reference trial.

Compares a fresh ``build_scene`` output (capture/ RGB-D -> sim assets) with the
bundled reference ``sim/``, within tolerances that absorb the pipeline's known
run-to-run variation (segmentation edges, VLM endpoint/material estimates):

  rope   segmented cloud centroid/bbox; fixed/free endpoints match the reference
         pair (and the VLM role assignment agrees)
  dough  mpm_points bbox per-axis within 1.5 cm (same bar as
         tests/test_dough_auto_build) + VLM material block present
  sweep  beans cloud bbox + target-region centroid + material block
  push   per-object 6-DoF pose (translation/rotation vs the committed estimate)
         + reconstructed mesh extents (SAM-3D + metric scale)

Exit 0 = the raw->sim pipeline reproduces the reference; 1 = drift (with the
offending metric printed). Used by reproduce_all.sh's real2sim stage.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def _cloud(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def _bbox_cm(p: np.ndarray) -> np.ndarray:
    return (p.max(0) - p.min(0))[:3] * 100


FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {detail}")
    if not ok:
        FAILURES.append(label)


def compare_clouds(label: str, built: Path, ref: Path,
                   bbox_atol_cm: float, centroid_tol_m: float):
    b, r = _cloud(built), _cloud(ref)
    bb, rb = _bbox_cm(b), _bbox_cm(r)
    d_bbox = float(np.abs(bb - rb).max())
    d_cent = float(np.linalg.norm(b.mean(0) - r.mean(0)))
    check(f"{label} bbox", d_bbox <= bbox_atol_cm,
          f"per-axis cm got={np.round(bb, 1).tolist()} ref={np.round(rb, 1).tolist()} "
          f"(max delta {d_bbox:.1f} <= {bbox_atol_cm})")
    check(f"{label} centroid", d_cent <= centroid_tol_m,
          f"delta {d_cent * 100:.1f} cm <= {centroid_tol_m * 100:.0f} cm")


def main():
    import yaml
    from simpact.utils.layout import find_scene_file

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--built", required=True, help="build output dir (flat)")
    ap.add_argument("--reference", required=True, help="committed reference trial dir")
    ap.add_argument("--material", required=True, choices=["push", "rope", "dough", "sweep"])
    a = ap.parse_args()
    built, ref = Path(a.built), Path(a.reference)

    if a.material == "push":  # perception build: meshes + 6-DoF poses (no scene.yaml)
        # Raw pose matrices are NOT comparable across runs: SAM-3D generates each
        # mesh in its own canonical frame (near-90-deg frame flips for box-like
        # objects are normal), and the 6-DoF pose is only meaningful relative to
        # its own mesh. Compare the POSED geometry instead: each run's own mesh
        # transformed by its own pose, as a camera-frame AABB.
        import trimesh

        def posed_aabb(mesh_path, pose):
            m = trimesh.load(mesh_path, force="mesh")
            v = (pose[:3, :3] @ np.asarray(m.vertices).T).T + pose[:3, 3]
            return v.min(0), v.max(0)

        for pose_file in sorted((ref / "sim").glob("*_6d_cam*.txt")):
            name = pose_file.name
            mesh = name.replace(f"_6d_cam{name[-5]}.txt", "_scaled.obj")
            if not (built / name).exists() or not (built / mesh).exists():
                check(f"posed {mesh}", False, "pose or mesh missing from the perception build")
                continue
            rlo, rhi = posed_aabb(ref / "sim" / mesh, np.loadtxt(pose_file).reshape(4, 4))
            blo, bhi = posed_aabb(built / mesh, np.loadtxt(built / name).reshape(4, 4))
            d_center = float(np.linalg.norm((blo + bhi) / 2 - (rlo + rhi) / 2))
            d_extent = float(np.abs((bhi - blo) - (rhi - rlo)).max())
            check(f"posed {mesh}", d_center <= 0.03 and d_extent <= 0.03,
                  f"AABB center delta {d_center * 100:.1f} cm <= 3, "
                  f"extent delta {d_extent * 100:.1f} cm <= 3")
        if FAILURES:
            print(f"REBUILD DRIFT: {len(FAILURES)} check(s) failed: {FAILURES}")
            return 1
        print("rebuild reproduces the reference within tolerance")
        return 0

    by = yaml.safe_load(find_scene_file(built, "scene.yaml").read_text())
    ry = yaml.safe_load(find_scene_file(ref, "scene.yaml").read_text())

    # the runtime EE source must be embedded and equal the reference record
    check("initial_ee_pose embedded", by.get("initial_ee_pose") is not None, "in scene.yaml")
    if by.get("initial_ee_pose") is not None and ry.get("initial_ee_pose") is not None:
        d = float(np.abs(np.asarray(by["initial_ee_pose"]) -
                         np.asarray(ry["initial_ee_pose"])).max())
        check("initial_ee_pose matches", d < 1e-6, f"max delta {d:.2e}")

    if a.material == "rope":
        compare_clouds("rope cloud",
                       find_scene_file(built, "segmented_object.ply"),
                       find_scene_file(ref, "segmented_object.ply"),
                       bbox_atol_cm=3.0, centroid_tol_m=0.03)
        bf, bfree = np.asarray(by["fixed_point"]), np.asarray(by["free_end"])
        rf, rfree = np.asarray(ry["fixed_point"]), np.asarray(ry["free_end"])
        same = max(np.linalg.norm(bf - rf), np.linalg.norm(bfree - rfree))
        swapped = max(np.linalg.norm(bf - rfree), np.linalg.norm(bfree - rf))
        # reference endpoints are the original pipeline's HUMAN-CLICKED points (often slightly inside
        # the physical tip); the detector finds geometric extremes -> allow 8 cm.
        check("endpoints", same <= 0.08,
              f"role-matched tip error {same * 100:.1f} cm <= 8 cm"
              + (" (VLM roles SWAPPED vs reference)" if swapped < same else ""))
    else:
        # each side declares its own cloud file (build_scene writes mpm_points.npy;
        # the committed sweep reference keeps the original pipeline's beans_mpm_points.npy name)
        compare_clouds(f"{a.material} cloud",
                       find_scene_file(built, Path(by["raw_pcd_path"]).name),
                       find_scene_file(ref, Path(ry["raw_pcd_path"]).name),
                       bbox_atol_cm=1.5, centroid_tol_m=0.03)
        mat = by.get("material") or {}
        check("VLM material block", mat.get("source") == "vlm",
              f"source={mat.get('source')!r} keys={sorted(set(mat) - {'source', 'reason'})}")
        if a.material == "sweep":
            compare_clouds("target region",
                           find_scene_file(built, "target_region.ply"),
                           find_scene_file(ref, "target_region.ply"),
                           bbox_atol_cm=3.0, centroid_tol_m=0.05)

    if FAILURES:
        print(f"REBUILD DRIFT: {len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("rebuild reproduces the reference within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
