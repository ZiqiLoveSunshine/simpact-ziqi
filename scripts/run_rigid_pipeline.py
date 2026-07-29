"""End-to-end rigid real2sim pipeline: RGB-D + language -> MuJoCo scene.

Concrete worked example of docs/RIGID_PIPELINE.md, driving the perception
**adapters** (simpact.real2sim.perception) behind their base.py interfaces:

  1. GroundedSAM2Segmenter.segment()   RGB + text prompt -> per-object masks
  2. mask extraction (glue)            mask -> full mask (.npy) + RGBA crop
  3. SAM3DReconstructor.reconstruct()  RGBA crop -> complete mesh (scale-free)
  4. scale (glue)                      masked depth -> metric mesh (*_scaled.obj)
  5. FoundationPoseEstimator.estimate() scaled mesh + RGB-D -> 6-DoF pose (cam)
  6. world-align (glue)                fit table plane (depth) -> z-up world poses
  7. generate XML (glue)               meshes + poses -> scene.xml (MuJoCo-ready)

SAM-3D (~20 GB) and FoundationPose are loaded in two phases (reconstruct all ->
free SAM-3D -> pose all) so they are never peak-resident together.

Example (--data_dir = a recorded RGB-D trial; the committed example works out of the box):
  .venv/bin/python scripts/run_rigid_pipeline.py \
      --data_dir examples/push_real2sim/0103_push_0 \
      --objects "white coconut milk carton. blue milk carton." --cam 1 \
      --K assets/calibration/0103/cam1_K.txt \
      --out_dir /tmp/rigid_demo --xml /tmp/rigid_demo/scene.xml
"""
import argparse
import gc
import os
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# point the adapters at the in-repo clones / local checkpoint unless overridden
os.environ.setdefault("SIMPACT_SAM3D_DIR", f"{REPO}/external/sam-3d-objects")
os.environ.setdefault("SIMPACT_FOUNDATIONPOSE_DIR", f"{REPO}/external/FoundationPose")
os.environ.setdefault("SIMPACT_SAM2_CHECKPOINT",
                      os.path.expanduser("~/sam2/checkpoints/sam2.1_hiera_large.pt"))

from simpact.real2sim.perception import (  # noqa: E402
    GroundedSAM2Segmenter, SAM3DReconstructor, FoundationPoseEstimator,
)


def log(msg):
    print(f"\n=== {msg} ===", flush=True)


# ---- glue: object-name <-> detection association --------------------------- #
def match(requested, labels, scores):
    out = {}
    for name in requested:
        cands = [(i, s) for i, (l, s) in enumerate(zip(labels, scores))
                 if l == name or name in l or l in name]
        if not cands:
            print(f"  ! no detection for {name!r} (labels={labels}) — skipped")
            continue
        out[name] = max(cands, key=lambda t: t[1])[0]
    return out


# ---- glue: mask -> RGBA crop (object on transparent bg) -------------------- #
def rgba_crop(rgb, mask, box, pad=10):
    H, W = mask.shape
    x0, y0, x1, y1 = (int(max(0, box[0] - pad)), int(max(0, box[1] - pad)),
                      int(min(W, box[2] + pad)), int(min(H, box[3] + pad)))
    a = (mask[y0:y1, x0:x1].astype(np.uint8) * 255)
    return np.dstack([rgb[y0:y1, x0:x1], a])


# ---- glue: metric scale from masked depth (bbox-diagonal ratio) ------------ #
def backproject(depth, mask, K):
    ys, xs = np.where(mask & (depth > 0.001))
    z = depth[ys, xs]
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1)


def bbox_diag(P):
    return float(np.linalg.norm(P.max(0) - P.min(0)))


def metric_scale(mesh, depth, mask, K):
    import open3d as o3d
    P = backproject(depth, mask, K)
    # drop scattered depth outliers (mask edge bleed onto table/background) that
    # would inflate the observed bbox and over-scale the mesh.
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(P)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    target = bbox_diag(np.asarray(pcd.points))
    s = target / bbox_diag(mesh.vertices)
    m = mesh.copy()
    c = m.vertices.mean(0)
    m.vertices = (m.vertices - c) * s + c   # scale about center (matching the real-robot convention)
    return m, s, target


# ---- glue: fit table plane from depth -> z-up world (no calibration) ------- #
def world_from_cam(depth, K):
    import open3d as o3d
    P = backproject(depth, depth > 0.001, K)
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(P)
    plane, _ = pcd.segment_plane(distance_threshold=0.006, ransac_n=3, num_iterations=2000)
    n = np.array(plane[:3]); d = plane[3]
    nn = np.linalg.norm(n); n = n / nn; d = d / nn
    if d < 0:
        n, d = -n, -d
    z_axis = n
    ref = np.array([1.0, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1.0, 0])
    x_axis = np.cross(ref, z_axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis], 0)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [0, 0, d]
    return T


# ---- glue: write a self-contained MuJoCo scene ----------------------------- #
def decimate(mesh, target_tris=40000):
    if len(mesh.faces) <= target_tris:
        return mesh
    import open3d as o3d
    om = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
    om = om.simplify_quadric_decimation(target_tris)
    return trimesh.Trimesh(vertices=np.asarray(om.vertices), faces=np.asarray(om.triangles))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--out_dir", default="/tmp/rigid_demo")
    ap.add_argument("--xml", default=None)
    ap.add_argument("--K", default=None,
                    help="3x3 intrinsics .txt; default <data_dir>/cam{cam}_K.txt")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    xml_path = a.xml or str(out / "scene.xml")
    requested = [s.strip() for s in a.objects.replace(".", ",").split(",") if s.strip()]

    from simpact.utils.layout import find_scene_file
    d = a.data_dir
    # RGB may be stored as .npy (raw capture) or an image (.png/.jpg — the committed
    # examples standardized on .png); accept either. Files resolve through the
    # bundled-trial layout (capture/) or a flat dir.
    _npy = find_scene_file(d, f"camera{a.cam}_rgb.npy", required=False)
    if _npy is not None:
        rgb = np.load(_npy)
    else:
        _img = next((p for e in (".png", ".jpg", ".jpeg")
                     if (p := find_scene_file(d, f"camera{a.cam}_rgb{e}", required=False))), None)
        if _img is None:
            raise FileNotFoundError(f"no camera{a.cam}_rgb.(npy|png|jpg) in {d}")
        rgb = np.array(Image.open(_img).convert("RGB"))
    depth = np.load(find_scene_file(d, f"camera{a.cam}_depth.npy")); depth[depth < 0.001] = 0
    K = np.loadtxt(a.K or find_scene_file(d, f"cam{a.cam}_K.txt")).reshape(3, 3)
    print(f"scene: rgb {rgb.shape} depth {depth.shape} | objects: {requested}")

    import torch
    torch.cuda.reset_peak_memory_stats()

    log("Stage 1+2: GroundedSAM2Segmenter.segment()")
    seg = GroundedSAM2Segmenter(device="cuda")
    sr = seg.segment(rgb, a.objects)
    print(f"  detected: {list(zip(sr.labels, [round(float(s),2) for s in sr.scores]))}")
    idx = match(requested, sr.labels, sr.scores)
    del seg; gc.collect(); torch.cuda.empty_cache()

    log("Stages 3-5: SAM3DReconstructor.reconstruct() + metric scale")
    recon = SAM3DReconstructor(seed=a.seed)
    results = []
    for name, i in idx.items():
        print(f"  [{name}]")
        mask, box = sr.masks[i], sr.boxes[i]
        np.save(out / f"{name}_mask.npy", mask)                          # stage 3
        crop_path = out / f"{name}_cropped.png"
        Image.fromarray(rgba_crop(rgb, mask, box)).save(crop_path)
        rec = recon.reconstruct(crop_path, out)                          # stage 4
        raw = trimesh.load(rec.mesh_path, force="mesh")
        scaled, s, tgt = metric_scale(raw, depth, mask, K)               # stage 5
        mesh_file = str(out / f"{name}_scaled.obj")
        decimate(scaled).export(mesh_file)
        vc = (np.asarray(raw.visual.vertex_colors)[:, :3].mean(0) / 255.0
              if hasattr(raw.visual, "vertex_colors") else np.array([0.6, 0.6, 0.6]))
        results.append(dict(name=name, mesh_file=mesh_file, mask=mask.astype(bool), rgba=vc))
        print(f"    mesh {len(raw.vertices)} verts | scale x{s:.3f} -> obj diag {tgt*100:.1f} cm")
    del recon; gc.collect(); torch.cuda.empty_cache()                    # free SAM-3D

    log("Stage 6: FoundationPoseEstimator.estimate()")
    fp = FoundationPoseEstimator(est_refine_iter=5)
    # camera->robot extrinsic from the scene's calibration (if any), so we can also emit the
    # robot-frame pose ({name}_mujoco_cam{cam}.txt) the planning context builder reads.
    try:
        from simpact.real2sim.camera_calibration import load_camera
        cam2robot = load_camera(a.data_dir, a.cam).cam_to_robot
    except Exception as e:  # noqa: BLE001 - optional; downstream can transform 6d_cam itself
        cam2robot = None
        print(f"  (no extrinsic -> skipping mujoco_cam poses: {e})")
    for r in results:
        pe = fp.estimate(rgb, depth, r["mask"], K, mesh_path=r["mesh_file"],
                         object_name=r["name"], camera_id=a.cam)
        r["pose_cam"] = pe.pose_cam.copy()  # keep camera-frame pose for the overlay
        r["pose"] = pe.pose_cam
        # Persist the camera-frame 6-DoF pose so this build dir is directly consumable by
        # the downstream planning loop (push_scene.build_objects reads
        # {name}_6d_cam{cam}.txt + {name}_scaled.obj and lifts to robot frame itself).
        np.savetxt(out / f"{r['name']}_6d_cam{a.cam}.txt", pe.pose_cam)
        if cam2robot is not None:  # robot-frame pose for the planning context builder
            np.savetxt(out / f"{r['name']}_mujoco_cam{a.cam}.txt", cam2robot @ pe.pose_cam)
        print(f"  [{r['name']}] pose t={r['pose'][:3,3].round(3)} (camera frame)")

    # Validation overlay: draw each estimated 6-DoF box + axes on the RGB. A
    # tight box around each object == correct pose+scale; an oversized box flags
    # a scale error.
    log("Validation: 6-DoF pose overlay on RGB")
    vis = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
    for r in results:
        vis = fp.draw_pose(vis, K, r["mesh_file"], r["pose_cam"])
    overlay_path = out / "pose_overlay.png"
    Image.fromarray(vis).save(overlay_path)
    print(f"  -> {overlay_path}")

    log("Stage 7: world alignment (table-plane fit) + settle on table")
    T = world_from_cam(depth, K)
    for r in results:
        r["pose"] = T @ r["pose"]
        m = trimesh.load(r["mesh_file"], force="mesh")
        Vw = (r["pose"][:3, :3] @ np.asarray(m.vertices).T).T + r["pose"][:3, 3]
        r["pose"][2, 3] -= Vw[:, 2].min()  # lowest vertex rests at z=0
        print(f"  {r['name']}: world xy=({r['pose'][0,3]:.2f},{r['pose'][1,3]:.2f}) "
              f"z0={r['pose'][2,3]*100:.1f} cm")

    log("Stage 8: write MuJoCo XML")
    from simpact.real2sim.scene import build_mujoco_scene
    build_mujoco_scene(results, xml_path, with_gripper=False)
    print(f"  -> {xml_path}")
    print(f"  peak GPU mem = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print("\nRIGID_PIPELINE_OK")


if __name__ == "__main__":
    main()
