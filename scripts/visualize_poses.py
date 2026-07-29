"""Validate shape completion + pose estimation against the scene point cloud.

Back-projects an RGB-D frame to a coloured 3D point cloud and overlays, per
object, its completed mesh (``{obj}_scaled.obj``) placed at the estimated 6-DoF
pose, plus the observed object points (from ``camera{cam}_mask_{obj}.npy``). If
shape + pose are correct, the posed mesh wraps the observed points and its visible
surface coincides with them.

Everything is shown in the **robot base frame** by default: the camera->robot
extrinsic transforms the cloud and poses (camera-frame ``_6d_cam`` -> robot via
``mujoco_cam = camera_to_robot @ 6d_cam``). The extrinsic is taken from
``--extrinsic``; else derived from the trial's ``{obj}_mujoco_cam`` +
``{obj}_6d_cam`` files; else the the original ``cam_utils`` default. Use ``--frame camera``
to stay in the camera frame.

Outputs (in --out_dir): ``overlay.png`` (multi-view render), ``scene_cloud.ply``
and ``{obj}_posed.obj`` (in the chosen frame, for MeshLab/Open3D), and a printed
per-object fit. ``--interactive`` opens an Open3D 3D window (needs a display).

Example:
  python scripts/visualize_poses.py \
    --data_dir /path/to/data/1111_push_0 --cam 1 \
    --frame robot --interactive --out_dir /tmp/pose_vis
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from simpact.real2sim.transform_6d import get_camera_to_robot

PALETTE = [((0.85, 0.15, 0.15), (0.15, 0.7, 0.15)),
           ((0.95, 0.55, 0.0), (0.1, 0.5, 0.9)),
           ((0.7, 0.2, 0.7), (0.2, 0.7, 0.7))]


def backproject(depth, K, rgb=None, mask=None):
    sel = depth > 0.001
    if mask is not None:
        sel &= mask.astype(bool)
    vs, us = np.where(sel)
    z = depth[vs, us]
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], 1)
    cols = rgb[vs, us, :3] / 255.0 if rgb is not None else None
    return pts, cols


def transform_points(P, T):
    return (T[:3, :3] @ P.T).T + T[:3, 3]


def resolve_extrinsic(cam: int, explicit, data_dir) -> tuple:
    """camera->robot 4x4 + a source label (or (None, ...) for camera frame).

    Prefers an explicit ``--extrinsic`` file, else resolves the extrinsic **per scene**
    from ``data_dir`` (``simpact.real2sim.transform_6d.get_camera_to_robot`` → embedded
    cam files or the ``scene.yaml`` ``camera:`` profile ref into the registry) — the same
    extrinsic the real2sim pipeline uses to build ``_mujoco_cam`` poses. The stored
    per-trial ``_mujoco_cam`` files predate this calibration for older trials, so they are
    deliberately NOT used to derive the frame here.
    """
    if explicit:
        return np.loadtxt(explicit).reshape(4, 4), f"file:{Path(explicit).name}"
    try:
        return get_camera_to_robot(cam, data_dir), f"per-scene extrinsic ({data_dir}, cam{cam})"
    except Exception as e:
        return None, f"none ({e})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--objects", default=None, help='names "a. b."; default: all _6d_cam files')
    ap.add_argument("--frame", choices=["robot", "camera"], default="robot")
    ap.add_argument("--extrinsic", default=None, help="camera->robot 4x4 .txt (else derived/default)")
    ap.add_argument("--k_file", default=None, help="3x3 intrinsics; default cam_utils/cam{cam}_K.txt")
    ap.add_argument("--out_dir", default="/tmp/pose_vis")
    ap.add_argument("--crop", type=float, default=0.35, help="keep scene within this radius of objects (m)")
    ap.add_argument("--max_scene_pts", type=int, default=15000)
    ap.add_argument("--mesh_pts", type=int, default=4000)
    ap.add_argument("--interactive", action="store_true", help="open an Open3D 3D window (needs a display)")
    a = ap.parse_args()

    D = Path(a.data_dir)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rgb = np.load(D / f"camera{a.cam}_rgb.npy")
    depth = np.load(D / f"camera{a.cam}_depth.npy").astype(np.float64)
    K = np.loadtxt(a.k_file or (D.parents[1] / "cam_utils" / f"cam{a.cam}_K.txt")).reshape(3, 3)

    names = ([s.strip() for s in a.objects.replace(".", ",").split(",") if s.strip()] if a.objects
             else sorted(Path(p).name.replace(f"_6d_cam{a.cam}.txt", "")
                         for p in glob.glob(str(D / f"*_6d_cam{a.cam}.txt"))))

    # frame transform (camera -> chosen frame)
    if a.frame == "robot":
        T, src = resolve_extrinsic(a.cam, a.extrinsic, a.data_dir)
        if T is None:
            print("WARNING: no extrinsic found; falling back to camera frame.")
            T, a.frame = np.eye(4), "camera"
        else:
            print(f"frame=robot, camera->robot extrinsic {src}")
    else:
        T = np.eye(4)

    scene_pts, scene_cols = backproject(depth, K, rgb)
    scene_pts = transform_points(scene_pts, T)

    objs, centers = [], []
    for n in names:
        pose_cam = np.loadtxt(D / f"{n}_6d_cam{a.cam}.txt").reshape(4, 4)
        M = T @ pose_cam  # object -> chosen frame
        mesh = trimesh.load(D / f"{n}_scaled.obj", force="mesh")
        Vw = transform_points(np.asarray(mesh.vertices), M)
        mesh_f = trimesh.Trimesh(vertices=Vw, faces=mesh.faces, process=False)
        surf = mesh_f.sample(a.mesh_pts)
        mp = D / f"camera{a.cam}_mask_{n}.npy"
        obs = transform_points(backproject(depth, K, mask=np.load(mp))[0], T) if mp.exists() else np.empty((0, 3))
        fit = float(cKDTree(surf).query(obs)[0].mean()) * 1000 if len(obs) else float("nan")
        objs.append(dict(name=n, mesh=mesh_f, surf=surf, obs=obs, fit=fit, M=M, src_obj=D / f"{n}_scaled.obj"))
        centers.append(M[:3, 3])
        mesh_f.export(out / f"{n}_posed.obj")
        print(f"{n:32s} observed pts={len(obs):6d}  fit(obs->mesh) mean = {fit:6.1f} mm")

    centers = np.array(centers)
    keep = cKDTree(centers).query(scene_pts)[0] < (a.crop + 0.25) if len(centers) else np.ones(len(scene_pts), bool)
    sp, sc = scene_pts[keep], scene_cols[keep]
    if len(sp) > a.max_scene_pts:
        idx = np.random.default_rng(0).choice(len(sp), a.max_scene_pts, replace=False)
        sp, sc = sp[idx], sc[idx]

    try:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(sp); pc.colors = o3d.utility.Vector3dVector(sc)
        o3d.io.write_point_cloud(str(out / "scene_cloud.ply"), pc)
    except Exception as e:
        print(f"(ply export skipped: {e})")

    _render_png(out, sp, sc, objs, a.frame)
    print(f"\nwrote {out}/overlay.png, scene_cloud.ply, *_posed.obj  (frame={a.frame})")

    if a.interactive:
        _interactive(sp, sc, objs, a.frame)


def _render_png(out, sp, sc, objs, frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def toplot(P):  # robot: z-up already; camera: x right, y down, z fwd -> (x, z, -y)
        return (P[:, 0], P[:, 1], P[:, 2]) if frame == "robot" else (P[:, 0], P[:, 2], -P[:, 1])

    views = ([("robot 3/4", 22, -60), ("top", 89, -90), ("side", 6, 0)] if frame == "robot"
             else [("camera view", 18, -75), ("top", 88, -90), ("side", 8, 0)])
    allpts = np.vstack([sp] + [o["surf"] for o in objs])
    fig = plt.figure(figsize=(18, 6))
    for vi, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, vi + 1, projection="3d")
        ax.scatter(*toplot(sp), s=1, c=sc, alpha=0.25)
        for i, o in enumerate(objs):
            oc, mc = PALETTE[i % len(PALETTE)]
            if len(o["obs"]):
                ax.scatter(*toplot(o["obs"]), s=4, c=[oc], label=f"{o['name'][:14]} observed")
            ax.scatter(*toplot(o["surf"]), s=2, c=[mc], alpha=0.5, label=f"{o['name'][:14]} mesh@pose")
        ax.set_title(title); ax.view_init(elev=elev, azim=azim)
        X, Y, Z = toplot(allpts); ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
        if vi == 0:
            ax.legend(loc="upper left", fontsize=7, markerscale=3)
    fig.suptitle(f"pose+shape validation ({frame} frame): cloud vs observed vs posed mesh")
    fig.tight_layout(); fig.savefig(out / "overlay.png", dpi=110)


def _interactive(sp, sc, objs, frame):
    try:
        import open3d as o3d
    except Exception as e:
        print(f"interactive needs open3d: {e}")
        return
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(sp); pc.colors = o3d.utility.Vector3dVector(sc)
    geoms = [pc, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]  # frame at base origin
    for o in objs:  # textured meshes, transformed into the chosen frame
        om = o3d.io.read_triangle_mesh(str(o["src_obj"]), enable_post_processing=True)
        om.transform(o["M"]); om.compute_vertex_normals()
        geoms.append(om)
    print("opening Open3D window (close it to exit)... needs a display / X forwarding")
    try:
        o3d.visualization.draw_geometries(
            geoms, window_name=f"pose+shape validation ({frame} frame)")
    except Exception as e:
        print(f"could not open a window ({e}); run locally with a display, or use overlay.png / the .ply+.obj")


if __name__ == "__main__":
    main()
