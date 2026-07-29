"""Stage 7 of the real2sim pipeline: lift a per-camera 6-DoF object pose into
the robot base frame.

The object pose from stage 6 (FoundationPose) is expressed object->camera. This
composes it with the fixed camera->robot extrinsic to produce the object pose in
the robot base frame, which stage 8 (``generate_xml``) consumes.

Ported from the original ``real2sim/transform_6d.py``. The original version also transformed
the mesh / scene point cloud for an interactive debug view; that is preserved
behind ``debug`` (open3d imported lazily) but is irrelevant to the saved output,
which is simply ``extrinsic @ pose``. The extrinsic is resolved **per scene** from the
trial dir (embedded cam files or a ``scene.yaml`` ``camera:`` profile reference into the
``assets/calibration/`` registry) — never a code-baked default.
"""
import argparse
from pathlib import Path

import numpy as np

from simpact.real2sim.camera_calibration import load_camera


def get_camera_to_robot(camera_id, data_dir):
    """4x4 camera->robot extrinsic, resolved **per scene** from ``data_dir``: an embedded
    ``cam{id}_{K,to_robot}.txt`` or the ``scene.yaml`` ``camera:`` profile reference into
    the registry (``assets/calibration/<profile>/``). There is no code-baked default — a
    scene with no resolvable calibration raises (via ``load_camera``), so the extrinsic is
    never silently frozen to one rig/date."""
    return load_camera(data_dir, camera_id).cam_to_robot


def transform_to_robot_frame(pose_cam, camera_id, data_dir):
    """Compose an object->camera pose with the camera->robot extrinsic.

    Args:
        pose_cam: (4, 4) object->camera homogeneous transform.
        camera_id: Which camera the pose was estimated in.
        data_dir: scene dir carrying the camera calibration (embedded cam files or a
            ``scene.yaml`` ``camera:`` profile reference) — resolved per scene.

    Returns:
        (4, 4) object pose in the robot base frame.
    """
    return get_camera_to_robot(camera_id, data_dir) @ np.asarray(pose_cam)


def transform_object_pose(data_dir, object_name, camera_id, *, debug=False):
    """Read ``{object}_6d_cam{id}.txt`` from ``data_dir``, lift it into the robot
    frame, and write ``{object}_mujoco_cam{id}.txt``. Returns the final pose.

    Uses the scene's own ``cam{id}_to_robot.txt`` if present (else packaged calibration).
    """
    data_dir = Path(data_dir)
    pose_cam = np.loadtxt(data_dir / f"{object_name}_6d_cam{camera_id}.txt")
    final_pose = transform_to_robot_frame(pose_cam, camera_id, data_dir)

    out_path = data_dir / f"{object_name}_mujoco_cam{camera_id}.txt"
    np.savetxt(out_path, final_pose)
    print(f"Saved final 6D pose to {out_path}")

    if debug:
        _debug_visualize(data_dir, object_name, camera_id, pose_cam)

    return final_pose


def _debug_visualize(data_dir, object_name, camera_id, pose_cam):
    import open3d as o3d  # lazy: open3d is heavy and only needed for the view

    extrinsic = get_camera_to_robot(camera_id, data_dir)
    mesh = o3d.io.read_triangle_mesh(
        str(Path(data_dir) / f"{object_name}_scaled.obj"), enable_post_processing=True
    )
    mesh.transform(pose_cam)
    mesh.transform(extrinsic)

    scene_pcd = o3d.io.read_point_cloud(
        str(Path(data_dir) / f"scene_pcd{camera_id}.ply")
    )
    scene_pcd.transform(extrinsic)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([mesh, scene_pcd, coord])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--object_name", type=str, required=True)
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    final_pose = transform_object_pose(
        args.test_dir, args.object_name, args.camera_id, debug=args.debug
    )
    print(final_pose)
