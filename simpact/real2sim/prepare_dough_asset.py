import numpy as np
import open3d as o3d
import os
import argparse
from simpact.real2sim.estimate_scale import create_point_cloud_from_rgbd, load_intrinsics_from_file
import yaml
import trimesh

try:
    from franky import Affine, Robot
except ImportError:
    Affine = Robot = None  # franky (robot hardware) optional; only needed when run as a script

import numpy as np

def sample_between_surface_and_table_columns(
    surface_xyz: np.ndarray,
    table_z: float,
    dz: float = 0.002,
    include_surface: bool = True,
    include_table: bool = True,
    direction: str = "toward_table",
) -> np.ndarray:
    """
    Densely sample points between a surface point cloud and a horizontal table plane z=table_z
    by sampling along vertical columns for each surface point.

    Args:
        surface_xyz: (N, 3) array of surface points.
        table_z: z value of the table plane.
        dz: vertical sampling step (meters or your unit).
        include_surface: include the original surface point in samples.
        include_table: include the table plane point (x,y,table_z) when reachable.
        direction:
            - "toward_table": sample from surface toward table along z.
            - "both": sample full segment between min(z, table_z) and max(z, table_z).

    Returns:
        samples_xyz: (M, 3) array of sampled points.
    """
    surface_xyz = np.asarray(surface_xyz, dtype=np.float64)
    if surface_xyz.ndim != 2 or surface_xyz.shape[1] != 3:
        raise ValueError("surface_xyz must be shape (N, 3).")
    if dz <= 0:
        raise ValueError("dz must be > 0.")

    xs = surface_xyz[:, 0]
    ys = surface_xyz[:, 1]
    zs = surface_xyz[:, 2]

    out_chunks = []

    for x, y, z in zip(xs, ys, zs):
        if direction == "toward_table":
            z0, z1 = z, table_z
        elif direction == "both":
            z0, z1 = min(z, table_z), max(z, table_z)
        else:
            raise ValueError("direction must be 'toward_table' or 'both'.")

        # If already on the plane (or extremely close), just decide inclusions.
        if np.isclose(z0, z1):
            pts = []
            if include_surface:
                pts.append([x, y, z])
            # surface and table are same here; avoid duplicates
            out_chunks.append(np.array(pts, dtype=np.float64))
            continue

        # Create z samples along the segment
        step = dz if z1 > z0 else -dz
        z_samples = np.arange(z0, z1, step, dtype=np.float64)

        if include_surface:
            if len(z_samples) == 0 or not np.isclose(z_samples[0], z0):
                z_samples = np.concatenate(([z0], z_samples))
        else:
            # if first is exactly z0, drop it
            if len(z_samples) and np.isclose(z_samples[0], z0):
                z_samples = z_samples[1:]

        if include_table:
            # ensure z1 included
            if len(z_samples) == 0 or not np.isclose(z_samples[-1], z1):
                z_samples = np.concatenate((z_samples, [z1]))
        else:
            # if last hits z1, drop it
            if len(z_samples) and np.isclose(z_samples[-1], z1):
                z_samples = z_samples[:-1]

        pts = np.column_stack([np.full_like(z_samples, x), np.full_like(z_samples, y), z_samples])
        out_chunks.append(pts)

    if not out_chunks:
        return np.empty((0, 3), dtype=np.float64)

    return np.vstack(out_chunks)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate scale factor between a 3D mesh and segmented point cloud from RGBD images.")
    parser.add_argument("--data_dir", type=str, default="data/rope", help="Directory containing input files.")
    parser.add_argument("--object_name", type=str, default="rope", help="Name of the object")
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of interior points to sample")
    parser.add_argument("--bg_obj_name", default=None, help="Background object name to save in the scene yaml")
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    robot = Robot(args.host)

    # 1. Define file paths
    scene_rgb1_path = os.path.join(args.data_dir, "camera0_rgb.npy")
    scene_depth1_path = os.path.join(args.data_dir, "camera0_depth.npy")
    intrinsics1_path = os.path.join("./cam_utils", "cam0_intrinsics.txt")
    object_mask1_path = os.path.join(args.data_dir, f"camera0_mask_{args.object_name}.npy")
    if args.bg_obj_name is not None:
        bg_mask1_path = os.path.join(args.data_dir, f"camera0_mask_{args.bg_obj_name}.npy")
    extrinsics1_path = os.path.join("./cam_utils", "optimized_transform1_0103_1751.txt")

    scene_rgb2_path = os.path.join(args.data_dir, "camera1_rgb.npy")
    scene_depth2_path = os.path.join(args.data_dir, "camera1_depth.npy")
    intrinsics2_path = os.path.join("./cam_utils", "cam1_intrinsics.txt")
    object_mask2_path = os.path.join(args.data_dir, f"camera1_mask_{args.object_name}.npy")
    if args.bg_obj_name is not None:
        bg_mask2_path = os.path.join(args.data_dir, f"camera1_mask_{args.bg_obj_name}.npy")
    extrinsics2_path = os.path.join("./cam_utils", "optimized_transform2_0103_1824.txt")

    object_mesh_path = os.path.join(args.data_dir, f"{args.object_name}_textured.glb")

    object_mask1_np = None
    bg_mask1_np = None
    object_mask2_np = None
    bg_mask2_np = None
    # 2. Load intrinsics from the text file
    try:
        rgb1_intrinsics, depth1_intrinsics = load_intrinsics_from_file(intrinsics1_path)
        scene_rgb1_np = np.load(scene_rgb1_path)
        scene_depth1_np = np.load(scene_depth1_path)
        camera1_to_robot = np.loadtxt(extrinsics1_path)

        scene_pcd1 = create_point_cloud_from_rgbd(scene_rgb1_np, scene_depth1_np, rgb1_intrinsics)
        scene_pcd1.transform(camera1_to_robot)
        o3d.io.write_point_cloud(f"{args.data_dir}/scene_pcd_camera1.ply", scene_pcd1)
        
        object_mask1_np = np.load(object_mask1_path)
        if args.bg_obj_name is not None:
            bg_mask1_np = np.load(bg_mask1_path)
        
        print("files loaded successfully.")

    except FileNotFoundError as e:
        print(f"File not found: {e}.")
        # exit()
    
    try:
        rgb2_intrinsics, depth2_intrinsics = load_intrinsics_from_file(intrinsics2_path)
        scene_rgb2_np = np.load(scene_rgb2_path)
        scene_depth2_np = np.load(scene_depth2_path)
        camera2_to_robot = np.loadtxt(extrinsics2_path)

        scene_pcd2 = create_point_cloud_from_rgbd(scene_rgb2_np, scene_depth2_np, rgb2_intrinsics)
        scene_pcd2.transform(camera2_to_robot)
        o3d.io.write_point_cloud(f"{args.data_dir}/scene_pcd_camera2.ply", scene_pcd2)

        object_mask2_np = np.load(object_mask2_path)
        if args.bg_obj_name is not None:
            bg_mask2_np = np.load(bg_mask2_path)
        
        print("files loaded successfully.")
    except FileNotFoundError as e:
        print(f"File not found: {e}.")
        # exit()
    
    # import pdb; pdb.set_trace()

    obj_pcd1 = None
    bg_pcd1 = None
    if object_mask1_np is not None:
        scene_pcd1 = create_point_cloud_from_rgbd(scene_rgb1_np, scene_depth1_np, rgb1_intrinsics)
        scene_pcd1.transform(camera1_to_robot)
        o3d.io.write_point_cloud(f"{args.data_dir}/scene_pcd_camera1.ply", scene_pcd1)
        obj_depth1_np = scene_depth1_np.copy()
        obj_depth1_np[object_mask1_np == 0] = 0
        obj_pcd1 = create_point_cloud_from_rgbd(scene_rgb1_np, obj_depth1_np, rgb1_intrinsics)
        obj_pcd1.transform(camera1_to_robot)
        if args.bg_obj_name is not None:
            bg_depth1_np = scene_depth1_np.copy()
            bg_depth1_np[bg_mask1_np == 0] = 0
            bg_pcd1 = create_point_cloud_from_rgbd(scene_rgb1_np, bg_depth1_np, rgb1_intrinsics)
            bg_pcd1.transform(camera1_to_robot)

    obj_pcd2 = None
    bg_pcd2 = None
    if object_mask2_np is not None:
        scene_pcd2 = create_point_cloud_from_rgbd(scene_rgb2_np, scene_depth2_np, rgb2_intrinsics)
        scene_pcd2.transform(camera2_to_robot)
        o3d.io.write_point_cloud(f"{args.data_dir}/scene_pcd_camera2.ply", scene_pcd2)
        obj_depth2_np = scene_depth2_np.copy()
        obj_depth2_np[object_mask2_np == 0] = 0
        obj_pcd2 = create_point_cloud_from_rgbd(scene_rgb2_np, obj_depth2_np, rgb2_intrinsics)
        obj_pcd2.transform(camera2_to_robot)
        if args.bg_obj_name is not None:
            bg_depth2_np = scene_depth2_np.copy()
            bg_depth2_np[bg_mask2_np == 0] = 0
            bg_pcd2 = create_point_cloud_from_rgbd(scene_rgb2_np, bg_depth2_np, rgb2_intrinsics)
            bg_pcd2.transform(camera2_to_robot)

    obj_pcd2_points = np.asarray(obj_pcd2.points)
    keep_mask = obj_pcd2_points[:, 2] > 0.09
    obj_pcd2_points = obj_pcd2_points[keep_mask, :]
    obj_pcd2_colors = np.asarray(obj_pcd2.colors)
    obj_pcd2_colors = obj_pcd2_colors[keep_mask, :]
    obj_pcd2.points = o3d.utility.Vector3dVector(obj_pcd2_points)
    obj_pcd2.colors = o3d.utility.Vector3dVector(obj_pcd2_colors)

    import pdb; pdb.set_trace()

    merged_pcd = obj_pcd2
    if args.bg_obj_name is not None:
        assert bg_pcd2 is not None
        o3d.io.write_point_cloud(f"{args.data_dir}/background_segmented_surface.ply", bg_pcd2)
    # o3d.visualization.draw_geometries([merged_pcd], window_name="Merged Object Point Cloud")
    hull_mesh, _ = merged_pcd.compute_convex_hull()
    
    o3d.visualization.draw_geometries([hull_mesh, merged_pcd], 
                                      window_name="Convex Hull of Segmented Object Point Cloud")

    # Convert Open3D mesh to trimesh
    vertices = np.asarray(hull_mesh.vertices)
    triangles = np.asarray(hull_mesh.triangles)
    trimesh_hull = trimesh.Trimesh(vertices=vertices, faces=triangles)

    # Sample volumetric points
    interior_points = trimesh.sample.volume_mesh(trimesh_hull, args.num_samples)
    np.save(f"{args.data_dir}/{args.object_name}_mpm_points.npy", interior_points)

    gripper_hmat = robot.state.O_T_EE.matrix

    with open(os.path.join(args.data_dir, "scene.yaml"), 'w') as f:
        # Minimal MPM schema: object label, the particle cloud (relative filename), the
        # EE pose the rollout applies the plan to, and (sweep only) the target region.
        # The MPM centre is computed live from the cloud; init_gripper_pose was dead.
        scene_dict = {
            'object_name': args.object_name,
            'raw_pcd_path': f"{args.object_name}_mpm_points.npy",
            'initial_ee_pose': gripper_hmat.tolist(),
        }
        if args.bg_obj_name is not None:
            scene_dict['bg_pcd_path'] = "background_segmented_surface.ply"
        yaml.dump(scene_dict, f, indent=2)
