import open3d as o3d
import numpy as np
import plotly.graph_objects as go
import ast
import copy
import trimesh
import json
import os
import cv2
import argparse


def load_intrinsics_from_file(filepath):
    """
    Loads RGB and Depth intrinsics from a specified text file format.
    
    Args:
        filepath (str): The path to the intrinsics.txt file.

    Returns:
        tuple: A tuple containing two dictionaries (rgb_intrinsics, depth_intrinsics).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
        # The dictionary strings are on the 2nd and 4th lines (index 1 and 3)
        rgb_intrinsics_str = lines[1].strip()
        depth_intrinsics_str = lines[3].strip()
        
        # Use ast.literal_eval to safely parse the string into a dictionary
        rgb_intrinsics = ast.literal_eval(rgb_intrinsics_str)
        depth_intrinsics = ast.literal_eval(depth_intrinsics_str)
        
    return rgb_intrinsics, depth_intrinsics

def create_point_cloud_from_rgbd(rgb_img, depth_img, intrinsic):
    """
    Creates an Open3D PointCloud from RGB and Depth numpy arrays.
    """
    rgb_img = rgb_img.astype(np.uint8)
    depth_img = depth_img.astype(np.float32)
    
    rgb_o3d = o3d.geometry.Image(rgb_img)
    depth_o3d = o3d.geometry.Image(depth_img)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, 
        depth_o3d, 
        depth_scale=1.0, 
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            width=rgb_img.shape[1],
            height=rgb_img.shape[0],
            fx=intrinsic['fx'],
            fy=intrinsic['fy'],
            cx=intrinsic['cx'],
            cy=intrinsic['cy']
        )
    )
    
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd

def visualize_pcd_in_html(pcd, filename="point_cloud.html"):
    """
    Saves a point cloud to an interactive HTML file using Plotly.
    
    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to visualize.
        filename (str): The name of the output HTML file.
    """
    # Extract points and colors from the Open3D point cloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Create a 3D scatter plot
    fig = go.Figure(
        data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,          # Adjust marker size
                color=colors,    # Set color to the point's RGB value
                opacity=0.8
            )
        )]
    )
    
    # Adjust layout for a better view
    fig.update_layout(
        title="Interactive 3D Point Cloud Visualization",
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            aspectmode='data' # This helps in maintaining the aspect ratio
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Write the figure to an HTML file
    fig.write_html(filename)
    print(f"✅ Successfully saved visualization to '{filename}'")

def visualize_segmented_pcd_in_html(scene_pcd, object_pcd, filename="segmented_scene.html"):
    """Saves a visualization with the object highlighted."""
    scene_points = np.asarray(scene_pcd.points)
    object_points = np.asarray(object_pcd.points)
    object_colors = np.asarray(object_pcd.colors)

    # Create two traces: one for the scene (in gray) and one for the object (in color)
    scene_trace = go.Scatter3d(
        x=scene_points[:, 0], y=scene_points[:, 1], z=scene_points[:, 2],
        mode='markers',
        marker=dict(size=2, color='lightgray', opacity=0.5),
        name='Scene'
    )
    object_trace = go.Scatter3d(
        x=object_points[:, 0], y=object_points[:, 1], z=object_points[:, 2],
        mode='markers',
        marker=dict(size=2, color=object_colors),
        name='Object of Interest'
    )
    
    fig = go.Figure(data=[scene_trace, object_trace])
    fig.update_layout(
        title="Segmented Object in Scene Point Cloud",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.write_html(filename)
    print(f"✅ Successfully saved visualization to '{filename}'")



# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate scale factor between a 3D mesh and segmented point cloud from RGBD images.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing input files.")
    parser.add_argument("--object_name", type=str, default="", help="Name of the object")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera to use as target point cloud")
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    
    # 1. Define file paths
    scene_rgb1_path = os.path.join(args.data_dir, "camera0_rgb.npy")
    scene_depth1_path = os.path.join(args.data_dir, "camera0_depth.npy")
    intrinsics1_path = os.path.join("./cam_utils", "cam0_intrinsics.txt")
    object_mask1_path = os.path.join(args.data_dir, f"camera0_mask_{args.object_name}.npy")
    extrinsics1_path = os.path.join("./cam_utils", "optimized_transform1_0103_1751.txt")
    
    scene_rgb2_path = os.path.join(args.data_dir, "camera1_rgb.npy")
    scene_depth2_path = os.path.join(args.data_dir, "camera1_depth.npy")
    intrinsics2_path = os.path.join("./cam_utils", "cam1_intrinsics.txt")
    object_mask2_path = os.path.join(args.data_dir, f"camera1_mask_{args.object_name}.npy")
    extrinsics2_path = os.path.join("./cam_utils", "optimized_transform2_0103_1824.txt")
    
    object_mesh_path = os.path.join(args.data_dir, f"{args.object_name}_textured.glb")

    # 2. Load intrinsics from the text file
    try:
        rgb1_intrinsics, depth1_intrinsics = load_intrinsics_from_file(intrinsics1_path)
        scene_rgb1_np = np.load(scene_rgb1_path)
        scene_depth1_np = np.load(scene_depth1_path)
        object_mask1_np = np.load(object_mask1_path)
        camera1_to_robot = np.loadtxt(extrinsics1_path)

        print("\nGenerating point cloud from RGBD data...")
        scene_pcd1 = create_point_cloud_from_rgbd(scene_rgb1_np, scene_depth1_np, rgb1_intrinsics)
        o3d.io.write_point_cloud(os.path.join(args.data_dir, f"scene_pcd0.ply"), scene_pcd1)

        # 4. Extract the object points using the mask
        valid_depth1_mask = (scene_depth1_np > 0.) & (scene_depth1_np <= 3.)
        object_mask_for_pcd1 = object_mask1_np[valid_depth1_mask]
        object_indices_in_pcd1 = np.where(object_mask_for_pcd1)[0]
        target_pcd1 = scene_pcd1.select_by_index(object_indices_in_pcd1)

        if target_pcd1.has_points():
            print(f"Successfully extracted {len(target_pcd1.points)} points for pointcloud 1.")
            target_pcd1.transform(camera1_to_robot)
        else:
            print("⚠️ Warning: No points were selected for pointcloud 1. Check if your masks are empty or misaligned.")    

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}.")
        # exit()

    try:
        rgb2_intrinsics, depth2_intrinsics = load_intrinsics_from_file(intrinsics2_path)
        scene_rgb2_np = np.load(scene_rgb2_path)
        scene_depth2_np = np.load(scene_depth2_path)
        object_mask2_np = np.load(object_mask2_path)
        camera2_to_robot = np.loadtxt(extrinsics2_path)
        
        print("files loaded successfully.")

        scene_pcd2 = create_point_cloud_from_rgbd(scene_rgb2_np, scene_depth2_np, rgb2_intrinsics)
        o3d.io.write_point_cloud(os.path.join(args.data_dir, f"scene_pcd1.ply"), scene_pcd2)

        valid_depth2_mask = (scene_depth2_np > 0.) & (scene_depth2_np <= 3.)
        object_mask_for_pcd2 = object_mask2_np[valid_depth2_mask]
        object_indices_in_pcd2 = np.where(object_mask_for_pcd2)[0]
        target_pcd2 = scene_pcd2.select_by_index(object_indices_in_pcd2)

        if target_pcd2.has_points():
            print(f"Successfully extracted {len(target_pcd2.points)} points for pointcloud 2.")
            target_pcd2.transform(camera2_to_robot)
        else:
            print("⚠️ Warning: No points were selected for pointcloud 2. Check if your masks are empty or misaligned.")

    except Exception as e:
        print(f"❌ Error loading files: {e}")
        # exit()
    
    # 5. Visualize the result
    # visualize_segmented_pcd_in_html(scene_pcd1, target_pcd1, "segmented_scene_camera1.html")
    # visualize_segmented_pcd_in_html(scene_pcd2, target_pcd2, "segmented_scene_camera2.html")
        
    # visualize_pcd_in_html(scene_pcd1, "scene1_visualization.html")
    # visualize_pcd_in_html(scene_pcd2, "scene2_visualization.html")
    # exit()
    
    if args.camera_id == 0:
        target_pcd = target_pcd1
    elif args.camera_id == 1:
        target_pcd = target_pcd2
    elif args.camera_id == 2:
        target_pcd = target_pcd1 + target_pcd2
    else:
        raise ValueError(f"Invalid camera id: {args.camera_id}")
    
    o3d.visualization.draw_geometries([target_pcd])
    # save pcd
    o3d.io.write_point_cloud(os.path.join(args.data_dir,f"{args.object_name}_segmented_object.ply"), target_pcd)
    # exit()
    
    # 6. Load the object's mesh and create the source point cloud
    print("Preparing source point cloud from mesh...")
    try:
        mesh = o3d.io.read_triangle_mesh(object_mesh_path, enable_post_processing=True)
        source_pcd = mesh.sample_points_uniformly(number_of_points=len(target_pcd.points) * 2)
    
    except Exception as e:
        print(f"❌ Error loading mesh: {e}")
    
    # Removing statistical ouliers for better alignment
    print(f"Target cloud has {len(target_pcd.points)} points before outlier removal.")
    print("Removing outliers from target point cloud...")
    cl, ind = target_pcd.remove_statistical_outlier(nb_neighbors=70, std_ratio=0.5)
    target_pcd = target_pcd.select_by_index(ind)
    print(f"Target point cloud cleaned, {len(target_pcd.points)} points remaining.")
    
    if args.debug:
        o3d.visualization.draw_geometries([target_pcd])
    
    # 8. Calculate the scaling factor
    source_bbox = source_pcd.get_axis_aligned_bounding_box()
    source_obox = source_pcd.get_oriented_bounding_box()
    # source_bbox = source_pcd.get_oriented_bounding_box()
    target_bbox = target_pcd.get_axis_aligned_bounding_box()
    target_obox = target_pcd.get_oriented_bounding_box()
    # target_bbox = target_pcd.get_oriented_bounding_box()
    source_size = np.linalg.norm(source_bbox.get_max_bound() - source_bbox.get_min_bound())
    target_size = np.linalg.norm(target_bbox.get_max_bound() - target_bbox.get_min_bound())
    # source_size = source_bbox.get_max_extent()
    # target_size = target_bbox.get_max_extent()
    scale_factor = target_size / source_size

    if args.debug:
        source_bbox.color = (1, 0, 0)  # Red
        source_obox.color = (1, 0, 0)  # Red
        target_obox.color = (0, 1, 0)  # Green
        target_bbox.color = (0, 1, 0)  # Green
        o3d.visualization.draw_geometries([source_pcd, target_pcd, 
                                           source_bbox, target_bbox,
                                           source_obox, target_obox])
    
    # DEBUG: assign a fixed scale factor for certain objects
    scale_factor = 0.15
    
    print(f"Scale factor to match target: {scale_factor:.4f}")
    
    # 9. Apply the scaling
    mesh.scale(scale_factor, center=mesh.get_center())
    # o3d.io.write_point_cloud(os.path.join(args.data_dir, f"{args.object_name}_scaled.ply"), mesh)
    o3d.io.write_triangle_mesh(os.path.join(args.data_dir, f"{args.object_name}_scaled.obj"), mesh)

    if args.debug:
        o3d.visualization.draw_geometries([mesh, target_pcd])
    
    print(f"✅ Scaling complete and mesh saved to {args.object_name}_scaled.obj")
    