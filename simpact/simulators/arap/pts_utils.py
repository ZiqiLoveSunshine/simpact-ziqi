import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
import sklearn.neighbors as skn
import trimesh
import copy
from .vis_utils import display_inlier_outlier, create_motion_lines
import time


def voxel_downsample_with_indices(points, voxel_size=0.05):
    """
    Voxel downsample and get indices of selected points
    
    Returns:
        downsampled_points: Mx3 array of keypoints
        indices: List of lists, where indices[i] contains all original 
                 point indices that belong to voxel i
        original_indices: M array of one representative index per voxel
    """
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Downsample with trace
    downpcd, trace, trace_map = pcd.voxel_down_sample_and_trace(
        voxel_size, 
        min_bound=pcd.get_min_bound(),
        max_bound=pcd.get_max_bound()
    )
    
    # Extract information
    downsampled_points = np.asarray(downpcd.points)
    
    # trace[i] contains indices of original points that map to downsampled point i
    indices_per_voxel = [list(trace[i]) for i in range(len(downsampled_points))]
    
    # Get one representative index per voxel (first point in each voxel)
    representative_indices = np.array([trace[i][0] for i in range(len(downsampled_points))])
    
    return downsampled_points, indices_per_voxel, representative_indices


def get_largest_dbscan_component(pcd, nn_radius=0.01, min_points=30, 
                                 print_progress=False, vis_pcd=False):
    """
    Run DBSCAN clustering on the point cloud and return the largest component.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        nn_radius (float): The radius for DBSCAN clustering.
        min_points (int): The minimum number of points to form a cluster.
        print_progress (bool): Whether to print progress information.
    
    Returns:
        component_pcd (o3d.geometry.PointCloud): The largest component of the point cloud.
    """
    
    # Get the largest component of the fruit point cloud using dbscan clustering
    labels = np.array(pcd.cluster_dbscan(eps=nn_radius, min_points=min_points, 
                                         print_progress=print_progress))
    max_num_label = np.argmax(np.bincount(labels[labels >= 0]))
    component_pcd = pcd.select_by_index(np.where(labels == max_num_label)[0])
    
    if vis_pcd:
        o3d.visualization.draw_geometries([component_pcd])

    return component_pcd

def assign_edge_weights(leaf_pcd, all_sim_pts, connect_ary, w1=3.0, w2=2.0, w3=1.0):
    """
    Heuristic to assign weights to edges based on the distance of points to the 
    central axis of the leaf point cloud.
    
    Args:
        leaf_pcd (o3d.geometry.PointCloud): The input leaf point cloud.
        all_sim_pts (np.ndarray): shape (N, 3), the points to be connected.
        connect_ary (np.ndarray): shape (M, 2), the indices of the points to be connected.
        w1 (float): Weight for points close to the central axis.
        w2 (float): Weight for points moderately close to the central axis.
        w3 (float): Weight for points far from the central axis.

    Returns:
        edge_weights (list): A list of weights for each edge in connect_ary.

    """
    centroid, eigen_values, eigen_vectors = pca_points(np.array(leaf_pcd.points))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03, origin=[0, 0, 0])
    coord_frame.rotate(eigen_vectors, center=[0, 0, 0])
    coord_frame.translate(centroid)

    edge_weights = []
    threshold = 0.006
    for edge in connect_ary:
        pt1, pt2 = all_sim_pts[edge]
        trans_pt1 = np.dot(eigen_vectors.T, pt1 - centroid)
        trans_pt2 = np.dot(eigen_vectors.T, pt2 - centroid)

        y1, y2 = np.abs(trans_pt1[1]), np.abs(trans_pt2[1])

        if y1 < threshold and y2 < threshold:
            edge_weights.append(w1)
        elif y1 < threshold or y2 < threshold:
            edge_weights.append(w2)
        else:
            edge_weights.append(w3)
    return edge_weights

def connect_points(points, nn_radius):
    """
    Find edges between points within a given radius using a radius neighbors graph.
    
    Args:
        points (np.ndarray): shape (N, 3), the points to be connected.
        nn_radius (float): The radius for connecting points.
    
    Returns:
        np.ndarray: shape (M, 2), the indices of the points to be connected.
    """
    neigh = skn.radius_neighbors_graph(points, nn_radius, mode='distance')
    return np.array(neigh.nonzero()).T

def connect_leaf2branch(leaf_pts, branch_pts, nn_radius=0.01, vis_pts=False):
    """
    Find edges between leaf points and branch points within a given radius using a radius neighbors graph.
    
    Args:
        leaf_pts (np.ndarray): shape (N_leaf, 3), the leaf points.
        branch_pts (np.ndarray): shape (N_branch, 3), the branch points.
        nn_radius (float): The radius for connecting points.
        vis_pts (bool): Whether to visualize the points.
    
    Returns:
        np.ndarray: shape (M, 2), the indices of the leaf and branch points to be connected.
    """
    branch_ball_tree = skn.BallTree(branch_pts)

    branch_idx_lst, dist = branch_ball_tree.query_radius(leaf_pts, r=nn_radius, return_distance=True)

    leaf2branch_edges = []
    for i, leaf_indices in enumerate(branch_idx_lst):
        if len(leaf_indices) > 0:
            for j in range(len(leaf_indices)):
                leaf2branch_edges.append([i, leaf_indices[j]])
    leaf2branch_edges = np.array(leaf2branch_edges)
    
    select_leaf_indices = leaf2branch_edges[:, 0]
    select_branch_indices = leaf2branch_edges[:, 1]

    if vis_pts:
        lines = create_motion_lines(leaf_pts[select_leaf_indices, :], branch_pts[select_branch_indices, :])

        leaf_pcd = o3d.geometry.PointCloud()
        leaf_pcd.points = o3d.utility.Vector3dVector(leaf_pts)
        leaf_pcd.paint_uniform_color([0, 0.7, 0.7])

        branch_pcd = o3d.geometry.PointCloud()
        branch_pcd.points = o3d.utility.Vector3dVector(branch_pts)
        branch_pcd.paint_uniform_color([0.7, 0.7, 0])

        o3d.visualization.draw_geometries([leaf_pcd, branch_pcd, lines])

    return np.concatenate([select_leaf_indices.reshape(-1, 1), 
                           select_branch_indices.reshape(-1, 1)], axis=1)

def rotation_vector_to_rotation_matrix(rotation_vector):
    """
    Convert a rotation vector to a rotation matrix using Rodrigues' rotation formula.
    
    Args:
        rotation_vector (np.ndarray): shape (3,), the rotation vector.
    
    Returns:
        np.ndarray: shape (3, 3), the rotation matrix.
    """
    theta = np.linalg.norm(rotation_vector)  # Magnitude of the rotation vector
    if theta < 1e-6:
        # If the angle is very small, close to zero, use the approximation of the identity matrix
        return np.eye(3)
    else:
        # Normalized rotation axis
        axis = rotation_vector / theta
        axis_skew = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        # Rodrigues' rotation formula
        R = np.eye(3) + np.sin(theta) * axis_skew + (1 - np.cos(theta)) * np.dot(axis_skew, axis_skew)
        return R

def rotation_matrix_to_rotation_vector(R):
    """
    Convert a rotation matrix to a rotation vector using the axis-angle representation.
    
    Args:
        R (np.ndarray): shape (3, 3), the rotation matrix.
            
    Returns:
        np.ndarray: shape (3,), the rotation vector.

    Raises:
        ValueError: If the input matrix is not a valid rotation matrix.
    """
    if not np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError("The provided matrix is not a valid rotation matrix.")
    
    # Calculate the angle of rotation
    trace = np.trace(R)
    theta = np.arccos((trace - 1) / 2)
    
    if theta == 0:
        return np.zeros(3)  # No rotation
    elif np.isclose(theta, np.pi):
        # Special case where the angle is pi
        x = np.sqrt((R[0, 0] + 1) / 2)
        y = np.sqrt((R[1, 1] + 1) / 2)
        z = np.sqrt((R[2, 2] + 1) / 2)
        # Determine the signs correctly
        if R[0, 1] < 0: y = -y
        if R[0, 2] < 0: z = -z
        if R[1, 2] < 0: z = -z
        return np.array([x, y, z]) * theta
    else:
        # Normal case
        rx = R[2, 1] - R[1, 2]
        ry = R[0, 2] - R[2, 0]
        rz = R[1, 0] - R[0, 1]
        r = np.array([rx, ry, rz])
        r_normalized = r / (2 * np.sin(theta))
        return r_normalized * theta

def trans_matrix2pose(trans_matrix: np.ndarray) -> np.ndarray:
    """
    Transform (4x4) transformation matrix to pose (6D vector).
    The pose vector is in the format [x, y, z, rotation_vector[0], rotation_vector[1], rotation_vector[2]].
    
    Args:
    
        trans_matrix (np.ndarray): shape (4, 4), the transformation matrix.
    
    Returns:

        np.ndarray: shape (6,), the pose vector in the format 
            [x, y, z, rotation_vector[0], rotation_vector[1], rotation_vector[2]].

    """
    center = trans_matrix[:3, 3]
    rotate = rotation_matrix_to_rotation_vector(trans_matrix[:3, :3])
    return np.append(center, rotate)

def remove_close_points(pcd_source, pcd_query, threshold):
    """
    Remove points from pcd_query that are within a certain distance from pcd_source.
    
    Args:
        pcd_source (o3d.geometry.PointCloud): The source point cloud.   
        pcd_query (o3d.geometry.PointCloud): The query point cloud.
        threshold (float): The distance threshold.
    
    Returns:
        o3d.geometry.PointCloud: The query point cloud with points removed that are 
            within the threshold distance from the source.
    """
    kdtree = o3d.geometry.KDTreeFlann(pcd_source)
    keep_indices = []
    for i, point in enumerate(pcd_query.points):
        [k, idx, dist] = kdtree.search_radius_vector_3d(point, threshold)
        if k == 0:  # No points within the threshold distance
            keep_indices.append(i)
    return pcd_query.select_by_index(keep_indices)

def select_close_points(pcd_source, pcd_query, threshold):
    """
    Get points from pcd_query that are within a certain distance from pcd_source.
    
    Args:
        pcd_source (o3d.geometry.PointCloud): The source point cloud.
        pcd_query (o3d.geometry.PointCloud): The query point cloud.
        threshold (float): The distance threshold.
    
    Returns:
        o3d.geometry.PointCloud: The query point cloud with points selected that are
            within the threshold distance from the source.
    """
    kdtree = o3d.geometry.KDTreeFlann(pcd_source)
    keep_indices = []
    for i, point in enumerate(pcd_query.points):
        [k, idx, dist] = kdtree.search_radius_vector_3d(point, threshold)
        if k > 0:  # No points within the threshold distance
            keep_indices.append(i)
    return pcd_query.select_by_index(keep_indices)

def sample_points_on_hull(points, num_samples):
    """
    Sample points uniformly on the convex hull of a set of points.
    
    Args:
        points (np.ndarray): shape (N, 2), the points to sample from.
        num_samples (int): The number of points to sample.
    
    Returns:
        dict: A dictionary containing the convex hull and the sampled points.
            keys are 'hull' and 'sampled_points'.
    """
    hull = ConvexHull(points)
    distances = []
    segment_lst = []
    print('hull.vertices:', hull.vertices)
    for i in range(len(hull.vertices)):
        curr_idx = hull.vertices[i]
        next_idx = hull.vertices[(i+1) % len(hull.vertices)]
        print('curr_idx:', curr_idx)
        print('next_idx:', next_idx)

        curr_pt = points[curr_idx]
        next_pt = points[next_idx]

        print('curr_pt:', curr_pt)
        print('next_pt:', next_pt)

        segment_lst.append([curr_pt, next_pt])
        distances.append(np.linalg.norm(next_pt - curr_pt))
    total_perimeter = sum(distances)

    # Accumulate the distances
    distance_accum = np.cumsum(distances)

    # Determine how far apart each sample point should be
    step_size = total_perimeter / num_samples

    # Sample points
    sampled_points = []

    points_distance = np.linspace(0, total_perimeter, num_samples, endpoint=False)
    for i in range(0, num_samples):
        current_distance = points_distance[i]
        for seg_idx in range(hull.nsimplex):
            if current_distance < distance_accum[seg_idx]:
                break
        segment = segment_lst[seg_idx]
        if seg_idx == 0:
            local_distance = current_distance
        else:
            local_distance = current_distance - distance_accum[seg_idx-1]
        direction = (segment[1] - segment[0]) / distances[seg_idx]
        point = segment[0] + direction * local_distance
        sampled_points.append(point)
    
    return {
        'hull': hull, 
        'sampled_points': np.array(sampled_points)
    }


def get_discrete_move_directions(sample_action_mode: str) -> np.ndarray:
    """
    Returns an array of discrete movement direction vectors based on the specified sampling mode.

    Args:
        sample_action_mode (str): Specifies the set of movement directions to generate.
            Supported values:
            - '3D-6directions': 6 axis-aligned unit vectors along x, y, and z.
            - '3D-14directions': 6 axis-aligned + 8 diagonal vectors pointing to the cube corners.
            - '2D-8directions': 8 directions in the x-z plane forming a unit circle.

    Returns:
        np.ndarray: An (N, 3) array of movement vectors where N depends on the sampling mode.

    Raises:
        ValueError: If the sample_action_mode is not one of the supported modes.
    """
    if sample_action_mode == '3D-6directions':
        move_vec_ary = np.array([[0, 0, 1], [0, 0, -1], 
                                 [0, 1, 0], [0, -1, 0], 
                                 [1, 0, 0], [-1, 0, 0]])
    elif sample_action_mode == '3D-14directions':
        move_vec_ary = np.array([[0, 0, 1], [0, 0, -1], 
                                 [0, 1, 0], [0, -1, 0], 
                                 [1, 0, 0], [-1, 0, 0]])
        diagonal_dirs = np.array([[ 1, 1, 1], [ 1, 1, -1], [ 1, -1, 1], [ 1, -1, -1],
                                  [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]]) / np.sqrt(3)
        move_vec_ary = np.concatenate([move_vec_ary, diagonal_dirs], axis=0)
    elif sample_action_mode == '2D-8directions':
        angles_lst = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, 
                      np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]
        move_vec_ary = np.array([[0, np.cos(angle), np.sin(angle)] for angle in angles_lst])
    else:
        raise ValueError(f"Unsupported sample_action_mode: {sample_action_mode}")
    
    return move_vec_ary

def pca_points(points):
    """
    Run PCA anaysis on a set of points.
    
    Args:
        points (np.ndarray): shape (N, 3), the points to perform PCA on.
    
    Returns:
        tuple: A tuple containing the centroid, eigenvalues, and eigenvectors.
    """
    centroid = points.mean(axis=0)
    covariance_matrix = np.cov(points - centroid, rowvar=False)

    # Perform PCA
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

    # Sort the eigenvalues: eigen_vectors[:, i] is the eigenvector corresponding to the eigen_values[i]
    sorted_indices = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sorted_indices]
    eigen_vectors = eigen_vectors[:, sorted_indices]
    return centroid, eigen_values, eigen_vectors

def distance_point_to_line(point, line_start, line_end, ret_shortest_vec=False):
    """
    Compute the distance from a point to a line segment defined by two endpoints.
    
    Args:
        point (np.ndarray): shape (3,), the point to compute the distance from.
        line_start (np.ndarray): shape (3,), the start point of the line segment.
        line_end (np.ndarray): shape (3,), the end point of the line segment.
        ret_shortest_vec (bool): Whether to return the shortest vector from the point to the line.
    
    Returns:
        float: The distance from the point to the line segment.
    """
    # Convert to numpy arrays
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    
    # Direction vector of the line
    line_vec = line_end - line_start
    
    # Vector from line_start to the point
    point_vec = point - line_start
    
    # Project point_vec onto line_vec
    line_vec_normalized = line_vec / np.linalg.norm(line_vec)
    projection_length = np.dot(point_vec, line_vec_normalized)
    projection_vec = projection_length * line_vec_normalized
    
    # Calculate the vector from the point to the line
    shortest_vec = point_vec - projection_vec
    
    # Distance is the norm of this vector
    distance = np.linalg.norm(shortest_vec)
    if ret_shortest_vec:
        return distance, shortest_vec
    else:
        return distance

def average_distance_points_to_line(points, line_start, line_end):
    distances = [distance_point_to_line(point, line_start, line_end) for point in points]
    return np.mean(distances)

def compute_crop_space(img_points, colors, xyxy, crop_scale, cam2rob_trans, all_mask, color_intr, depth_max=2.0):
    """
    Crop the space point cloud based on the 2D bounding box

    Args:
    - img_points: (np.ndarray), shape [H, W, 3], the 3D points in the camera frame
    - colors: (np.ndarray), shape [H, W, 3], the colors of the points
    - xyxy: (list), the bounding box in x_min, y_min, x_max, y_max format
    - crop_scale: (float), the scale factor to crop the bounding
    - cam2rob_trans: (np.ndarray), shape [4, 4], the transformation matrix from camera to robot frame
    - neg_mask: (np.ndarray), shape [H, W], the mask to filter out the points

    Returns:
    - crop_space_pcd: (open3d.geometry.PointCloud), the cropped point cloud in the robot frame
    """
    # Down size bounding box from xyxy with same center
    x_min, y_min, x_max, y_max = xyxy
    center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    new_x_min, new_y_min = center - (center - np.array([x_min, y_min])) * crop_scale
    new_x_max, new_y_max = center + (np.array([x_max, y_max]) - center) * crop_scale

    new_xyxy = [int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)]
    print ('new xyxy:', new_xyxy)

    # # Crop corresponding points based on artificial points
    # crop_points = []
    # crop_colors = []
    # for x in range(int(new_xyxy[1]), int(new_xyxy[3])):
    #     for y in range(int(new_xyxy[0]), int(new_xyxy[2])):
    #         # if in mask_all
    #         if all_mask[x, y]:
    #             crop_points.append(img_points[x, y, :])
    #             crop_colors.append(colors[x, y, :])
    #         elif np.any(img_points[x, y] != 0):
    #             crop_points.append(img_points[x, y, :])
    #             crop_colors.append(colors[x, y, :])
    #         else:
    #             pt = rs.rs2_deproject_pixel_to_point(color_intr, [y,x], depth_max)
    #             # print ('pt:', pt)
    #             crop_points.append(pt)
    #             crop_colors.append([0, 0, 0])
    # crop_points = np.array(crop_points)
    # crop_colors = np.array(crop_colors)

    # Crop corresponding points just based on xyxy
    crop_points = img_points[int(new_xyxy[1]):int(new_xyxy[3]), int(new_xyxy[0]):int(new_xyxy[2]), :]
    crop_colors = colors[int(new_xyxy[1]):int(new_xyxy[3]), int(new_xyxy[0]):int(new_xyxy[2]), :]
    
    crop_space_pcd = o3d.geometry.PointCloud()
    crop_space_pcd.points = o3d.utility.Vector3dVector(crop_points.reshape(-1, 3))
    crop_space_pcd.colors = o3d.utility.Vector3dVector(crop_colors.reshape(-1, 3) / 255.0)
    crop_space_pcd.transform(cam2rob_trans)

    cl, ind = crop_space_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
    display_inlier_outlier(crop_space_pcd, ind)
    crop_space_pcd = crop_space_pcd.select_by_index(ind)

    # # Crop corresponding points
    # crop_points = img_points[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
    # crop_colors = colors[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
    # if neg_mask is not None:
    #     # Filter out the points based on the mask
    #     crop_points_temp = []
    #     crop_colors_temp = []
    #     for i in range(neg_mask.shape[0]):
    #         for j in range(neg_mask.shape[1]):
    #             if neg_mask[i, j] == False and i >= xyxy[1] and i < xyxy[3] and j >= xyxy[0] and j < xyxy[2]:
    #                 crop_points_temp.append(img_points[i, j])
    #                 crop_colors_temp.append(colors[i, j])
    #     crop_points = np.array(crop_points_temp)
    #     crop_colors = np.array(crop_colors_temp)
    # crop_space_pcd = o3d.geometry.PointCloud()
    # crop_space_pcd.points = o3d.utility.Vector3dVector(crop_points.reshape(-1, 3))
    # crop_space_pcd.colors = o3d.utility.Vector3dVector(crop_colors.reshape(-1, 3) / 255.0)
    # crop_space_pcd.transform(cam2rob_trans)
    
    return crop_space_pcd

# def is_free_frontier_point(point, octomap_scene, voxel_size, layer = 1):
#     key = octomap_scene.coordToKey(np.array([point[0], point[1], point[2]]))
#     node = octomap_scene.search(key)
#     try:
#         node.getOccupancy()
#     except octomap.NullPointerException:
#         return False
#     if node.getOccupancy() < 0.4 + 1e-6:
#         if layer == 0:
#             return True
#         # check if the point is in the free space frontier by checking the layer of the point is not free
#         for x in range(-layer, layer):
#             for y in range(-layer, layer):
#                 for z in range(-layer, layer):
#                     pts = point + np.array([x*voxel_size, y*voxel_size, z*voxel_size])
#                     key = octomap_scene.coordToKey(np.array([pts[0], pts[1], pts[2]]))
#                     node = octomap_scene.search(key)
#                     try:
#                         node.getOccupancy()
#                     except octomap.NullPointerException:
#                         return False
#                     if node.getOccupancy() < 0.4 + 1e-6:
#                         continue
#                     return True
#     return False

def get_volume_try_fix_by_trimesh(vertices, faces):
    fixed = True
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    if not trimesh_mesh.is_volume:
        trimesh_mesh.remove_unreferenced_vertices()
        trimesh_mesh.remove_degenerate_faces()
        trimesh_mesh.remove_duplicate_faces()
        trimesh_mesh.remove_infinite_values()
        is_watertight = trimesh_mesh.fill_holes()
        if not is_watertight:
            print('The mesh is not watertight after filling holes')
            fixed = False
        if not trimesh_mesh.is_winding_consistent:
            trimesh_mesh.fix_normals()
            if not trimesh_mesh.is_volume:
                print('The mesh is not watertight after fixing normals')
            fixed = False
    volume = trimesh_mesh.volume
    return volume, fixed

def sdf_negetive_inside_trimesh(vertices, faces, points):
    if len(points) == 0:
        return np.array([])
    strat_time = time.time()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    sdf = trimesh.proximity.signed_distance(mesh, points) # signed distance function that is positive inside the mesh
    sdf = -sdf
    print('Time to compute SDF:', time.time() - strat_time)
    return sdf
