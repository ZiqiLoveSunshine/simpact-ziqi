
import numpy as np
try:
    import warp as wp
except ImportError as e:
    raise ImportError(
        "warp is required for MPM simulation. Install it with: pip install warp-lang"
    ) from e

def extract_transformation(box_collider_params):
    # Method 1: Access matrix elements directly
    transform_matrix_4x4 = wp.transform_to_matrix(box_collider_params.xform)
    # Convert to numpy by accessing elements
    transform_matrix_np = np.array([
        [transform_matrix_4x4[0, 0], transform_matrix_4x4[0, 1], transform_matrix_4x4[0, 2], transform_matrix_4x4[0, 3]],
        [transform_matrix_4x4[1, 0], transform_matrix_4x4[1, 1], transform_matrix_4x4[1, 2], transform_matrix_4x4[1, 3]],
        [transform_matrix_4x4[2, 0], transform_matrix_4x4[2, 1], transform_matrix_4x4[2, 2], transform_matrix_4x4[2, 3]],
        [transform_matrix_4x4[3, 0], transform_matrix_4x4[3, 1], transform_matrix_4x4[3, 2], transform_matrix_4x4[3, 3]]
    ])
    return transform_matrix_np

def set_collision_params(collider_param, center, quat, twist):
    collider_param.xform = wp.transform(
        [center[0], center[1], center[2]], [quat[0], quat[1], quat[2], quat[3]]
    )
    collider_param.twist = wp.spatial_vector(
        [twist[0], twist[1], twist[2], twist[3], twist[4], twist[5]]
    )
