"""real2sim — reconstruct a MuJoCo scene from real RGB-D observations.

Two tiers:

* **Geometry/assembly** (this package's library modules): mesh processing,
  scale, pose transforms, and MuJoCo XML generation. Depends on the optional
  ``real2sim`` extra (mujoco, coacd, open3d, trimesh, ...).
* **Perception** (``simpact.real2sim.perception``): segmentation, image-to-3D,
  and 6D pose estimation. These wrap large external model repos (Grounded-SAM-2,
  Hunyuan3D-2.1, FoundationPose) via adapters; see that subpackage.

Path helpers are always importable; library functions degrade to ``None`` when
the ``real2sim`` extra is not installed.
"""

from simpact.real2sim.paths import get_assets_dir
from simpact.utils.config import get_calibration_dir  # registry: assets/calibration/<profile>

try:
    from simpact.real2sim.generate_xml import create_mujoco_xml, sanitize_name
    from simpact.real2sim.convex_decomp import decompose_mesh_coacd
    from simpact.real2sim.convert_gripper_pose import (
        ee_pose_from_matrix,
        ee_pose_to_matrix,
    )
    from simpact.real2sim.estimate_scale import (
        create_point_cloud_from_rgbd,
        load_intrinsics_from_file,
    )
    from simpact.real2sim.mask_extraction import extract_masks
    from simpact.real2sim.transform_6d import (
        get_camera_to_robot,
        transform_object_pose,
        transform_to_robot_frame,
    )

    _REAL2SIM_AVAILABLE = True
except ImportError:
    create_mujoco_xml = sanitize_name = None
    decompose_mesh_coacd = None
    ee_pose_from_matrix = ee_pose_to_matrix = None
    create_point_cloud_from_rgbd = load_intrinsics_from_file = None
    extract_masks = None
    get_camera_to_robot = transform_object_pose = transform_to_robot_frame = None
    _REAL2SIM_AVAILABLE = False

__all__ = [
    "get_assets_dir",
    "get_calibration_dir",
    "create_mujoco_xml",
    "sanitize_name",
    "decompose_mesh_coacd",
    "ee_pose_from_matrix",
    "ee_pose_to_matrix",
    "create_point_cloud_from_rgbd",
    "load_intrinsics_from_file",
    "extract_masks",
    "get_camera_to_robot",
    "transform_object_pose",
    "transform_to_robot_frame",
]
