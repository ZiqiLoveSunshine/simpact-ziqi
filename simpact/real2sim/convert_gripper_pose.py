import numpy as np
from scipy.spatial.transform import Rotation

    
def ee_pose_from_matrix(T_world_ee, ee_offset = np.array([0.0, 0.0, 0.105])):
    """
    Set gripper pose from 4x4 homogeneous transformation matrix.
    
    Args:
        T_world_ee: 4x4 numpy array, homogeneous transform of end-effector
                    in world frame (from Franka API)
    """
    # Extract position and rotation from homogeneous matrix
    ee_position = T_world_ee[:3, 3]
    R_world_ee = T_world_ee[:3, :3]
    
    # Convert rotation matrix to quaternion (w, x, y, z) - MuJoCo format
    rot = Rotation.from_matrix(R_world_ee)
    quat_xyzw = rot.as_quat()  # scipy returns [x, y, z, w]
    ee_orientation = np.array([quat_xyzw[3], quat_xyzw[0], 
                               quat_xyzw[1], quat_xyzw[2]])  # [w, x, y, z]
    
    # Transform: mocap_pos = ee_pos - R * ee_offset
    mocap_position = ee_position - R_world_ee @ ee_offset

    return mocap_position, ee_orientation
    
    
def ee_pose_to_matrix(mocap_pos, mocap_quat, ee_offset = np.array([0.0, 0.0, 0.105])):
    """
    Get end-effector pose as 4x4 homogeneous matrix.
    
    Returns:
        T_world_ee: 4x4 numpy array
    """
    
    # Convert quaternion to rotation matrix
    quat_xyzw = [mocap_quat[1], mocap_quat[2], 
                 mocap_quat[3], mocap_quat[0]]  # [x, y, z, w]
    R_world_hand = Rotation.from_quat(quat_xyzw).as_matrix()
    
    # Transform: ee_pos = mocap_pos + R * ee_offset
    ee_position = mocap_pos + R_world_hand @ ee_offset
    
    # Construct 4x4 homogeneous matrix
    T_world_ee = np.eye(4)
    T_world_ee[:3, :3] = R_world_hand
    T_world_ee[:3, 3] = ee_position
    
    return T_world_ee


if __name__ == "__main__":
    import mujoco
    import argparse
    import time
    from franky import Affine, Robot, Gripper

    import sys
    from pathlib import Path

    # Add parent directory to path
    parent_dir = Path(__file__).parent.parent
    sys.path.append(str(parent_dir))

    from executor.push_6d import FloatingGripperController

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    parser.add_argument('--xml_path', type=str, default='scene.xml')
    args = parser.parse_args()

    real_robot = Robot(args.host)
    real_gripper = Gripper(args.host)
    
    # Open gripper initially
    speed = 0.02  # [m/s]
    # There are also asynchronous versions of the methods
    success_future = real_gripper.move_async(0.05, speed)

    controller = FloatingGripperController(args.xml_path)

    with mujoco.viewer.launch_passive(
        controller.model, 
        controller.data
    ) as viewer:

        while viewer.is_running():

            state = real_robot.state
            # print("\nPose: ", real_robot.current_pose)
            gripper_hmat = state.O_T_EE.matrix

            pos, quat = ee_pose_from_matrix(gripper_hmat)
            print('Gripper pose: ', pos, quat)

            controller.set_gripper_pose(pos, quat)
            # Set gripper width (divide by 2 for MuJoCo)
            controller.set_gripper_width(real_gripper.width / 2)

            # Get runtime poses
            left_pos, left_quat = controller.get_body_pose('left_finger')
            right_pos, right_quat = controller.get_body_pose('right_finger')

            # print('Left finger: ', left_pos, ", right finger: ", right_pos)

            # Step simulation
            mujoco.mj_step(controller.model, controller.data)
            viewer.sync()

            time.sleep(0.01)