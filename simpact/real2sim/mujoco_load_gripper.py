import mujoco
import mujoco.viewer
import numpy as np
import time

class FloatingGripperController:
    def __init__(self, xml_path):
        """
        Initialize the floating gripper controller.
        
        Args:
            xml_path: Path to the MuJoCo XML file
        """
        # Load model and create data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Get mocap body ID + the welded hand body (whose freejoint we teleport)
        self.mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper_mocap')
        self.hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
        
        # Find the mocap index (not the same as body ID!)
        # Mocap bodies have a separate indexing system
        self.mocap_id = -1
        for i in range(self.model.nmocap):
            if self.model.body_mocapid[self.mocap_body_id] == i:
                self.mocap_id = i
                break
        
        if self.mocap_id == -1:
            # If the above doesn't work, try this simpler approach
            # Mocap bodies are indexed separately from regular bodies
            self.mocap_id = 0  # Usually the first (and only) mocap body
        
        # Get actuator IDs for fingers
        self.left_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_finger')
        self.right_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_finger')
        
        print(f"Model loaded successfully!")
        print(f"Number of mocap bodies: {self.model.nmocap}")
        print(f"Mocap body ID: {self.mocap_body_id}")
        print(f"Mocap index: {self.mocap_id}")
        print(f"Left finger actuator ID: {self.left_finger_id}")
        print(f"Right finger actuator ID: {self.right_finger_id}")
        
    def set_gripper_pose(self, position, orientation=None):
        """
        Set the gripper position and orientation.
        
        Args:
            position: [x, y, z] position in meters
            orientation: quaternion [w, x, y, z] or None to keep current
        """
        self.data.mocap_pos[self.mocap_id] = position
        
        if orientation is not None:
            self.data.mocap_quat[self.mocap_id] = orientation
    
    def snap_to_mocap(self):
        """Teleport the welded ``hand`` body to the current mocap target.

        ``set_gripper_pose`` only moves the mocap target; the ``hand`` (a freejoint
        body welded to ``gripper_mocap``) is otherwise dragged there by the weld
        over the first sim steps, sweeping across the scene from its spawn pose and
        knocking nearby objects over. Writing the hand's freejoint qpos to the mocap
        pose (the weld's relative transform is identity — both are defined at the
        same pose in the XML) makes the gripper start *at* its pose, so the weld is
        already satisfied and there is no fly-in. Call this right after
        ``set_gripper_pose`` and before the first ``mj_forward``/step.
        """
        if self.hand_body_id < 0:
            return
        jadr = self.model.body_jntadr[self.hand_body_id]
        if jadr < 0 or self.model.jnt_type[jadr] != mujoco.mjtJoint.mjJNT_FREE:
            return  # hand isn't free-jointed as expected; leave it alone
        qadr = self.model.jnt_qposadr[jadr]
        self.data.qpos[qadr:qadr + 3] = self.data.mocap_pos[self.mocap_id]
        self.data.qpos[qadr + 3:qadr + 7] = self.data.mocap_quat[self.mocap_id]
        vadr = self.model.jnt_dofadr[jadr]
        self.data.qvel[vadr:vadr + 6] = 0.0  # no residual velocity from the teleport

    def set_gripper_width(self, width):
        """
        Set the gripper opening width.
        
        Args:
            width: Opening width in meters (0 = closed, 0.04 = fully open)
        """
        width = np.clip(width, 0.0, 0.04)
        self.data.ctrl[self.left_finger_id] = width
        self.data.ctrl[self.right_finger_id] = width
    
    def open_gripper(self):
        """Fully open the gripper."""
        self.set_gripper_width(0.04)
    
    def close_gripper(self):
        """Fully close the gripper."""
        self.set_gripper_width(0.0)
    
    def get_gripper_pose(self):
        """
        Get current gripper pose.
        
        Returns:
            position: [x, y, z]
            orientation: quaternion [w, x, y, z]
        """
        position = self.data.mocap_pos[self.mocap_id].copy()
        orientation = self.data.mocap_quat[self.mocap_id].copy()
        return position, orientation
    
    def get_gripper_width(self):
        """Get current gripper width."""
        return self.data.ctrl[self.left_finger_id]


class KeyboardController:
    """Handles keyboard input for interactive control."""
    
    def __init__(self):
        self.keys_pressed = set()
        self.translation_speed = 0.005  # meters per step
        self.rotation_speed = 0.05  # radians per step
        self.gripper_speed = 0.001  # meters per step
        
    def key_callback(self, keycode):
        """Store pressed keys."""
        self.keys_pressed.add(keycode)
    
    def get_control_commands(self):
        """
        Convert keyboard input to control commands.
        
        Returns:
            translation: [dx, dy, dz]
            rotation: [roll, pitch, yaw]
            gripper_delta: change in gripper width
        """
        translation = np.zeros(3)
        rotation = np.zeros(3)
        gripper_delta = 0.0
        
        # Translation controls using arrow keys
        # Up/Down arrows: forward/backward (X axis)
        if 265 in self.keys_pressed:  # Up arrow
            translation[0] += self.translation_speed
        if 264 in self.keys_pressed:  # Down arrow
            translation[0] -= self.translation_speed
        
        # Left/Right arrows: left/right (Y axis)
        if 263 in self.keys_pressed:  # Left arrow
            translation[1] += self.translation_speed
        if 262 in self.keys_pressed:  # Right arrow
            translation[1] -= self.translation_speed
        
        # Q/E: up/down (Z axis)
        if ord('Q') in self.keys_pressed or ord('q') in self.keys_pressed:
            translation[2] += self.translation_speed
        if ord('E') in self.keys_pressed or ord('e') in self.keys_pressed:
            translation[2] -= self.translation_speed
        
        # Rotation controls (using arrow keys and other keys)
        # I/K: pitch up/down
        if ord('I') in self.keys_pressed or ord('i') in self.keys_pressed:
            rotation[1] += self.rotation_speed  # pitch up
        if ord('K') in self.keys_pressed or ord('k') in self.keys_pressed:
            rotation[1] -= self.rotation_speed  # pitch down
        
        # J/L: yaw left/right
        if ord('J') in self.keys_pressed or ord('j') in self.keys_pressed:
            rotation[2] += self.rotation_speed  # yaw left
        if ord('L') in self.keys_pressed or ord('l') in self.keys_pressed:
            rotation[2] -= self.rotation_speed  # yaw right
        
        # U/O: roll left/right
        if ord('U') in self.keys_pressed or ord('u') in self.keys_pressed:
            rotation[0] += self.rotation_speed  # roll left
        if ord('O') in self.keys_pressed or ord('o') in self.keys_pressed:
            rotation[0] -= self.rotation_speed  # roll right
        
        # Gripper controls
        # Z: open gripper
        if ord('Z') in self.keys_pressed or ord('z') in self.keys_pressed:
            gripper_delta += self.gripper_speed
        
        # X: close gripper
        if ord('X') in self.keys_pressed or ord('x') in self.keys_pressed:
            gripper_delta -= self.gripper_speed
        
        # Clear pressed keys
        self.keys_pressed.clear()
        
        return translation, rotation, gripper_delta


def quaternion_multiply(q1, q2):
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion [w, x, y, z]."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def print_controls():
    """Print control instructions."""
    print("\n" + "="*60)
    print("INTERACTIVE GRIPPER CONTROL")
    print("="*60)
    print("\nTRANSLATION CONTROLS:")
    print("  ↑/↓ - Move Forward/Backward (X axis)")
    print("  ←/→ - Move Left/Right (Y axis)")
    print("  Q/E - Move Up/Down (Z axis)")
    print("\nROTATION CONTROLS:")
    print("  I/K - Pitch Up/Down")
    print("  J/L - Yaw Left/Right")
    print("  U/O - Roll Left/Right")
    print("\nGRIPPER CONTROLS:")
    print("  Z - Open Gripper")
    print("  X - Close Gripper")
    print("\nOTHER:")
    print("  R - Reset to initial pose")
    print("  ESC - Exit")
    print("="*60 + "\n")


def interactive_control(controller):
    """
    Run interactive control mode with keyboard input.
    
    Args:
        controller: FloatingGripperController instance
    """
    # Initialize gripper pose
    initial_position = np.array([0.3, 0.0, 0.4])
    initial_orientation = np.array([0.0, 1.0, 0.0, 0.0])  # w, x, y, z
    
    controller.set_gripper_pose(initial_position, initial_orientation)
    controller.open_gripper()
    
    # Forward step to initialize
    mujoco.mj_forward(controller.model, controller.data)
    
    # Create keyboard controller
    kb_controller = KeyboardController()
    
    # Print instructions
    print_controls()
    
    # Launch viewer with keyboard callback
    with mujoco.viewer.launch_passive(
        controller.model, 
        controller.data,
        key_callback=kb_controller.key_callback
    ) as viewer:
        
        while viewer.is_running():
            # Get control commands from keyboard
            translation, rotation, gripper_delta = kb_controller.get_control_commands()
            
            # Check for reset command
            if ord('R') in kb_controller.keys_pressed or ord('r') in kb_controller.keys_pressed:
                controller.set_gripper_pose(initial_position, initial_orientation)
                controller.open_gripper()
                kb_controller.keys_pressed.discard(ord('R'))
                kb_controller.keys_pressed.discard(ord('r'))
                print("Reset to initial pose")
            
            # Get current pose
            position, orientation = controller.get_gripper_pose()
            gripper_width = controller.get_gripper_width()
            
            # Apply translation
            new_position = position + translation
            
            # Apply rotation
            if np.any(rotation != 0):
                # Convert rotation delta to quaternion
                delta_quat = euler_to_quaternion(rotation[0], rotation[1], rotation[2])
                # Multiply with current orientation
                new_orientation = quaternion_multiply(delta_quat, orientation)
                # Normalize
                new_orientation = new_orientation / np.linalg.norm(new_orientation)
            else:
                new_orientation = orientation
            
            # Apply gripper control
            new_gripper_width = gripper_width + gripper_delta
            
            # Update gripper
            controller.set_gripper_pose(new_position, new_orientation)
            controller.set_gripper_width(new_gripper_width)
            
            # Step simulation
            mujoco.mj_step(controller.model, controller.data)
            viewer.sync()
            
            # Display current state
            if np.any(translation != 0) or np.any(rotation != 0) or gripper_delta != 0:
                print(f"Position: [{new_position[0]:.3f}, {new_position[1]:.3f}, {new_position[2]:.3f}] | "
                      f"Gripper: {new_gripper_width:.4f}m", end='\r')
            
            time.sleep(0.01)


def main():
    # Path to your XML file
    xml_path = "scene.xml"  # Update this path
    
    try:
        # Create controller
        controller = FloatingGripperController(xml_path)
        
        # Run interactive control
        interactive_control(controller)
        
        print("\nExiting...")
        
    except FileNotFoundError:
        print(f"Error: Could not find XML file at '{xml_path}'")
        print("Please update the xml_path variable with the correct path to your XML file.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()