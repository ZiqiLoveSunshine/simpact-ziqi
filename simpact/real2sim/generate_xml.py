import os
import sys
import argparse
import re
import numpy as np
from scipy.spatial.transform import Rotation as R

from simpact.real2sim.paths import get_assets_dir


def sanitize_name(name):
    """Converts a human-readable name to a valid filename format."""
    return name.strip().lower().replace(" ", "_")


def create_mujoco_xml(object_string, cam_id, data_dir, output_xml):
    """
    Generates a MuJoCo XML file with multiple specified objects and tables.

    Args:
        object_string (str): A string describing objects, separated by '.' or ','.
        data_dir (str): The directory where object.obj and object_pose.txt are stored.
        output_xml (str): Path to save the generated XML file.
    """

    # --- 1. Parse the input string to get a list of object names ---
    # Split by periods or commas, and remove any empty strings
    object_names = [
        name.strip() for name in re.split("[.,]", object_string) if name.strip()
    ]
    if not object_names:
        print("Warning: No object names found in the input string.")
        return

    print(f"Found objects: {object_names}")

    asset_snippets = []
    texture_asset_snippets = []  # For textures and materials
    worldbody_snippets = []

    # Variables to determine the bounds of all objects
    min_coords = np.array([np.inf, np.inf, np.inf])
    max_coords = np.array([-np.inf, -np.inf, -np.inf])
    objects_loaded = 0

    # --- 2. Loop through each object name and generate its XML parts ---
    for i, name in enumerate(object_names):
        # os.path.join keeps absolute data_dir intact; the original hardcoded a "./"
        # prefix that silently broke the file checks for absolute paths.
        mesh_path = os.path.join(data_dir, f"{name}_scaled.obj")
        texture_path = os.path.join(data_dir, f"{name}_scaled_0.png")
        pose_path = os.path.join(data_dir, f"{name}_mujoco_cam{cam_id}.txt")

        print(pose_path)

        sanitized = sanitize_name(name)

        # Check if files exist
        if not os.path.exists(mesh_path) or not os.path.exists(pose_path):
            print(f"⚠️  Warning: Skipping '{name}'. Could not find required files.")
            continue

        # Load the 4x4 pose matrix
        pose = np.loadtxt(pose_path)
        if pose.shape != (4, 4):
            print(
                f"⚠️  Warning: Skipping '{name}'. Pose matrix in '{pose_path}' is not 4x4."
            )
            continue

        position = pose[:3, 3]
        rotation_matrix = pose[:3, :3]
        # print(rotation_matrix)
        # exit()

        # Update the min/max coordinates for table calculation
        min_coords = np.minimum(min_coords, position)
        max_coords = np.maximum(max_coords, position)

        # Convert to MuJoCo-compatible quaternion [w, x, y, z]
        quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        # --- Handle Textures and Materials ---
        geom_appearance_attr = 'rgba="0.2 0.6 0.3 1"'  # Default color
        if os.path.exists(texture_path):
            print(f"  -> Found texture for '{name}': {texture_path}")
            # Define texture and material assets
            texture_asset_snippets.append(
                f'    <texture name="{sanitized}_tex" type="2d" file="{texture_path}" />'
            )
            texture_asset_snippets.append(
                f'    <material name="{sanitized}_mat" texture="{sanitized}_tex" shininess="0.3" specular="0.75" />'
            )
            # Set the geom to use the material instead of a plain color
            geom_appearance_attr = f'material="{sanitized}_mat"'

        # --- Generate Asset Snippet ---
        asset_snippets.append(f'  <mesh name="{sanitized}_mesh" file="{mesh_path}" />')

        # Object XML
        worldbody_snippets.append(
            f"""<body name="{sanitized}" pos="{position[0]:.4f} {position[1]:.4f} {position[2]:.4f}" 
                quat="{quat_wxyz[0]:.4f} {quat_wxyz[1]:.4f} {quat_wxyz[2]:.4f} {quat_wxyz[3]:.4f}">
            <joint type="free" name="{sanitized}_joint" damping="0.1"/>
            <geom type="mesh" mass="0.5" friction="0.3 0.005 0.0001" mesh="{sanitized}_mesh" {geom_appearance_attr}/>
            </body>"""
        )

        objects_loaded += 1

    # --- 3. Calculate single table properties if any objects were loaded ---
    table_body_snippet = ""
    origin_snippet = ""  # defined unconditionally: the original only set it when
    # objects_loaded > 0, raising UnboundLocalError otherwise.
    if objects_loaded > 0:
        # Calculate table center based on the average of min/max coordinates
        table_center_x = (min_coords[0] + max_coords[0]) / 2.0
        table_center_y = (min_coords[1] + max_coords[1]) / 2.0

        # Calculate table size based on the span of coordinates, with padding
        padding = 0.15  # 15cm padding around the objects
        table_size_x = (max_coords[0] - min_coords[0]) / 2.0 + padding
        table_size_y = (max_coords[1] - min_coords[1]) / 2.0 + padding

        # Position the table top just below the lowest object
        table_thickness = 0.04
        table_top_z = min_coords[2] - 0.01  # 2cm below lowest object's origin
        table_center_z = table_top_z - (table_thickness / 2.0)

        #     table_body_snippet = f"""
        # <body name="table" pos="{table_center_x:.4f} {table_center_y:.4f} {table_center_z:.4f}">
        #   <geom type="box" size="{table_size_x:.4f} {table_size_y:.4f} {table_thickness / 2.0:.4f}" rgba="0.8 0.6 0.4 1" />
        # </body>"""

        # For simplicity, use a fixed large table size for now
        table_body_snippet = """
        <body name="table" pos="0.5243 -0.0009 0.14">
            <geom type="box" size="1.0 1.0 0.0200" rgba="0.6 0.6 0.6 1" friction="0.3 0.005 0.0001"/>
        </body>"""

        origin_snippet = """
        <body name="robot_base" pos="0 0 0">
            <geom name="base_marker" type="sphere" size="0.02" rgba="1 0 0 1"/>
        </body>"""

    # --- 4. Add in robot gripper for pushing or grasping ---
    # if gripper_mode == 0:
    #     gripper_snippet = """<body name="gripper_body" pos="0.467187 0.09 0.24"
    #             quat="1.0 0.0 0.0 0.0">
    #       <joint name="gripper_x" type="slide" axis="1 0 0" range="-0.5 0.5" damping="10"/>
    #       <joint name="gripper_y" type="slide" axis="0 1 0" range="-0.3 0.3" damping="10"/>
    #       <joint name="gripper_z" type="slide" axis="0 0 1" range="-0.1 0.2" damping="10"/>
    #       <geom name="gripper" type="box" size="0.03 0.1 0.042" rgba="0 1 0 0.3" mass="0.5" friction="1.0 0.005 0.0001" group="0"/>
    #       <site name="gripper_site" pos="0 0 0" size="0.01"/>
    #     </body>"""

    #     actuator_snippet = """<actuator>
    #         <!-- Position servos for the gripper -->
    #         <position name="gripper_x_act" joint="gripper_x" kp="1000" kv="100" forcerange="-50 50"/>
    #         <position name="gripper_y_act" joint="gripper_y" kp="1000" kv="100" forcerange="-50 50"/>
    #         <position name="gripper_z_act" joint="gripper_z" kp="1000" kv="100" forcerange="-50 50"/>
    #     </actuator>"""
    # else:
    #     raise NotImplementedError("Gripper mode 1 (grasper) not implemented yet.")
    gripper_xml = get_assets_dir() / "franka_mujoco" / "franka_gripper.xml"
    gripper_snippet = f"""<include file="{gripper_xml}" />"""

    # --- 5. Assemble the final XML from all the generated parts ---
    assets_block = "\n".join(asset_snippets)
    texture_assets_block = "\n".join(texture_asset_snippets)
    objects_block = "\n".join(worldbody_snippets)

    worldbody_block = (
        table_body_snippet
        + "\n"
        + objects_block
        + "\n"
        # + gripper_snippet
        # + "\n"
        + origin_snippet
    )

    xml_content = f"""
<mujoco model="multi_object_scene">
  <compiler angle="degree" coordinate="local" />
  <option integrator="implicitfast" noslip_iterations="3" timestep="0.002" gravity="0 0 -9.81" cone="pyramidal">
    <flag multiccd="enable" warmstart="enable"/>
  </option>
    
    {gripper_snippet}

  <asset>
    {assets_block}
    
    {texture_assets_block}
  </asset>

  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 4" dir="0 0 -1"/>
    <geom type="plane" size="3 3 0.1" rgba=".9 .9 .9 1"/>
    
    <camera name="top_view" pos="0.365 -0.457 1.494" xyaxes="1.000 0.013 0.000 -0.012 0.943 0.334"/>
    <camera name="side_view" pos="0.228 -0.997 0.404" xyaxes="1.000 -0.031 0.000 0.005 0.178 0.984"/>
    <camera name="front_view" pos="1.593 0.044 0.662" xyaxes="0.009 1.000 -0.000 -0.334 0.003 0.943"/>

    {worldbody_block}
  </worldbody>
  
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
    <map znear="0.01"/>
    <scale framelength="0.1" framewidth="0.005"/>  <!-- Adjust frame size -->
  </visual>

</mujoco>
"""
    # --- 4. Write the final XML to a file ---
    with open(output_xml, "w") as f:
        f.write(xml_content)
    print(f"\n✅ Successfully generated multi-object scene at '{output_xml}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MuJoCo XML scene with an object on a table."
    )

    parser.add_argument("--object_list", default="jar. tissue packet.")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="directory to save snapshot"
    )
    parser.add_argument(
        "--output_xml", type=str, default="scene.xml", help="output XML file path"
    )
    parser.add_argument("--cam_id", type=int, default=0, help="camera id")
    # parser.add_argument(
    #     "--gripper_mode",
    #     type=int,
    #     default=0,
    #     help="enter 0 for box pusher, 1 for box grasper",
    # )
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args()

    create_mujoco_xml(
        object_string=args.object_list,
        cam_id=args.cam_id,
        data_dir=args.data_dir,
        output_xml=args.output_xml,
    )
