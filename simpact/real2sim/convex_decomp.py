"""
Convex Decomposition for MuJoCo
This script decomposes a concave mesh into convex parts
and generates a MuJoCo XML file to use them.
"""

import numpy as np
import trimesh
import os
import subprocess
import tempfile
import glob

def decompose_mesh_coacd(input_mesh_path, output_dir, threshold=0.05, max_convex_hull=32, 
                        resolution=2000, mcts_iterations=150, pca=False):
    """
    Decompose mesh using CoACD (Approximate Convex Decomposition)
    
    Args:
        input_mesh_path: Path to input OBJ file
        output_dir: Directory to save decomposed parts
        threshold: Concavity threshold in [0.01, 1] (0.01: most fine-grained; 1: most coarse)
        max_convex_hull: Maximum number of convex hulls (-1 for no limit)
        resolution: Surface sampling resolution for Hausdorff distance computation
        mcts_iterations: Number of MCTS iterations
        pca: Use PCA to align input mesh (suitable for non-axis-aligned mesh)
    """
    try:
        # NOTE: open3d must be imported before coacd in this env — importing
        # coacd first segfaults (native library clash). Do not reorder.
        import open3d  # noqa: F401
        import coacd

        # Load mesh with trimesh
        mesh = trimesh.load(input_mesh_path, force="mesh")
        
        # Create CoACD mesh object (this is the key difference!)
        coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        
        # Run CoACD decomposition with proper API
        result = coacd.run_coacd(
            coacd_mesh,  # Pass the coacd.Mesh object, not raw arrays
            threshold=threshold,
            max_convex_hull=max_convex_hull,
            preprocess_mode="auto",
            preprocess_resolution=50,
            resolution=resolution,
            mcts_nodes=20,
            mcts_iterations=mcts_iterations,
            mcts_max_depth=3,
            pca=pca,
            merge=True,  # Try to reduce total number of parts by merging
            seed=0
        )
        
        # Save each part
        os.makedirs(output_dir, exist_ok=True)
        part_paths = []
        
        print(f"\nDecomposition successful! Generated {len(result)} convex parts.")
        
        for i, (vertices, faces) in enumerate(result):
            part_mesh = trimesh.Trimesh(vertices, faces)
            part_path = os.path.join(output_dir, f"part_{i:03d}.obj")
            part_mesh.export(part_path)
            part_paths.append(part_path)
            print(f"  Part {i+1}/{len(result)}: {len(vertices)} vertices, {len(faces)} faces")
        
        return part_paths
        
    except ImportError:
        print("CoACD not installed. Install with: pip install coacd")
        return None
    except Exception as e:
        print(f"CoACD decomposition failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    

def preserve_textures(input_mesh_path, part_paths, output_dir):
    """
    Preserve and apply original texture to decomposed parts
    
    Args:
        input_mesh_path: Path to original mesh file
        part_paths: List of paths to decomposed mesh parts
        output_dir: Directory containing the parts
    
    Returns:
        Path to texture file if found, None otherwise
    """
    import trimesh
    import shutil
    
    print("\nChecking for textures...")
    
    # Load original mesh
    original_mesh = trimesh.load(input_mesh_path, process=False)
    
    # Check if mesh has texture
    texture_path = None
    material = None
    
    if hasattr(original_mesh, 'visual') and hasattr(original_mesh.visual, 'material'):
        material = original_mesh.visual.material
        
        # Check for texture image
        if hasattr(material, 'image') and material.image is not None:
            print("  ✓ Found texture in mesh")
            
            # Save texture image
            texture_filename = "texture.png"
            texture_path = os.path.join(output_dir, texture_filename)
            material.image.save(texture_path)
            print(f"  ✓ Saved texture to: {texture_filename}")
            
        elif hasattr(material, 'baseColorTexture'):
            print("  ✓ Found texture reference")
            # Try to find texture file in same directory as mesh
            mesh_dir = os.path.dirname(input_mesh_path)
            
            # Common texture file patterns
            base_name = os.path.splitext(os.path.basename(input_mesh_path))[0]
            texture_patterns = [
                f"{base_name}_0.png",
            ]
            
            for pattern in texture_patterns:
                potential_path = os.path.join(mesh_dir, pattern)
                if os.path.exists(potential_path):
                    texture_filename = os.path.basename(potential_path)
                    texture_path = os.path.join(output_dir, texture_filename)
                    shutil.copy(potential_path, texture_path)
                    print(f"  ✓ Copied texture: {texture_filename}")
                    break
    
    # Also check for MTL file (OBJ format)
    if input_mesh_path.lower().endswith('.obj'):
        mtl_path = input_mesh_path.replace('.obj', '.mtl')
        if os.path.exists(mtl_path):
            print(f"  ✓ Found material file: {os.path.basename(mtl_path)}")
            
            # Copy MTL file
            shutil.copy(mtl_path, os.path.join(output_dir, os.path.basename(mtl_path)))
            
            # Parse MTL to find texture
            with open(mtl_path, 'r') as f:
                for line in f:
                    if line.strip().startswith('map_Kd'):
                        tex_file = line.split()[1]
                        tex_src = os.path.join(os.path.dirname(input_mesh_path), tex_file)
                        if os.path.exists(tex_src):
                            texture_path = os.path.join(output_dir, os.path.basename(tex_file))
                            shutil.copy(tex_src, texture_path)
                            print(f"  ✓ Copied texture from MTL: {os.path.basename(tex_file)}")
                            break
    
    if texture_path:
        print(f"\n✓ Texture preserved and ready to use")
        return os.path.basename(texture_path)
    else:
        print("  ℹ No texture found in mesh file")
        print("  You can manually add texture files to the output directory")
        return None


def apply_texture_to_parts(part_paths, texture_filename, output_dir):
    """
    Create OBJ files with texture coordinates and MTL references
    
    Args:
        part_paths: List of paths to mesh part files
        texture_filename: Name of texture file
        output_dir: Directory containing parts and texture
    
    Returns:
        Updated list of part paths with texture support
    """
    import trimesh
    
    print(f"\nApplying texture to {len(part_paths)} parts...")
    
    # Create a shared MTL file
    mtl_filename = "material.mtl"
    mtl_path = os.path.join(output_dir, mtl_filename)
    
    with open(mtl_path, 'w') as f:
        f.write("newmtl material\n")
        f.write("Ka 1.0 1.0 1.0\n")  # Ambient
        f.write("Kd 1.0 1.0 1.0\n")  # Diffuse
        f.write("Ks 0.5 0.5 0.5\n")  # Specular
        f.write("Ns 32.0\n")          # Shininess
        f.write(f"map_Kd {texture_filename}\n")  # Texture map
    
    print(f"  ✓ Created material file: {mtl_filename}")
    
    # Update each OBJ file to reference the MTL
    textured_paths = []
    for part_path in part_paths:
        # Read OBJ file
        with open(part_path, 'r') as f:
            lines = f.readlines()
        
        # Add MTL reference at the top if not present
        has_mtl = any(line.startswith('mtllib') for line in lines)
        has_usemtl = any(line.startswith('usemtl') for line in lines)
        
        if not has_mtl:
            # Insert MTL reference after comments
            insert_idx = 0
            for i, line in enumerate(lines):
                if not line.startswith('#'):
                    insert_idx = i
                    break
            
            lines.insert(insert_idx, f"mtllib {mtl_filename}\n")
            lines.insert(insert_idx + 1, "usemtl material\n")
            
            # Write updated OBJ
            with open(part_path, 'w') as f:
                f.writelines(lines)
        
        textured_paths.append(part_path)
    
    print(f"  ✓ Applied material to all parts")
    return textured_paths


def generate_mujoco_xml(prefix, part_paths, xml_output_path, mesh_dir_relative="meshes", texture_filename=None):
    """
    Generate MuJoCo XML file with decomposed mesh parts
    
    Args:
        part_paths: List of paths to mesh part files
        xml_output_path: Path to save XML file
        mesh_dir_relative: Relative path to mesh directory from XML file
    """
    xml_content = f"""<?xml version="1.0" ?>
<mujoco model="decomposed_object">
    <compiler angle="radian"/>
    
    <asset>
"""

    # Add texture if available
    if texture_filename:
        xml_content += f'        <texture name="{prefix}_decomposed_texture" type="2d" file="{mesh_dir_relative}/{texture_filename}"/>\n'
        xml_content += f'        <material name="{prefix}_decomposed_material" texture="{prefix}_decomposed_texture" rgba="1 1 1 1"/>\n\n'

    
    # Add mesh assets
    for i, part_path in enumerate(part_paths):
        mesh_name = os.path.basename(part_path)
        xml_content += f'        <mesh name="{prefix}_part_{i}" file="{mesh_dir_relative}/{mesh_name}"/>\n'
    
    xml_content += f"""    </asset>
    
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
        
        <body name="{prefix}_decomposed" pos="0.5019 0.1040 0.1895" quat="0.8638 0.5019 -0.0389 -0.0222">
"""
    
    # Add geometry for each part
    for i in range(len(part_paths)):
        if texture_filename:
            xml_content += f'            <geom type="mesh" friction="1.0 0.005 0.0001" mesh="{prefix}_part_{i}" material="{prefix}_decomposed_material"/>\n'
        else:
            xml_content += f'            <geom type="mesh" friction="1.0 0.005 0.0001" mesh="{prefix}_part_{i}" rgba="0.8 0.6 0.4 1"/>\n'
    
    xml_content += """        </body>
        
        <body name="sphere" pos="0 0 0.5">
            <joint type="free"/>
            <geom type="sphere" size="0.03" rgba="1 0 0 1" mass="0.1"/>
        </body>
    </worldbody>
</mujoco>
"""
    
    # Save XML file
    with open(xml_output_path, 'w') as f:
        f.write(xml_content)
    
    print(f"\nMuJoCo XML saved to: {xml_output_path}")


def load_and_simulate(xml_path):
    """
    Load the MuJoCo model and run a simple simulation
    """
    try:
        import mujoco
        import mujoco.viewer
        
        # Load model
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        
        print("\nLaunching MuJoCo viewer...")
        
        # Launch viewer
        mujoco.viewer.launch(model, data)
        
    except ImportError:
        print("\nMuJoCo Python bindings not installed.")
        print("Install with: pip install mujoco")
        print(f"\nYou can still load the XML file manually: {xml_path}")


def main():
    """
    Main function to decompose mesh and create MuJoCo scene
    """
    # Configuration
    OBJ_NAME = "blue bowl"
    INPUT_MESH = f"data/1024/{OBJ_NAME}_scaled.obj"
    OUTPUT_DIR = f"data/1024/{OBJ_NAME}_decomposed"
    XML_OUTPUT = "scene_blue_bowl.xml"
    
    # Check if input file exists
    if not os.path.exists(INPUT_MESH):
        print(f"\nError: Input mesh '{INPUT_MESH}' not found!")
        print("Please update INPUT_MESH variable with your mesh path.")
        return
    
    # Try decomposition methods in order of preference
    print(f"\nInput mesh: {INPUT_MESH}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    part_paths = None
    
    # # Try CoACD first (usually best results)
    # print("Attempting CoACD decomposition...")
    # print("Parameters: threshold=0.05 (lower=more parts), max_convex_hull=32")
    # part_paths = decompose_mesh_coacd(
    #     INPUT_MESH, 
    #     OUTPUT_DIR, 
    #     threshold=0.03,  # 0.01 for fine-grained, 1.0 for coarse
    #     max_convex_hull=64,  # -1 for no limit
    #     resolution=2000,
    #     mcts_iterations=150
    # )
    
    # if not part_paths:
    #     print("\nAll decomposition methods failed!")
    #     return
    
    # print(f"\nDecomposed into {len(part_paths)} convex parts")
    
    
    # Preserve textures from original mesh
    part_paths = sorted(glob.glob(f"{OUTPUT_DIR}/part_*.obj"))
    texture_filename = preserve_textures(INPUT_MESH, part_paths, OUTPUT_DIR)
    
    # if texture_filename:
    #     apply_texture = input("\nApply texture to decomposed parts? (y/n) [default: y]: ").lower()
    #     if apply_texture != 'n':
    #         part_paths = apply_texture_to_parts(part_paths, texture_filename, OUTPUT_DIR)
    
    print(texture_filename)
    # Generate MuJoCo XML
    generate_mujoco_xml(OBJ_NAME, part_paths, XML_OUTPUT, mesh_dir_relative=OUTPUT_DIR, texture_filename=texture_filename)
    
    # # Try to launch viewer
    # print("\n" + "=" * 60)
    # response = input("Would you like to launch the MuJoCo viewer? (y/n): ")
    # if response.lower() == 'y':
    #     load_and_simulate(XML_OUTPUT)
    # else:
    #     print(f"\nYou can load the scene later with:")
    #     print(f"  python -m mujoco.viewer --mjcf={XML_OUTPUT}")


if __name__ == "__main__":
    main()