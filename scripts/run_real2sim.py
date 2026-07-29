#!/usr/bin/env python3
"""Offline real2sim driver — reproduce the deterministic stages of the original
``real2sim/test.sh`` pipeline over a recorded trial directory.

The full pipeline is:

    1. capture RGB-D            (get_stream.py)      [hardware]
    2. segmentation             (Grounded-SAM-2)     [model]
    3. mask extraction          (mask_extraction)    -- reproduced here
    4. image->3D mesh           (Hunyuan3D)          [model]
    5. scale to real world      (estimate_scale)     [needs stage-4 mesh]
    6. 6-DoF pose               (FoundationPose)     [model]
    7. pose -> robot frame      (transform_6d)       -- reproduced here
    8. build MuJoCo scene       (generate_xml)       -- reproduced here

Stages 1/2/4/6 need cameras or external perception models. This driver runs the
deterministic library stages (3, 7, 8) over a directory that already contains
the recorded outputs of the model stages (``*_gsam2.json``, ``*_scaled.obj``,
``*_6d_cam{id}.txt``) — e.g. one of the the original ``real2sim/data/<trial>`` dirs. It is
the offline, hardware-free way to verify the pipeline wiring end to end.

Example:
    python scripts/run_real2sim.py \\
        --data-dir /path/to/data/0211_obstacle_0 \\
        --objects "orange bottle. brown purple box." \\
        --camera-id 1
"""
import argparse
import re
from pathlib import Path

from simpact.real2sim.generate_xml import create_mujoco_xml
from simpact.real2sim.mask_extraction import extract_masks
from simpact.real2sim.transform_6d import transform_object_pose
from simpact.utils.config import get_data_dir


def parse_objects(object_string):
    """Split a dotted/comma object string into a clean list of names."""
    return [name.strip() for name in re.split("[.,]", object_string) if name.strip()]


def resolve_data_dir(data_dir):
    """Accept an absolute/existing path, else resolve under SIMPACT_DATA_DIR."""
    p = Path(data_dir)
    if p.is_absolute() or p.exists():
        return p
    return get_data_dir() / data_dir


def run(data_dir, objects, camera_id, output_xml=None):
    data_dir = resolve_data_dir(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    object_names = parse_objects(objects)
    object_string = "".join(f"{name}. " for name in object_names).strip()
    print(f"Data dir: {data_dir}")
    print(f"Objects:  {object_names}")

    # --- Stage 3: mask extraction from the recorded Grounded-SAM-2 JSON --------
    json_path = data_dir / f"camera{camera_id}_rgb_gsam2.json"
    if json_path.exists():
        rgb = data_dir / f"camera{camera_id}_rgb.png"
        print(f"\n[3/8] mask extraction <- {json_path.name}")
        extract_masks(
            json_path,
            data_dir / f"camera{camera_id}_mask",
            png=True,
            crop=True,
            image_path=str(rgb) if rgb.exists() else None,
        )
    else:
        print(f"\n[3/8] SKIP mask extraction: {json_path.name} not found")

    # --- Stage 5 (scale) is a model-output stage: consumed, not regenerated ----
    # estimate_scale needs the stage-4 (Hunyuan3D) mesh, which recorded trials do
    # not keep; only the *_scaled.obj it produced is present. We consume that.
    missing_scaled = [
        n for n in object_names if not (data_dir / f"{n}_scaled.obj").exists()
    ]
    if missing_scaled:
        print(f"\n[5/8] NOTE: missing *_scaled.obj for {missing_scaled} "
              "(stage 4/5 are model stages; provide a scaled mesh to build them)")

    # --- Stage 7: lift each 6-DoF pose into the robot base frame ---------------
    print(f"\n[7/8] transform_6d (camera {camera_id} -> robot frame)")
    for name in object_names:
        pose_file = data_dir / f"{name}_6d_cam{camera_id}.txt"
        if not pose_file.exists():
            print(f"  SKIP {name}: {pose_file.name} not found")
            continue
        transform_object_pose(data_dir, name, camera_id)

    # --- Stage 8: assemble the MuJoCo scene ------------------------------------
    if output_xml is None:
        output_xml = data_dir / f"scene_{data_dir.name}.xml"
    Path(output_xml).parent.mkdir(parents=True, exist_ok=True)  # ensure out dir exists
    print(f"\n[8/8] generate_xml -> {output_xml}")
    create_mujoco_xml(
        object_string=object_string,
        cam_id=camera_id,
        data_dir=str(data_dir),
        output_xml=str(output_xml),
    )
    return Path(output_xml)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", required=True, help="recorded trial directory")
    parser.add_argument(
        "--objects", required=True, help='dotted object list, e.g. "jar. box."'
    )
    parser.add_argument("--camera-id", type=int, default=1)
    parser.add_argument("--output-xml", default=None, help="output scene XML path")
    args = parser.parse_args()
    run(args.data_dir, args.objects, args.camera_id, args.output_xml)


if __name__ == "__main__":
    main()
