#!/usr/bin/env python3
"""Record the robot's current end-effector pose to a file, for offline scene building.

The EE pose is robot proprioception — it is NOT in the RGB-D and the original pipeline's capture
(get_stream) never saved it. Run this ALONGSIDE your RGB-D snapshot so the recorded
bundle is self-contained: build_scene.py auto-discovers ``initial_ee_pose.txt`` and
uses it for both rope's context.txt and MPM's scene.yaml initial_ee_pose.

    python scripts/save_ee_pose.py --host 172.16.0.2 \
        --out /path/to/bundle/initial_ee_pose.txt

Writes a 4x4 world<-EE homogeneous matrix (np.savetxt). Requires the optional
``franky`` hardware dependency; on a machine without a robot, provide the pose file
by other means and point build_scene.py at it with --ee-pose.
"""
import argparse

from simpact.real2sim.build_scene import save_ee_pose_from_robot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="172.16.0.2", help="robot FCI IP")
    ap.add_argument("--out", required=True, help="output path (initial_ee_pose.txt)")
    args = ap.parse_args()
    path = save_ee_pose_from_robot(args.host, args.out)
    print(f"saved initial EE pose (4x4) -> {path}")


if __name__ == "__main__":
    main()
