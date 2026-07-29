"""Resolution of the robot/gripper model assets.

These now live in the top-level ``assets/robot/`` directory (a repo sibling of the code
package, env-overridable via ``$SIMPACT_ASSETS_DIR``) alongside the camera-calibration
registry — code and rig data are kept separate. Calibration resolves via
``simpact.utils.config.get_calibration_dir`` / ``simpact.real2sim.camera_calibration``.
"""
from simpact.utils.config import get_assets_dir as _get_assets_root


def get_assets_dir():
    """Directory of MuJoCo robot assets (franka_mujoco gripper model)."""
    return _get_assets_root() / "robot"
