"""SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models (CVPR 2026).

Official implementation — offline real2sim + a closed-loop VLM action optimizer.
Project page: https://simpact-bot.github.io/  ·  arXiv:2512.05955

Top-level package. Submodules (``simulators``, ``real2sim``, ``executor``,
``generator``) are imported lazily so that ``import simpact`` succeeds without
optional hardware/simulation dependencies (warp, pyrealsense2, frankx).
"""

from simpact.utils.config import (
    get_project_root,
    get_data_dir,
    get_outputs_dir,
    get_rollouts_dir,
    get_assets_dir,
    get_calibration_dir,
    get_materials_dir,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "get_project_root",
    "get_data_dir",
    "get_outputs_dir",
    "get_rollouts_dir",
    "get_assets_dir",
    "get_calibration_dir",
    "get_materials_dir",
]
