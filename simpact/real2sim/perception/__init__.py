"""Perception backends for real2sim (segmentation, image-to-3D, 6D pose).

Each backend wraps a large external model repo located via an environment
variable (see ``repos.py``). Interfaces are always importable; instantiating a
backend raises ``PerceptionRepoNotFound`` when its repo is not configured.

The rigid backends (``GroundedSAM2Segmenter``, ``SAM3DReconstructor``,
``FoundationPoseEstimator``) are implemented and run in one uv env; see
docs/RIGID_PIPELINE.md and scripts/run_rigid_pipeline.py. ``Hunyuan3DReconstructor``
remains a pending port.
"""

from simpact.real2sim.perception.base import (
    ImageTo3DReconstructor,
    PoseEstimate,
    PoseEstimator,
    Reconstruction,
    SegmentationResult,
    Segmenter,
)
from simpact.real2sim.perception.foundationpose import FoundationPoseEstimator
from simpact.real2sim.perception.grounded_sam2 import GroundedSAM2Segmenter
from simpact.real2sim.perception.hunyuan3d import Hunyuan3DReconstructor
from simpact.real2sim.perception.sam3d import SAM3DReconstructor
from simpact.real2sim.perception.repos import (
    PerceptionRepoNotFound,
    REPO_ENV_VARS,
    get_repo_dir,
)

__all__ = [
    # interfaces
    "Segmenter",
    "ImageTo3DReconstructor",
    "PoseEstimator",
    # result types
    "SegmentationResult",
    "Reconstruction",
    "PoseEstimate",
    # backends
    "GroundedSAM2Segmenter",
    "SAM3DReconstructor",
    "Hunyuan3DReconstructor",
    "FoundationPoseEstimator",
    # repo resolution
    "PerceptionRepoNotFound",
    "REPO_ENV_VARS",
    "get_repo_dir",
]
