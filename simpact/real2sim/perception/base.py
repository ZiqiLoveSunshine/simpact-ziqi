"""Abstract interfaces for the real2sim perception stages.

These decouple the rest of the pipeline (scale/pose/XML generation) from the
concrete model backends. Each backend lives in its own module and is selected
explicitly; all backends conform to the interfaces here.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SegmentationResult:
    """Per-object segmentation output for a single image."""

    masks: np.ndarray  # (N, H, W) boolean/uint8 masks
    labels: list[str]  # length-N class labels
    scores: np.ndarray  # (N,) detection confidences
    boxes: np.ndarray  # (N, 4) xyxy boxes


@dataclass
class Reconstruction:
    """Image-to-3D reconstruction output."""

    mesh_path: Path  # path to the generated mesh (GLB/OBJ)
    textured: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class PoseEstimate:
    """6-DoF object pose in the camera frame."""

    pose_cam: np.ndarray  # (4, 4) homogeneous transform, object->camera
    object_name: str
    camera_id: int = 0
    score: Optional[float] = None


class Segmenter(ABC):
    """Open-vocabulary instance segmentation (e.g. GroundingDINO + SAM2)."""

    @abstractmethod
    def segment(self, image: np.ndarray, text_prompt: str) -> SegmentationResult:
        """Segment objects described by ``text_prompt`` in ``image`` (RGB, HxWx3)."""
        raise NotImplementedError


class ImageTo3DReconstructor(ABC):
    """Single-image-to-3D mesh reconstruction (e.g. Hunyuan3D)."""

    @abstractmethod
    def reconstruct(self, image_path: Path, output_dir: Path) -> Reconstruction:
        """Reconstruct a textured 3D mesh from a single cropped object image."""
        raise NotImplementedError


class PoseEstimator(ABC):
    """Model-based 6-DoF pose estimation (e.g. FoundationPose)."""

    @abstractmethod
    def estimate(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: np.ndarray,
        mesh_path: Path,
        object_name: str,
        camera_id: int = 0,
    ) -> PoseEstimate:
        """Estimate the 6-DoF pose of ``object_name`` given RGB-D + mask + mesh."""
        raise NotImplementedError
