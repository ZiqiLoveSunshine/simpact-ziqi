"""Camera abstractions for real2sim capture.

``sensor``/``camera_apis`` define device-agnostic interfaces (Klamp't-based);
``camera`` wraps Intel RealSense via ``pyrealsense2``, which is
hardware-only and imported under a guard. The RealSense classes are exposed
lazily and degrade to ``None`` when ``pyrealsense2`` is not installed.
"""

from simpact.real2sim.cameras.camera_apis import (
    CameraIntrinsics,
    CameraSettings,
    RGBDCamera,
    RGBDCameraSettings,
)

try:
    from simpact.real2sim.cameras.camera import Camera, rs as _rs

    # Camera class imports even without the driver; it can only be *instantiated*
    # when pyrealsense2 is actually installed.
    _REALSENSE_AVAILABLE = _rs is not None
except ImportError:
    # real2sim extra (e.g. open3d) not installed
    Camera = None
    _REALSENSE_AVAILABLE = False

__all__ = [
    "CameraIntrinsics",
    "CameraSettings",
    "RGBDCamera",
    "RGBDCameraSettings",
    "Camera",
]
