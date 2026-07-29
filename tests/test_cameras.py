"""Phase 4B import tests for the camera subpackage.

Device-agnostic interfaces (camera_apis/sensor) must import without any camera
hardware. The RealSense wrapper degrades to ``Camera is None`` when
``pyrealsense2`` is absent rather than raising at import time.
"""
import importlib

import pytest


def test_camera_apis_imports_without_hardware():
    mod = importlib.import_module("simpact.real2sim.cameras")
    for name in (
        "CameraIntrinsics",
        "CameraSettings",
        "RGBDCamera",
        "RGBDCameraSettings",
    ):
        assert getattr(mod, name) is not None


def test_realsense_camera_guarded():
    import simpact.real2sim.cameras as cams

    # The Camera class is importable even without the driver (it's defined under
    # an import guard); _REALSENSE_AVAILABLE reflects whether pyrealsense2 exists.
    assert hasattr(cams, "Camera")
    try:
        import pyrealsense2  # noqa: F401

        assert cams._REALSENSE_AVAILABLE is True
    except ImportError:
        assert cams._REALSENSE_AVAILABLE is False
        # instantiating without the driver must raise a clear ImportError
        if cams.Camera is not None:
            with pytest.raises(ImportError, match="pyrealsense2"):
                cams.Camera({"serial_number": "x"})
