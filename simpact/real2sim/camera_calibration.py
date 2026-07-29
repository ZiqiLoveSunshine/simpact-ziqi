"""Camera-calibration resolution: keyed registry + per-scene reference.

Camera parameters split by natural unit: intrinsics ``K`` are per-camera, extrinsics
``cam_to_robot`` are per-calibration-session. Neither a single frozen global (the old
``transform_6d._EXTRINSIC_BY_CAMERA`` — wrong for scenes from other sessions,
an audit finding) nor per-scene duplication is ideal, so calibration lives in a
**keyed registry** under ``assets/calibration/<profile>/`` and a scene records which
profile it used.

``load_camera(scene_dir, cam, profile)`` resolves in order (first that resolves wins):
  1. **embedded** ``scene_dir/cam{cam}_{K,to_robot}.txt`` — a self-contained/portable
     scene bundle;
  2. a **profile** — explicit ``profile=`` arg, else ``scene.yaml``'s
     ``camera: {profile, cam}`` — looked up in the registry
     (``$SIMPACT_ASSETS_DIR``/``assets`` ``/calibration/<profile>/``);
  3. otherwise a clear error (never a silent frozen default).

The registry ships the rig data outside the code package (repo ``assets/``), so the
package stays pure code. See docs/DEFORMABLE_INTEGRATION.md §14 and the assets-layout note.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from simpact.utils.config import get_calibration_dir


@dataclass
class CameraParams:
    K: np.ndarray               # (3, 3) intrinsics
    cam_to_robot: np.ndarray    # (4, 4) extrinsic (camera -> robot base)
    image_size: Optional[tuple] = None  # (W, H) if known
    source: str = ""            # human-readable provenance


def list_profiles() -> list[str]:
    """Names of the calibration profiles available in the registry."""
    root = get_calibration_dir()
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def load_profile(profile: str, cam: int) -> CameraParams:
    """Load ``cam{cam}`` from a named registry profile (``assets/calibration/<profile>/``)."""
    root = get_calibration_dir() / str(profile)
    K = root / f"cam{cam}_K.txt"
    T = root / f"cam{cam}_to_robot.txt"
    if not (K.exists() and T.exists()):
        avail = list_profiles()
        raise FileNotFoundError(
            f"calibration profile {profile!r} has no cam{cam} (looked in {root}). "
            f"Available profiles: {avail}")
    img = _read_image_size(root / "profile.yaml", cam)
    return CameraParams(np.loadtxt(K).reshape(3, 3), np.loadtxt(T).reshape(4, 4),
                        img, f"registry:{profile}/cam{cam}")


def _read_image_size(profile_yaml: Path, cam: int):
    if not profile_yaml.exists():
        return None
    import yaml
    y = yaml.safe_load(profile_yaml.read_text()) or {}
    entry = (y.get("cameras") or {}).get(f"cam{cam}") or {}
    sz = entry.get("image_size")
    return tuple(sz) if sz else None


def _scene_profile(scene_dir: Path, cam: int):
    """Read a ``camera: {profile, cam}`` reference from the scene's scene.yaml, if any."""
    from simpact.utils.layout import find_scene_file
    yml = find_scene_file(scene_dir, "scene.yaml", required=False) or (scene_dir / "scene.yaml")
    if not yml.exists():
        return None, cam
    import yaml
    cam_block = (yaml.safe_load(yml.read_text()) or {}).get("camera") or {}
    return cam_block.get("profile"), int(cam_block.get("cam", cam))


def load_camera(scene_dir=None, cam: int = 1, profile: Optional[str] = None) -> CameraParams:
    """Resolve camera params: embedded scene files > profile (arg or scene.yaml) > error."""
    if scene_dir is not None:
        from simpact.utils.layout import find_scene_file
        scene_dir = Path(scene_dir)
        K = find_scene_file(scene_dir, f"cam{cam}_K.txt", required=False)
        T = find_scene_file(scene_dir, f"cam{cam}_to_robot.txt", required=False)
        if K is not None and T is not None:  # embedded / portable scene bundle
            return CameraParams(np.loadtxt(K).reshape(3, 3), np.loadtxt(T).reshape(4, 4),
                                None, f"embedded:{scene_dir.name}/cam{cam}")
        if profile is None:  # fall to a scene.yaml profile reference
            profile, cam = _scene_profile(scene_dir, cam)
    if profile is not None:
        return load_profile(profile, cam)
    raise FileNotFoundError(
        f"no camera calibration for cam{cam}: no embedded cam{cam}_K.txt/cam{cam}_to_robot.txt"
        + (f" in {scene_dir}" if scene_dir is not None else "")
        + " and no profile given (arg or scene.yaml 'camera:' block). "
        f"Available registry profiles: {list_profiles()}")
