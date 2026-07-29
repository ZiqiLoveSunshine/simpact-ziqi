"""Shared headless renderer for deformable rollouts (MPM / ARAP), PyVista-based.

Renders the material as **3D shaded sphere glyphs** (``render_points_as_spheres``) from
the scene's real camera pose, so the deformable's shape reads as a solid object and the
tool (gripper jaws / pusher box) is **depth-composited** against the material — a box
behind the pile is occluded by it, a box in front occludes it (the flat 2-D projection
this replaced drew the tool on top regardless of depth). PyVista ``off_screen`` renders
headless via VTK (validated on the 5090; docs/DEFORMABLE_INTEGRATION.md §4).

The camera is built from the scene's real ``K`` + ``cam_to_robot`` extrinsic (position,
optical axis, up, vertical FOV from ``fy``) so before/after/video frames share the real
viewpoint. Also writes an mp4 of a full rollout via ``VideoRecorder`` (cv2, mp4v).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

IMG_W, IMG_H = 640, 480
DOUGH_COLOR = (0.82, 0.68, 0.45)   # uniform tan — no per-point colormap (#1)
TOOL_COLOR = (0.85, 0.12, 0.12)    # red tool box
TARGET_COLOR = (0.80, 0.20, 0.80)  # magenta target region


def project(pts: np.ndarray, K: np.ndarray, cam_to_robot: np.ndarray):
    """Project robot-frame points to (u,v) pixels + a front-of-camera mask (utility)."""
    r2c = np.linalg.inv(cam_to_robot)
    Pc = (r2c[:3, :3] @ pts.T).T + r2c[:3, 3]
    uvw = (K @ Pc.T).T
    front = uvw[:, 2] > 1e-6
    uv = uvw[:, :2] / np.where(uvw[:, 2:3] == 0, 1e-6, uvw[:, 2:3])
    return uv, front


def _set_camera(plotter, K, cam_to_robot, img_size):
    """Point the VTK camera to match a pinhole (K, cam_to_robot) — OpenCV convention:
    camera looks down +z_cam, image y points down, so world-up = -y_cam."""
    W, H = img_size
    C = cam_to_robot[:3, 3]
    view_dir = cam_to_robot[:3, 2]      # camera optical axis in world
    up = -cam_to_robot[:3, 1]           # image-y is down -> up is -y_cam
    cam = plotter.camera
    cam.position = tuple(C)
    cam.focal_point = tuple(C + view_dir)
    cam.up = tuple(up)
    cam.view_angle = float(np.degrees(2.0 * np.arctan2(H / 2.0, K[1, 1])))  # vertical FOV from fy
    # principal-point offset (cx,cy vs image centre), VTK normalized window centre
    cam.SetWindowCenter((W - 2.0 * K[0, 2]) / W, (2.0 * K[1, 2] - H) / H)


def _box_mesh(center, quat_xyzw, size):
    import pyvista as pv
    from scipy.spatial.transform import Rotation as R
    box = pv.Cube(center=(0, 0, 0), x_length=size[0], y_length=size[1], z_length=size[2])
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = center
    return box.transform(T, inplace=False)


def render_deformable(
    points: np.ndarray,
    K: np.ndarray,
    cam_to_robot: np.ndarray,
    out_path=None,
    *,
    title: str = "",
    color=DOUGH_COLOR,
    tool_boxes: Optional[Sequence[tuple]] = None,
    extra_points: Optional[np.ndarray] = None,
    extra_color=TARGET_COLOR,
    point_size: float = 14.0,
    img_size=(IMG_W, IMG_H),
    return_array: bool = False,
):
    """Render ``points`` as shaded spheres (+ optional ``tool_boxes`` and
    ``extra_points``) from the camera pose. Saves ``out_path`` (PNG) and/or returns the
    HxWx3 uint8 frame. ``tool_boxes`` = ``(center, quat_xyzw, size)`` drawn as
    depth-composited translucent red boxes; ``extra_points`` (e.g. a sweep target
    region) are drawn as ``extra_color`` spheres."""
    import pyvista as pv
    pv.OFF_SCREEN = True

    pl = pv.Plotter(off_screen=True, window_size=list(img_size))
    pl.set_background("white")
    pl.enable_depth_peeling(10)  # correct ordering for the translucent tool box
    if extra_points is not None and len(extra_points):
        pl.add_mesh(pv.PolyData(np.asarray(extra_points, float)), color=extra_color,
                    render_points_as_spheres=True, point_size=point_size * 0.7)
    pl.add_mesh(pv.PolyData(np.asarray(points, float)), color=color,
                render_points_as_spheres=True, point_size=point_size, diffuse=0.9,
                specular=0.3, specular_power=15)
    for center, quat, size in (tool_boxes or []):
        pl.add_mesh(_box_mesh(center, quat, size), color=TOOL_COLOR, opacity=0.45)
    if title:
        pl.add_text(title, position="upper_edge", font_size=10, color="black")
    _set_camera(pl, np.asarray(K), np.asarray(cam_to_robot), img_size)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # pyvista.screenshot both writes the PNG (when given a path) and returns the array
    frame = pl.screenshot(str(out_path) if out_path is not None else None, return_img=True)
    pl.close()
    if return_array:
        return frame
    return str(out_path) if out_path is not None else frame


class VideoRecorder:
    """Accumulate rendered frames and write an mp4 (cv2 / mp4v — no ffmpeg dep).

    Used to save the **full simulation** of a rollout as a video artifact; this is NOT
    shown to the VLM optimization loop (only the before/after PNGs are).
    """

    def __init__(self, out_path, fps: int = 20, img_size=(IMG_W, IMG_H)):
        self.path = str(out_path)
        self.fps = fps
        self.size = tuple(img_size)
        self.frames: list = []

    def add(self, frame: np.ndarray) -> None:
        self.frames.append(np.asarray(frame, dtype=np.uint8))

    def save(self) -> Optional[str]:
        if not self.frames:
            return None
        import cv2
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
        for f in self.frames:
            if (f.shape[1], f.shape[0]) != self.size:
                f = cv2.resize(f, self.size)
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        return self.path
