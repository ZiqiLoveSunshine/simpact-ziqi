"""Tests for geometric rope endpoint detection (Phase 1 of VLM endpoint grounding).

Two layers:
  * a hermetic synthetic U-shaped rope with KNOWN ends (precise ground truth), and
  * the two committed real clouds — where we validate that the detector finds the
    *true geometric ends* and associates them 1:1 with the hand-clicked
    ``scene.yaml`` points. We deliberately do NOT assert a tight match to the
    hand-clicks: they were placed by a human ~3-6 cm inward from the actual rope
    tips (hundreds of real cloud points lie beyond each click), so the detector is
    more accurate than its "ground truth" here.

All CPU (numpy/scipy/open3d) — no GPU needed.
"""
import os

import numpy as np
import pytest
import yaml

from simpact.real2sim.detect_rope_endpoints import (
    TIP_RADIUS,
    detect_from_ply,
    detect_rope_endpoints,
)

SCENES = ["examples/rope_real2sim/1102_rope_8",
          "examples/rope_real2sim/1102_rope_11"]


# --- hermetic: synthetic U-rope with known endpoints ------------------------

def _synthetic_u(n=400, R=0.15, cx=0.45, cy=0.0, z=0.15, noise=0.001, seed=0):
    """A semicircular 'U' rope tube; true ends at (cx-R,cy,z) and (cx+R,cy,z)."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(np.pi, 2 * np.pi, n)          # lower half-circle (U opens up)
    x = cx + R * np.cos(theta)
    y = cy + R * np.sin(theta)
    pts = np.stack([x, y, np.full_like(x, z)], axis=1)
    pts = pts + rng.normal(0, noise, pts.shape)        # thin-tube jitter
    end0 = np.array([cx - R, cy, z])                    # theta = pi
    end1 = np.array([cx + R, cy, z])                    # theta = 2*pi
    return pts, end0, end1


def test_synthetic_u_finds_known_ends():
    pts, end0, end1 = _synthetic_u()
    r = detect_rope_endpoints(pts)
    assert r.method == "geodesic"
    assert r.component_frac == 1.0
    # each true end is matched by some detected tip within ~1.5 cm
    for gt in (end0, end1):
        d = min(np.linalg.norm(r.tip_a - gt), np.linalg.norm(r.tip_b - gt))
        assert d < 0.015, f"true end {gt} not detected (nearest tip {d*100:.1f} cm)"
    # the two detected tips are distinct and well separated (curve, not a blob)
    assert np.linalg.norm(r.tip_a - r.tip_b) > 0.20
    assert r.geodesic_len >= r.euclidean_len - 1e-6   # geodesic follows the arc


def test_synthetic_confidence_high():
    pts, _, _ = _synthetic_u()
    assert detect_rope_endpoints(pts).confidence >= 0.8


def test_pca_fallback_on_fragmented_cloud():
    # three far-apart specks -> no component holds >= half the nodes -> fallback
    rng = np.random.default_rng(1)
    clusters = [rng.normal([x, 0.0, 0.15], 0.002, (20, 3)) for x in (0.3, 0.5, 0.7)]
    r = detect_rope_endpoints(np.vstack(clusters), connect_radius=0.01)
    assert r.method == "pca_fallback"
    # extremes still span the outer clusters
    assert np.linalg.norm(r.tip_a - r.tip_b) > 0.30


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        detect_rope_endpoints(np.zeros((2, 3)))


# --- real committed clouds --------------------------------------------------

@pytest.mark.parametrize("scene", SCENES)
def test_real_cloud_two_clean_tips(scene):
    if not os.path.exists(f"{scene}/sim/segmented_object.ply"):
        pytest.skip(f"missing example cloud: {scene}")
    r = detect_from_ply(f"{scene}/sim/segmented_object.ply")
    assert r.method == "geodesic"
    assert r.component_frac == 1.0
    assert r.confidence >= 0.8
    # a real rope spans a good fraction of the workspace
    assert r.geodesic_len > 0.30 and r.euclidean_len > 0.30


@pytest.mark.parametrize("scene", SCENES)
def test_real_tips_lie_on_cloud(scene):
    ply = f"{scene}/sim/segmented_object.ply"
    if not os.path.exists(ply):
        pytest.skip(f"missing example cloud: {scene}")
    import open3d as o3d
    pts = np.asarray(o3d.io.read_point_cloud(ply).points)
    r = detect_rope_endpoints(pts)
    for tip in (r.tip_a, r.tip_b):
        assert np.linalg.norm(pts - tip, axis=1).min() < TIP_RADIUS


@pytest.mark.parametrize("scene", SCENES)
def test_real_tips_are_true_ends_and_associate(scene):
    """Each detected tip is a genuine geometric extreme (few cloud points beyond it)
    and maps 1:1 to the two hand-clicked scene.yaml points (correct end id)."""
    ply = f"{scene}/sim/segmented_object.ply"
    if not os.path.exists(ply):
        pytest.skip(f"missing example cloud: {scene}")
    import open3d as o3d
    pts = np.asarray(o3d.io.read_point_cloud(ply).points)
    y = yaml.safe_load(open(f"{scene}/sim/scene.yaml"))
    gt = {"fixed": np.array(y["fixed_point"], float),
          "free": np.array(y["free_end"], float)}
    r = detect_rope_endpoints(pts)

    # 1:1 association: the two tips claim different hand-clicks (no collision)
    near_a = min(gt, key=lambda k: np.linalg.norm(gt[k] - r.tip_a))
    near_b = min(gt, key=lambda k: np.linalg.norm(gt[k] - r.tip_b))
    assert near_a != near_b, "both tips associate to the same hand-click"

    n = len(pts)
    for tip, key in ((r.tip_a, near_a), (r.tip_b, near_b)):
        click = gt[key]
        # correct end (generous: hand-clicks are ~3-6 cm inward of the true tip)
        assert np.linalg.norm(tip - click) < 0.08
        # genuine extreme: far more cloud lies beyond the CLICK than beyond the TIP,
        # and only a sliver of the cloud is past the tip (it really is the end)
        axis = tip - click
        L = np.linalg.norm(axis)
        if L > 1e-3:
            proj = (pts - click) @ (axis / L)
            beyond_tip = int((proj > L).sum())
            beyond_click = int((proj > 0).sum())
            assert beyond_tip < beyond_click, f"{key}: tip not past the hand-click"
            assert beyond_tip < 0.05 * n, f"{key}: {beyond_tip}/{n} cloud past the tip"
