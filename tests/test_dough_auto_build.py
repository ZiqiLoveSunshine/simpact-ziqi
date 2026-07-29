"""Opt-in live integration: fully-automated dough real2sim build.

Re-runs the REAL pipeline (``simpact.real2sim.build_scene.build_scene`` — no
reimplementation) fully headless on a recorded RGB-D + EE-pose trial: live
Grounded-SAM-2 segmentation -> registry calibration -> single-view column fill ->
VLM material-ID. Asserts the generated MPM cloud reproduces the committed reference
reference (bbox within tolerance). This is the test form of a former one-off
demo script (removed with the experiments folder).

Runs against the bundled trial's capture/ by default (it carries the full RGB-D + EE
record); SIMPACT_DOUGH_RAW_DIR overrides with an external raw trial. Gated (skips
unless present): GOOGLE_API_KEY (material VLM), SIMPACT_GROUNDED_SAM2_DIR
(segmenter), open3d.

    SIMPACT_GROUNDED_SAM2_DIR=/path/to/Grounded-SAM-2 GOOGLE_API_KEY=... \
    uv run pytest tests/test_dough_auto_build.py
"""
import os

import numpy as np
import pytest

_BUNDLED_CAPTURE = "examples/dough_real2sim/1104_sand_6/capture"  # full RGB-D + EE record
RAW_DIR = os.environ.get("SIMPACT_DOUGH_RAW_DIR") or _BUNDLED_CAPTURE
REFERENCE = "examples/dough_real2sim/1104_sand_6/sim/mpm_points.npy"


def _bbox_cm(p):
    return (p.max(0) - p.min(0))[:3] * 100


@pytest.mark.requires_api
@pytest.mark.skipif(not os.path.isdir(RAW_DIR),
                    reason="no raw RGB-D trial (bundled capture missing and no SIMPACT_DOUGH_RAW_DIR)")
def test_dough_auto_build_matches_reference(tmp_path):
    # the live segmenter needs only a RESOLVABLE SAM2 checkpoint (the GSAM2 repo
    # clone is optional; a /path/to/... .env placeholder must not count)
    ckpt = os.environ.get("SIMPACT_SAM2_CHECKPOINT")
    if not (ckpt and os.path.exists(ckpt)):
        gs = os.environ.get("SIMPACT_GROUNDED_SAM2_DIR") or ""
        ckpt = os.path.join(gs, "checkpoints/sam2.1_hiera_large.pt")
        if not os.path.exists(ckpt):
            pytest.skip("no resolvable SAM2 checkpoint "
                        "(SIMPACT_SAM2_CHECKPOINT or SIMPACT_GROUNDED_SAM2_DIR/checkpoints/)")
    pytest.importorskip("open3d")
    from pathlib import Path

    from simpact.real2sim.build_scene import build_scene
    from simpact.real2sim.perception.grounded_sam2 import GroundedSAM2Segmenter

    raw = Path(RAW_DIR)
    object_name = os.environ.get("SIMPACT_DOUGH_RAW_OBJECT", "blue playdoh")
    profile = os.environ.get("SIMPACT_DOUGH_RAW_PROFILE", "1026")
    ee = raw / "initial_ee_pose.txt"

    out = tmp_path / "scene"
    build_scene(raw, out, "dough", object_name, cam=1, profile=profile,
                ee_pose_path=str(ee) if ee.exists() else None,
                allow_home_pose=not ee.exists(),
                segmenter=GroundedSAM2Segmenter(device="cuda", sam2_checkpoint=ckpt))

    pts = np.load(out / "mpm_points.npy")
    assert pts.ndim == 2 and pts.shape[1] == 3 and len(pts) > 0

    if os.path.exists(REFERENCE):  # the committed reference cloud
        ref = np.load(REFERENCE)
        got, want = _bbox_cm(pts), _bbox_cm(ref)
        # segmentation/VLM introduce small run-to-run variation; require the automated
        # build to reproduce the reference footprint to within ~1.5 cm per axis.
        np.testing.assert_allclose(got, want, atol=1.5,
                                   err_msg=f"bbox cm got={got.tolist()} want={want.tolist()}")
