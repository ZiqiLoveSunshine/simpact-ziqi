"""Tests for the perception adapter layer.

Interfaces and backends must import without the external model repos / GPU.
Repo-backed backends raise a clear, actionable error when their repo env var is
unset and resolve the repo when it is set (construction is lazy — no model load).
Grounded-SAM-2 is repo-OPTIONAL (GroundingDINO from transformers, SAM2 from the
wheel); it only requires the SAM2 checkpoint.
"""
import pytest

from simpact.real2sim import perception as P
from simpact.real2sim.perception import PerceptionRepoNotFound, REPO_ENV_VARS

# Backends that locate an external repo clone via add_repo_to_syspath in __init__.
REPO_REQUIRED = [
    (P.Hunyuan3DReconstructor, "hunyuan3d"),
    (P.FoundationPoseEstimator, "foundationpose"),
    (P.SAM3DReconstructor, "sam3d"),
]


def test_interfaces_and_backends_import():
    for name in (
        "Segmenter",
        "ImageTo3DReconstructor",
        "PoseEstimator",
        "SegmentationResult",
        "Reconstruction",
        "PoseEstimate",
        "GroundedSAM2Segmenter",
        "SAM3DReconstructor",
        "FoundationPoseEstimator",
    ):
        assert hasattr(P, name)


@pytest.mark.parametrize("backend,key", REPO_REQUIRED)
def test_repo_backend_requires_env(backend, key, monkeypatch):
    monkeypatch.delenv(REPO_ENV_VARS[key], raising=False)
    with pytest.raises(PerceptionRepoNotFound, match=REPO_ENV_VARS[key]):
        backend()


@pytest.mark.parametrize("backend,key", REPO_REQUIRED)
def test_repo_backend_rejects_bad_path(backend, key, monkeypatch):
    monkeypatch.setenv(REPO_ENV_VARS[key], "/no/such/repo/dir")
    with pytest.raises(PerceptionRepoNotFound, match="existing directory"):
        backend()


@pytest.mark.parametrize("backend,key", REPO_REQUIRED)
def test_repo_backend_resolves_when_set(backend, key, tmp_path, monkeypatch):
    # a configured (if empty) repo dir lets __init__ resolve the path lazily;
    # the heavy model load only happens on first use.
    monkeypatch.setenv(REPO_ENV_VARS[key], str(tmp_path))
    obj = backend()
    assert obj.repo == tmp_path


def test_gsam2_requires_sam2_checkpoint(monkeypatch):
    # repo-optional: with neither the checkpoint nor a repo set, fail clearly.
    monkeypatch.delenv("SIMPACT_SAM2_CHECKPOINT", raising=False)
    monkeypatch.delenv(REPO_ENV_VARS["grounded_sam2"], raising=False)
    with pytest.raises(PerceptionRepoNotFound, match="SAM2 checkpoint"):
        P.GroundedSAM2Segmenter()


def test_gsam2_accepts_checkpoint_without_repo(monkeypatch, tmp_path):
    ckpt = tmp_path / "sam2.1_hiera_large.pt"
    ckpt.write_bytes(b"")
    monkeypatch.setenv("SIMPACT_SAM2_CHECKPOINT", str(ckpt))
    monkeypatch.delenv(REPO_ENV_VARS["grounded_sam2"], raising=False)
    seg = P.GroundedSAM2Segmenter()  # no repo needed; construction is lazy
    assert seg.sam2_checkpoint == str(ckpt)
