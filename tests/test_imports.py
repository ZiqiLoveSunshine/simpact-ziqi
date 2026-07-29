"""Phase 1 import smoke tests.

These must pass WITHOUT optional hardware/sim deps (warp, torch, pyrealsense2)
installed — importing the package and resolving config paths is dependency-free.
"""
import importlib

import pytest


def test_import_package():
    import simpact
    assert simpact.__version__ == "0.1.0"


def test_config_helpers():
    from simpact import get_data_dir, get_outputs_dir, get_project_root
    assert get_project_root().name == "simpact"
    assert get_data_dir() is not None
    assert get_outputs_dir() is not None


def test_config_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SIMPACT_DATA_DIR", str(tmp_path))
    # reload so the env var is re-read
    import simpact.utils.config as cfg
    importlib.reload(cfg)
    assert str(cfg.get_data_dir()) == str(tmp_path)


def test_mpm_module_importable_or_skipped():
    """The mpm package import must not raise even when warp is absent."""
    mod = importlib.import_module("simpact.simulators.mpm")
    if not hasattr(mod, "MPM_Simulator_WARP") or mod.MPM_Simulator_WARP is None:
        pytest.skip("warp not installed; MPM_Simulator_WARP unavailable (expected)")
