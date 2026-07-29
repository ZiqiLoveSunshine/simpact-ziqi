"""Phase 2 import smoke tests for the ARAP simulator.

The ``simpact.simulators.arap`` package import must never hard-fail, even when
its optional deps (open3d, pypose, scipy, sklearn, trimesh, matplotlib) are
absent — symbols fall back to ``None`` in that case. When the deps ARE present,
the concrete classes must be importable.
"""
import importlib

import pytest


def test_arap_package_imports_without_optional_deps():
    """Importing the package must not raise regardless of optional deps."""
    mod = importlib.import_module("simpact.simulators.arap")
    assert hasattr(mod, "EmbedDeformGraph")  # symbol exists (may be None)


def test_arap_symbols_present_when_deps_installed():
    mod = importlib.import_module("simpact.simulators.arap")
    if mod.EmbedDeformGraph is None:
        pytest.skip("arap optional deps not installed; symbols are None (expected)")
    # deps present: the full public surface must resolve to real objects
    for name in (
        "DeformState",
        "EmbedDeformGraph",
        "PlantSimulatorConfig",
        "make_embed_deform_graph",
    ):
        assert getattr(mod, name) is not None, f"{name} should be importable"
