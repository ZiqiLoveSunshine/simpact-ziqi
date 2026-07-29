"""Shared test fixtures/config.

Makes the repo's ``scripts/`` directory importable so CLI drivers (e.g.
``run_real2sim``) can be exercised directly from tests without packaging them, and
registers shared skip markers (e.g. ``@pytest.mark.requires_api`` for tests that need a
live Gemini call — skipped unless ``GOOGLE_API_KEY`` is set).
"""
import os
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_api: needs GOOGLE_API_KEY (live VLM call)")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("GOOGLE_API_KEY"):
        return
    skip = pytest.mark.skip(reason="needs GOOGLE_API_KEY")
    for item in items:
        if "requires_api" in item.keywords:
            item.add_marker(skip)
