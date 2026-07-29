"""Phase 3 tests for the generator package.

Covers the secure client accessor and prompt-context loading. No network calls
and no API key are required — these verify the wiring, not Gemini itself.
"""
import pytest

from simpact import generator
from simpact.generator import (
    get_model_id,
    list_context_templates,
    load_context_template,
)


def test_public_surface_imports():
    for name in (
        "get_gemini_client",
        "get_model_id",
        "get_contexts_dir",
        "list_context_templates",
        "load_context_template",
    ):
        assert hasattr(generator, name)


def test_model_id_default_and_override(monkeypatch):
    monkeypatch.delenv("GOOGLE_MODEL_ID", raising=False)
    assert get_model_id() == "gemini-2.5-pro"
    monkeypatch.setenv("GOOGLE_MODEL_ID", "gemini-flash-test")
    assert get_model_id() == "gemini-flash-test"


def test_get_gemini_client_requires_key(monkeypatch):
    # reset the cached singleton and clear the key
    import simpact.generator.client as client_mod
    client_mod._client = None
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        client_mod.get_gemini_client()


def test_context_templates_present():
    names = list_context_templates()
    # the packaged templates cover the four released task cases (the templates
    # for tasks with no bundled example were removed in the release cleanup)
    assert "push" in names
    assert "rope" in names
    assert len(names) >= 4


def test_load_context_template_by_name_and_alias():
    by_full = load_context_template("context_push")
    by_alias = load_context_template("push")
    assert by_full == by_alias
    assert len(by_full) > 0


def test_load_context_template_missing_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_context_template("definitely_not_a_real_template")
