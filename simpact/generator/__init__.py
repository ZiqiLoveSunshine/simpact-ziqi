"""LLM-based proposal/planning generator.

Public surface is the secure Gemini client accessor and prompt-context loading.
API keys are read from the ``GOOGLE_API_KEY`` environment variable (never
hardcoded — see ``.env.example``); the original pipeline's per-script hardcoded keys are dropped
on migration.
"""

from simpact.generator.client import get_gemini_client, get_model_id
from simpact.generator.templates import (
    get_contexts_dir,
    list_context_templates,
    load_context_template,
)

__all__ = [
    "get_gemini_client",
    "get_model_id",
    "get_contexts_dir",
    "list_context_templates",
    "load_context_template",
]
