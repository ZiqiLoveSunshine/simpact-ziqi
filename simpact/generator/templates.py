"""Loading of LLM prompt context templates.

Every generator script reads a context template (e.g. ``push.txt``)
from disk. The original scripts passed an absolute ``--context_template`` path on the command
line; here the templates ship with the package under ``prompts/contexts/`` and
are loaded by name, with an optional override path for ad-hoc templates.
"""
from pathlib import Path

from simpact.utils.config import get_project_root


def get_contexts_dir() -> Path:
    """Directory holding the packaged context templates."""
    return get_project_root() / "prompts" / "contexts"


def list_context_templates() -> list[str]:
    """Return the available template names (without the ``.txt`` suffix)."""
    contexts_dir = get_contexts_dir()
    if not contexts_dir.is_dir():
        return []
    return sorted(p.stem for p in contexts_dir.glob("*.txt"))


def load_context_template(name_or_path: str) -> str:
    """Load a context template's text.

    ``name_or_path`` may be a bare template name (``"push"`` or
    ``"context_push"``), a filename, or an absolute/relative path to a custom
    template file. Raises ``FileNotFoundError`` with the list of known
    templates when a named template cannot be found.
    """
    # explicit path wins
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate.read_text()

    contexts_dir = get_contexts_dir()
    stem = candidate.name
    if stem.endswith(".txt"):
        stem = stem[: -len(".txt")]
    # allow both "push" and the legacy "context_push" spelling
    names = [stem]
    names.append(stem[len("context_"):] if stem.startswith("context_") else f"context_{stem}")
    for n in names:
        path = contexts_dir / f"{n}.txt"
        if path.is_file():
            return path.read_text()

    available = ", ".join(list_context_templates()) or "(none found)"
    raise FileNotFoundError(
        f"context template {name_or_path!r} not found in {contexts_dir}. "
        f"Available: {available}"
    )
