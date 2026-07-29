"""Secure, lazily-initialized Gemini client.

The API key is read from the ``GOOGLE_API_KEY`` environment variable (loaded
from ``.env`` by ``simpact.utils.config``) — never hardcoded. ``google-genai``
is imported lazily inside ``get_gemini_client`` so that importing the package
does not require the optional ``generator`` extra to be installed.
"""
import os

# ensure .env is loaded so GOOGLE_API_KEY is available
import simpact.utils.config  # noqa: F401

_client = None


def get_gemini_client():
    """Return a process-wide singleton ``google.genai.Client``.

    Raises ``RuntimeError`` if ``GOOGLE_API_KEY`` is unset, or ``ImportError``
    if the ``generator`` extra (``google-genai``) is not installed.
    """
    global _client
    if _client is None:
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai is required for the generator. "
                'Install it with: pip install -e ".[generator]"'
            ) from e
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Copy .env.example to .env and fill in your key."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def get_model_id() -> str:
    """Model id for generation, overridable via ``GOOGLE_MODEL_ID``."""
    return os.environ.get("GOOGLE_MODEL_ID", "gemini-2.5-pro")
