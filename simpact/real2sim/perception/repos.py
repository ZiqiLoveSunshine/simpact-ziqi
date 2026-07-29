"""Resolution of external perception model repositories.

The perception stages wrap large third-party repos that are NOT pip-installable
and ship their own model weights (Grounded-SAM-2, Hunyuan3D-2.1, FoundationPose).
Rather than vendoring them, each is located at runtime via an environment
variable pointing at a local clone. This keeps ``simpact`` importable and
testable without the repos present, and fails with a clear, actionable message
only when a perception stage is actually used.
"""
import os
import sys
from pathlib import Path

# env var -> human-readable repo name (for error messages)
REPO_ENV_VARS = {
    "grounded_sam2": "SIMPACT_GROUNDED_SAM2_DIR",
    "hunyuan3d": "SIMPACT_HUNYUAN3D_DIR",
    "foundationpose": "SIMPACT_FOUNDATIONPOSE_DIR",
    "sam3d": "SIMPACT_SAM3D_DIR",
}

_REPO_URLS = {
    "grounded_sam2": "https://github.com/IDEA-Research/Grounded-SAM-2",
    "hunyuan3d": "https://github.com/Tencent/Hunyuan3D-2.1 (model id: tencent/Hunyuan3D-2.1)",
    "foundationpose": "https://github.com/NVlabs/FoundationPose",
    "sam3d": "https://github.com/facebookresearch/sam-3d-objects",
}


class PerceptionRepoNotFound(RuntimeError):
    """Raised when an external perception repo is not configured/available."""


def get_repo_dir(name: str) -> Path:
    """Return the local path to external repo ``name``.

    Raises ``PerceptionRepoNotFound`` if the corresponding env var is unset or
    does not point at an existing directory.
    """
    if name not in REPO_ENV_VARS:
        raise KeyError(f"unknown perception repo {name!r}; known: {list(REPO_ENV_VARS)}")
    env_var = REPO_ENV_VARS[name]
    raw = os.environ.get(env_var)
    if not raw:
        raise PerceptionRepoNotFound(
            f"{name} backend requires the external repo. Clone "
            f"{_REPO_URLS[name]} and set {env_var} to its path."
        )
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise PerceptionRepoNotFound(
            f"{env_var}={raw!r} does not point at an existing directory."
        )
    return path


def add_repo_to_syspath(name: str, *subdirs: str) -> Path:
    """Resolve repo ``name`` and prepend it (and optional subdirs) to sys.path.

    Mirrors the original pipeline's ``sys.path.append(.../<repo>)`` pattern but driven by config.
    Returns the repo path.
    """
    repo = get_repo_dir(name)
    for entry in (repo, *(repo / s for s in subdirs)):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)
    return repo
