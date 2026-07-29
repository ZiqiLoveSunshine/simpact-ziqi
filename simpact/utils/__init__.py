"""Shared utilities: configuration and path resolution."""

from simpact.utils.config import (
    get_project_root,
    get_data_dir,
    get_outputs_dir,
    get_rollouts_dir,
)

__all__ = [
    "get_project_root",
    "get_data_dir",
    "get_outputs_dir",
    "get_rollouts_dir",
]
