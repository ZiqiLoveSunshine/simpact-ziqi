"""Bundled-trial layout resolution.

A bundled example trial (examples/<family>/<trial>/) separates its files by role:

    <trial>/capture/   raw recording — RGB-D, initial_ee_pose.txt
    <trial>/sim/       derived simulation assets — scene.yaml, point clouds, meshes
    <trial>/runs/      recorded planning outputs — propose.json, rollouts, ...

External bundles (propose→rollout→optimize trial dirs, perception build outputs) stay flat.
Loaders resolve any per-trial input through :func:`find_scene_file`, which
searches ``sim/``, ``capture/``, then the dir root — so both layouts work with
the same ``--data_dir``/``--scene`` argument.
"""
from pathlib import Path
from typing import Optional, Union

_SUBDIRS = ("sim", "capture", "")


def find_scene_file(scene_dir: Union[str, Path], name: str,
                    required: bool = True) -> Optional[Path]:
    """Resolve ``name`` inside a trial dir across both layouts.

    Searches ``<scene_dir>/sim/``, ``<scene_dir>/capture/``, then the flat
    ``<scene_dir>/`` root; returns the first hit. Raises ``FileNotFoundError``
    when ``required`` and absent, else returns ``None``.
    """
    root = Path(scene_dir)
    for sub in _SUBDIRS:
        p = (root / sub / name) if sub else (root / name)
        if p.exists():
            return p
    if required:
        raise FileNotFoundError(
            f"{name} not found under {root} (searched sim/, capture/, and the dir root)")
    return None
