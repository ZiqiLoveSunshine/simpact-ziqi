"""ARAP (as-rigid-as-possible) embedded deformation graph simulator.

Migrated verbatim from the original ``torch_arap`` (algorithm internals are research code
and must not be modified). Depends on optional packages (open3d, pypose, scipy,
scikit-learn, trimesh, matplotlib); install via the ``arap`` extra:

    pip install -e ".[arap]"

The public symbols are imported lazily so that ``import simpact.simulators.arap``
does not hard-fail when those optional dependencies are absent.
"""

try:
    from simpact.simulators.arap.embed_deform_graph import (
        DeformState,
        EmbedDeformGraph,
        PlantSimulatorConfig,
        make_embed_deform_graph,
    )

    _ARAP_AVAILABLE = True
except ImportError:
    # optional deps (open3d / pypose / scipy / sklearn / trimesh) not installed
    DeformState = None
    EmbedDeformGraph = None
    PlantSimulatorConfig = None
    make_embed_deform_graph = None
    _ARAP_AVAILABLE = False

__all__ = [
    "DeformState",
    "EmbedDeformGraph",
    "PlantSimulatorConfig",
    "make_embed_deform_graph",
]
