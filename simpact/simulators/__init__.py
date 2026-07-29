"""Physics simulators.

Backends (``mpm``, ``arap``) depend on optional packages (warp, torch) and are
imported lazily by their submodules. Import the concrete simulator directly,
e.g. ``from simpact.simulators.mpm import MPM_Simulator_WARP``.
"""

__all__ = []
