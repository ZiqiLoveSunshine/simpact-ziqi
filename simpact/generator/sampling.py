"""Random action sampling — seeded ported from the original ``generate_random_proposals.py``.

Samples action-primitive sequences from per-type parameter ranges. Used as a
baseline sampler and as the building block for the later CEM optimizer (which
samples around an evolving mean/std rather than uniform ranges).

Difference from the original sampler: the RNG is seedable (the original pipeline's is global/unseeded) for
reproducible tests; values are rounded to 4 decimals as in the original pipeline. Ranges are keyed
by primitive attribute names (e.g. GRASP uses ``width``), matching the schema.
"""
from __future__ import annotations

import math
import random
from typing import Optional

from simpact.actions.primitives import (
    ActionProposal,
    Descend,
    Flick,
    Grasp,
    Lift,
    Primitive,
    ProposalSet,
    Push,
    Release,
    Roll,
    Rotate,
)

# Default per-primitive parameter ranges (matches the original random sampler;
# ROTATE uses ±pi). Keyed by attribute name -> (min, max).
DEFAULT_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "PUSH": {"delta_x": (-0.5, 0.5), "delta_y": (-0.5, 0.5)},
    "LIFT": {"delta_z": (0.0, 0.5)},
    "DESCEND": {"delta_z": (0.0, 0.5)},
    "GRASP": {"width": (0.0, 0.1)},
    "RELEASE": {},
    "ROTATE": {"delta_yaw": (-math.pi, math.pi)},
    "ROLL": {"delta_roll": (-math.pi, math.pi)},
    "FLICK": {"delta_x": (-0.5, 0.5), "delta_y": (-0.5, 0.5), "delta_z": (-0.5, 0.5)},
}

_CTORS = {
    "PUSH": Push,
    "LIFT": Lift,
    "DESCEND": Descend,
    "GRASP": Grasp,
    "RELEASE": Release,
    "ROTATE": Rotate,
    "ROLL": Roll,
    "FLICK": Flick,
}


class RandomProposer:
    """Sample random primitive sequences with reproducible (seeded) RNG."""

    def __init__(
        self,
        ranges: Optional[dict[str, dict[str, tuple[float, float]]]] = None,
        seed: Optional[int] = None,
    ):
        self.ranges = ranges if ranges is not None else DEFAULT_RANGES
        self._rng = random.Random(seed)

    def sample_action(
        self,
        action_type: str,
        ranges: Optional[dict[str, dict[str, tuple[float, float]]]] = None,
    ) -> Primitive:
        r = (ranges or self.ranges).get(action_type, {})
        kwargs = {p: round(self._rng.uniform(lo, hi), 4) for p, (lo, hi) in r.items()}
        return _CTORS[action_type](**kwargs)

    def sample_sequence(
        self,
        action_types: list[str],
        min_len: int = 1,
        max_len: int = 5,
        ranges: Optional[dict[str, dict[str, tuple[float, float]]]] = None,
    ) -> list[Primitive]:
        n = self._rng.randint(min_len, max_len)
        return [
            self.sample_action(self._rng.choice(action_types), ranges) for _ in range(n)
        ]

    def sample(
        self,
        n: int = 20,
        action_types: Optional[list[str]] = None,
        min_len: int = 1,
        max_len: int = 5,
        ranges: Optional[dict[str, dict[str, tuple[float, float]]]] = None,
    ) -> ProposalSet:
        """Generate ``n`` random proposals (each a sequence of length [min,max])."""
        types = action_types if action_types is not None else list((ranges or self.ranges))
        proposals = [
            ActionProposal(
                action_sequence=self.sample_sequence(types, min_len, max_len, ranges)
            )
            for _ in range(n)
        ]
        return ProposalSet(proposals)
