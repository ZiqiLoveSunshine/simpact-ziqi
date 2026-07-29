"""Robot action primitives and proposal containers.

The shared, stable schema for the propose→rollout→optimize action pipeline: a *proposal* is an
ordered sequence of coarse manipulation primitives. This is the seam between
action sampling (generator), evaluation (executor sim rollouts), and optimization.

The JSON contract is byte-compatible with the original generator's output
(``{"action_proposals": [{"description"?, "action_sequence": [{"type", ...}]}]}``).
See docs/ and the original ``generator/propose_template_primitive.txt``.
"""

from simpact.actions.primitives import (
    ActionProposal,
    Descend,
    Flick,
    Grasp,
    GripperControl,
    Lift,
    Move,
    Primitive,
    PLAN_ACTION_TYPES,
    PRIMITIVE_TYPES,
    ProposalSet,
    Push,
    Release,
    Roll,
    Rotate,
    primitive_from_dict,
)

__all__ = [
    "Primitive",
    # propose-stage primitives
    "Push",
    "Lift",
    "Descend",
    "Grasp",
    "Release",
    "Rotate",
    "Roll",
    "Flick",
    "PRIMITIVE_TYPES",
    # optimizer-output plan actions
    "Move",
    "GripperControl",
    "PLAN_ACTION_TYPES",
    # parsing + containers
    "primitive_from_dict",
    "ActionProposal",
    "ProposalSet",
]
