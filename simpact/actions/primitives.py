"""Action primitive schema + JSON (de)serialization.

The 7 coarse primitives of the original proposal template:
PUSH, LIFT, DESCEND, GRASP, RELEASE, ROTATE, FLICK. Deltas are in the absolute
world frame; ROTATE's yaw is relative.

JSON contract matches the original generator's output exactly so recorded
``proposal*.json`` files round-trip. One real-world wrinkle handled here: the
**GRASP width key is inconsistent in the original pipeline** — the LLM emits ``grasp_width`` (and
that is what every recorded proposal uses), the random sampler and the prompt's
output-format block use ``width``, and the executor reads ``width`` (so it would
silently drop the LLM's grasp). We normalize: the canonical attribute is
``Grasp.width``; ``from_dict`` accepts either key; ``to_dict`` emits
``grasp_width`` (matching the model + recorded data).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar, Optional, Union


class Primitive:
    """Base for action primitives (each concrete subclass is a dataclass)."""

    TYPE: ClassVar[str] = ""

    def _value_field_names(self) -> list[str]:
        return [f.name for f in fields(self) if f.name != "reasoning"]

    def attr_values(self) -> dict[str, float]:
        """Numeric parameters keyed by attribute name (for range validation)."""
        return {n: getattr(self, n) for n in self._value_field_names()}

    def to_dict(self) -> dict:
        d: dict = {"type": self.TYPE, **self.attr_values()}
        if getattr(self, "reasoning", None) is not None:
            d["reasoning"] = self.reasoning
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Primitive":
        names = [f.name for f in fields(cls) if f.name != "reasoning"]
        kwargs: dict = {n: float(d[n]) for n in names}
        if d.get("reasoning") is not None:
            kwargs["reasoning"] = d["reasoning"]
        return cls(**kwargs)


@dataclass
class Push(Primitive):
    """Move horizontally by (delta_x, delta_y) at the current height."""

    delta_x: float
    delta_y: float
    reasoning: Optional[str] = None
    TYPE: ClassVar[str] = "PUSH"


@dataclass
class Lift(Primitive):
    """Move up by delta_z."""

    delta_z: float
    reasoning: Optional[str] = None
    TYPE: ClassVar[str] = "LIFT"


@dataclass
class Descend(Primitive):
    """Move down by delta_z."""

    delta_z: float
    reasoning: Optional[str] = None
    TYPE: ClassVar[str] = "DESCEND"


@dataclass
class Grasp(Primitive):
    """Set gripper to ``width`` (0.0 closed .. 0.1 open).

    Serializes to the ``grasp_width`` JSON key; parses either ``grasp_width`` or
    ``width`` (see module docstring).
    """

    width: float
    reasoning: Optional[str] = None
    TYPE: ClassVar[str] = "GRASP"

    def to_dict(self) -> dict:
        d: dict = {"type": self.TYPE, "grasp_width": self.width}
        if self.reasoning is not None:
            d["reasoning"] = self.reasoning
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Grasp":
        if "grasp_width" in d:
            w = d["grasp_width"]
        elif "width" in d:
            w = d["width"]
        else:
            raise ValueError("GRASP action missing 'grasp_width' (or 'width')")
        kwargs: dict = {"width": float(w)}
        if d.get("reasoning") is not None:
            kwargs["reasoning"] = d["reasoning"]
        return cls(**kwargs)


@dataclass
class Release(Primitive):
    """Open the gripper fully (== GRASP(0.1))."""

    reasoning: Optional[str] = None
    TYPE: ClassVar[str] = "RELEASE"


@dataclass
class Rotate(Primitive):
    """Rotate end-effector yaw by delta_yaw (radians, relative)."""

    delta_yaw: float
    reasoning: Optional[str] = None
    TYPE: ClassVar[str] = "ROTATE"


@dataclass
class Roll(Primitive):
    """Rotate end-effector roll by delta_roll (radians, relative).

    Kept in the schema for round-tripping recorded proposals; no shipped
    proposal template emits it (the roll-variant template was removed with the
    unused pivot task). CoR is at the wrist (~14.5 cm above the fingertips).
    """

    delta_roll: float
    reasoning: Optional[str] = None
    TYPE: ClassVar[str] = "ROLL"


@dataclass
class Flick(Primitive):
    """Move by (delta_x, delta_y, delta_z) in one quick motion."""

    delta_x: float
    delta_y: float
    delta_z: float
    reasoning: Optional[str] = None
    TYPE: ClassVar[str] = "FLICK"


# --- Optimizer-output ("regress") plan actions ------------------------------
# The VLM optimizer emits a *refined plan* in the "universal 6-DoF + gripper"
# format (regress_template.txt), distinct from the propose-stage primitives above.
# Same ProposalSet container; lowercase type tags as the templates specify.
@dataclass
class Move(Primitive):
    """Relative 6-DoF end-effector move (delta_pitch/roll are usually 0)."""

    delta_x: float
    delta_y: float
    delta_z: float
    delta_roll: float
    delta_pitch: float
    delta_yaw: float
    reasoning: Optional[str] = None
    TYPE: ClassVar[str] = "move"


@dataclass
class GripperControl(Primitive):
    """Set the gripper opening ``width``."""

    width: float
    reasoning: Optional[str] = None
    TYPE: ClassVar[str] = "gripper_control"


# propose-stage primitives drive sampling / task profiles
_PRIMITIVE_CLASSES = (Push, Lift, Descend, Grasp, Release, Rotate, Roll, Flick)
# optimizer-output plan actions
_PLAN_CLASSES = (Move, GripperControl)
# both are parseable by primitive_from_dict / ProposalSet
_BY_TYPE = {c.TYPE: c for c in _PRIMITIVE_CLASSES + _PLAN_CLASSES}
PRIMITIVE_TYPES = tuple(c.TYPE for c in _PRIMITIVE_CLASSES)  # ("PUSH", "LIFT", ...)
PLAN_ACTION_TYPES = tuple(c.TYPE for c in _PLAN_CLASSES)     # ("move", "gripper_control")


def primitive_from_dict(d: dict) -> Primitive:
    t = d.get("type")
    if t not in _BY_TYPE:
        raise ValueError(f"unknown primitive type {t!r}; known: {PRIMITIVE_TYPES}")
    return _BY_TYPE[t].from_dict(d)


@dataclass
class ActionProposal:
    """One candidate plan: an ordered sequence of primitives (+ optional description)."""

    action_sequence: list[Primitive]
    description: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict = {}
        if self.description is not None:
            d["description"] = self.description
        d["action_sequence"] = [a.to_dict() for a in self.action_sequence]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ActionProposal":
        seq = [primitive_from_dict(a) for a in d["action_sequence"]]
        return cls(action_sequence=seq, description=d.get("description"))


@dataclass
class ProposalSet:
    """A set of candidate proposals — the unit produced by samplers."""

    action_proposals: list[ActionProposal]

    def to_dict(self) -> dict:
        return {"action_proposals": [p.to_dict() for p in self.action_proposals]}

    @classmethod
    def from_dict(cls, d: dict) -> "ProposalSet":
        return cls([ActionProposal.from_dict(p) for p in d["action_proposals"]])

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        s = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            Path(path).write_text(s)
        return s

    @classmethod
    def from_json(cls, path_or_text: Union[str, Path]) -> "ProposalSet":
        p = Path(path_or_text)
        text = p.read_text() if p.exists() else str(path_or_text)
        return cls.from_dict(json.loads(text))

    def validate(
        self,
        allowed_types: Optional[set[str]] = None,
        ranges: Optional[dict[str, dict[str, tuple[float, float]]]] = None,
    ) -> list[str]:
        """Return a list of human-readable problems (empty list == valid).

        ``allowed_types``: restrict to these primitive types (e.g. a task only
        allows PUSH/LIFT/DESCEND/ROTATE). ``ranges``: ``{TYPE: {attr: (lo, hi)}}``
        bounds on numeric params (attribute names, e.g. GRASP uses ``width``).
        """
        errs: list[str] = []
        for pi, prop in enumerate(self.action_proposals):
            for ai, act in enumerate(prop.action_sequence):
                loc = f"proposal[{pi}].action[{ai}] {act.TYPE}"
                if allowed_types is not None and act.TYPE not in allowed_types:
                    errs.append(f"{loc}: type not allowed (allowed={sorted(allowed_types)})")
                if ranges and act.TYPE in ranges:
                    vals = act.attr_values()
                    for attr, (lo, hi) in ranges[act.TYPE].items():
                        if attr not in vals:
                            errs.append(f"{loc}: missing param {attr!r}")
                        elif not (lo <= vals[attr] <= hi):
                            errs.append(f"{loc}: {attr}={vals[attr]} out of [{lo}, {hi}]")
        return errs
