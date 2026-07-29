"""Turn an action proposal into a gripper waypoint trajectory.

Sim-side ported from the original ``executor/parse_waypoints.py::WaypointParserPrimitive``
(the hardware ``FrankyParser`` is intentionally left out — no ``franky`` import
here). Each primitive accumulates onto the current EE pose / gripper width:

  PUSH -> x,y ; LIFT -> +z ; DESCEND -> -|z| ; ROTATE -> yaw ; ROLL -> roll ;
  FLICK -> x,y,z ; GRASP -> width ; RELEASE -> open

The optimizer-output plan actions are also accepted (the original ``WaypointParser``), so a
refined plan can be rolled out the same way as a proposal:

  MOVE -> x,y,z + roll,pitch,yaw deltas ; GRIPPER_CONTROL -> width

Reads ``Grasp.width`` from the typed schema (so the LLM's ``grasp_width`` is
honored — the original pipeline's executor read ``'width'`` and silently dropped it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from simpact.actions.primitives import (
    ActionProposal,
    Descend,
    Flick,
    Grasp,
    GripperControl,
    Lift,
    Move,
    Primitive,
    Push,
    Release,
    Roll,
    Rotate,
    primitive_from_dict,
)

RELEASE_WIDTH = 1.0  # convention: RELEASE opens fully (clamped downstream)


@dataclass
class Waypoint:
    position: list  # [x, y, z]
    orientation: list  # [w, x, y, z]
    gripper_width: float
    duration: float

    def to_dict(self) -> dict:
        return {
            "position": list(self.position),
            "orientation": list(self.orientation),
            "gripper_width": self.gripper_width,
            "duration": self.duration,
        }


def _apply_rotation_delta(quat_wxyz, droll=0.0, dpitch=0.0, dyaw=0.0) -> list:
    cur = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # ->xyzw
    new = cur * R.from_euler("xyz", [droll, dpitch, dyaw])
    x, y, z, w = new.as_quat()
    return [float(w), float(x), float(y), float(z)]


def proposal_to_waypoints(
    proposal: Union[ActionProposal, Sequence],
    initial_position,
    initial_orientation=(0.0, 1.0, 0.0, 0.0),
    *,
    xyz_bound: Optional[tuple] = None,
    z_min: float = 0.0,
    default_duration: float = 2.0,
    initial_gripper: float = 0.0,
) -> list[Waypoint]:
    """Accumulate a primitive sequence into absolute gripper waypoints.

    ``initial_orientation`` is wxyz (default = gripper pointing down). ``z_min``
    and ``xyz_bound`` are workspace safety limits (frame-dependent; for the
    rigid scene the table top is z=0, so z_min defaults to 0).
    """
    seq = proposal.action_sequence if isinstance(proposal, ActionProposal) else list(proposal)
    actions = [a if isinstance(a, Primitive) else primitive_from_dict(a) for a in seq]

    pos = [float(v) for v in initial_position]
    orient = [float(v) for v in initial_orientation]
    grip = float(initial_gripper)

    waypoints = [Waypoint(pos.copy(), orient.copy(), grip, min(2.0, default_duration))]
    for a in actions:
        if isinstance(a, Grasp):
            grip = a.width
        elif isinstance(a, Release):
            grip = RELEASE_WIDTH
        elif isinstance(a, Push):
            pos[0] += a.delta_x; pos[1] += a.delta_y
        elif isinstance(a, Lift):
            pos[2] += a.delta_z
        elif isinstance(a, Descend):
            pos[2] -= abs(a.delta_z)
        elif isinstance(a, Rotate):
            orient = _apply_rotation_delta(orient, dyaw=a.delta_yaw)
        elif isinstance(a, Roll):
            orient = _apply_rotation_delta(orient, droll=a.delta_roll)
        elif isinstance(a, Flick):
            pos[0] += a.delta_x; pos[1] += a.delta_y; pos[2] += a.delta_z
        elif isinstance(a, Move):  # optimizer-output plan action (legacy WaypointParser format)
            pos[0] += a.delta_x; pos[1] += a.delta_y; pos[2] += a.delta_z
            if a.delta_roll or a.delta_pitch or a.delta_yaw:
                orient = _apply_rotation_delta(orient, a.delta_roll, a.delta_pitch, a.delta_yaw)
        elif isinstance(a, GripperControl):  # optimizer-output plan action
            grip = a.width

        if xyz_bound is not None:
            lo, hi = np.asarray(xyz_bound[0]), np.asarray(xyz_bound[1])
            pos = [float(v) for v in np.clip(pos, lo, hi)]
        if pos[2] < z_min:
            pos[2] = z_min
        waypoints.append(Waypoint(pos.copy(), orient.copy(), grip, default_duration))
    return waypoints
