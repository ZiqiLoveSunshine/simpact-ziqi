"""The task registry — single source of truth for the four optimization tasks.

Everything that differs between the push / rope / dough / sweep closed loops is
data or a small factory collected in a :class:`TaskSpec`:

* which prompt templates drive context / regress / verify,
* which primitive types the proposer may emit,
* how to construct the rollout ("RolloutLike": anything with
  ``run(plan, index, out_dir) -> path`` — ``PushSceneRollout``, ``ARAPRollout``,
  ``MPMRollout``, ``SweepRollout``),
* how to parse a rollout JSON for the regressor,
* the optional measured success gate,
* per-task CLI extras and the bundled example trial.

Consumed by ``scripts/optimize.py`` (the unified closed-loop driver) and
``scripts/propose_actions.py`` (task -> context template + allowed primitives).
Adding a task = one ``TaskSpec`` here + its three prompt templates + a bundled
example trial; no new scripts.

The allowed-primitive sets are the ones the recorded example runs used (the
former per-demo constants); they supersede the looser, partly stale
``TASK_PROFILES`` copies that lived in ``propose_actions.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from simpact.generator.regress import (
    parse_mpm_rollout,
    parse_rigid_rollout,
    parse_rope_rollout,
)

# (flag, argparse kwargs) — appended to the unified driver's parser per task.
CliArg = tuple[str, dict]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    context_template: str
    regress_template: str
    verify_template: str
    allowed_prims: frozenset[str]
    make_rollout: Callable[[Path, Any], Any]          # (scene_dir, args) -> RolloutLike
    parse_rollout: Callable[..., dict]                # rollout JSON -> regress text/dict
    describe_rollout: Callable[[dict], str]           # rollout JSON dict -> one log line
    make_gate: Optional[Callable[[Path, Any, Any], Any]] = None  # (scene, args, roll) -> gate|None
    cli: tuple[CliArg, ...] = ()
    default_object: Optional[str] = None              # None -> derive from roll.object_names
    default_instruction: str = ""
    example_trial: str = ""
    # -- real2sim (capture/ -> sim/) via simpact.real2sim.build_scene; None for push,
    #    whose scene comes from the rigid perception pipeline instead.
    build_material: Optional[str] = None              # build_scene --material
    build_object_prompt: Optional[str] = None         # segmentation prompt (--object)
    build_bg_prompt: Optional[str] = None             # sweep target region (--bg)


# ---- rollout factories (lazy imports keep CPU-only environments working) ----

def _push_rollout(scene: Path, a) -> Any:
    from simpact.executor.push_scene import PushSceneRollout
    return PushSceneRollout(scene, cam=a.cam, view=a.view, instruction=a.instruction,
                            objects=a.objects, init_ee_file=a.init_ee_file,
                            initial_ee=a.initial_ee)


def _rope_rollout(scene: Path, a) -> Any:
    from simpact.executor.rope_rollout import ARAPRollout
    return ARAPRollout(scene)


def _dough_rollout(scene: Path, a) -> Any:
    from simpact.executor.mpm_rollout import MPMRollout
    return MPMRollout(scene, num_steps=a.num_steps)


def _sweep_rollout(scene: Path, a) -> Any:
    from simpact.executor.mpm_rollout import SweepRollout
    return SweepRollout(scene, num_steps=a.num_steps)


# ---- measured success gates -------------------------------------------------

def _push_gate(scene: Path, a, roll) -> Any:
    """Alignment of the two objects on the chosen axis (measured), or None."""
    if a.align_axis == "none" or len(roll.object_names) != 2:
        return None
    from simpact.generator.verify import alignment_gate
    print(f"success gate: '{roll.object_names[0]}' & '{roll.object_names[1]}' aligned "
          f"on {a.align_axis} within {a.align_tol * 100:.0f} cm (measured)")
    return alignment_gate(roll.object_names[0], roll.object_names[1],
                          axis=a.align_axis, tol_m=a.align_tol)


def _sweep_gate(scene: Path, a, roll) -> Any:
    from simpact.generator.verify import coverage_gate
    from simpact.utils.layout import find_scene_file
    return coverage_gate(find_scene_file(scene, "target_region.ply"),
                         min_coverage=a.min_coverage)


# ---- per-rollout log lines --------------------------------------------------

def _describe_push(d: dict) -> str:
    snaps = d["snapshots"]
    moved = {n: round(float(np.linalg.norm(
        np.asarray(snaps[-1]["objects"][n]["position"])
        - np.asarray(snaps[0]["objects"][n]["position"]))) * 100, 1)
        for n in snaps[0]["objects"]}
    return f"object motion (cm) = {moved}"


def _describe_rope(d: dict) -> str:
    g, p = d["grasp_point"], d["place_point"]
    return f"grasp {[round(v, 3) for v in g]} -> place {[round(v, 3) for v in p]}"


def _describe_dough(d: dict) -> str:
    m = d["mpm"]
    return f"{m['num_grasps']} squeeze(s), bbox {[round(v, 3) for v in m['bbox_size']]}"


def _describe_sweep(d: dict) -> str:
    m = d["mpm"]
    return f"{m['num_segments']} segment(s), centroid {[round(v, 3) for v in m['centroid']]}"


# ---- the registry -----------------------------------------------------------

TASKS: dict[str, TaskSpec] = {
    "push": TaskSpec(
        name="push",
        context_template="push",
        regress_template="push",
        verify_template="push",
        allowed_prims=frozenset({"PUSH", "LIFT", "DESCEND", "ROTATE"}),
        make_rollout=_push_rollout,
        parse_rollout=parse_rigid_rollout,
        describe_rollout=_describe_push,
        make_gate=_push_gate,
        cli=(
            ("--view", dict(default="top_view",
                            choices=["top_view", "side_view", "front_view", "top", "front"],
                            help="rollout render camera")),
            ("--align_axis", dict(default="y", choices=["x", "y", "z", "none"],
                                  help="success gate: objects aligned on this axis (none=off)")),
            ("--align_tol", dict(type=float, default=0.02,
                                 help="alignment tolerance in metres")),
            ("--objects", dict(default=None, help='names "a. b."; default: all in the trial')),
            ("--initial_ee", dict(default=None, help="override gripper start mocap xyz")),
            ("--init_ee_file", dict(default=None,
                                    help="explicit EE-pose file (context.txt or 4x4 matrix)")),
        ),
        default_object=None,  # derived from the trial's reconstructed objects
        default_instruction=("Push the white carton so it lines up side by side with the "
                             "blue carton (both at the same depth), without moving the "
                             "blue carton."),
        example_trial="examples/push_real2sim/0103_push_0",  # sim/ carries the golden reconstruction
        # push's scene comes from the rigid perception pipeline (run_rigid_pipeline.py),
        # not build_scene — build_object_prompt is that pipeline's segmentation prompt.
        build_object_prompt="white coconut milk carton. blue milk carton.",
    ),
    "rope": TaskSpec(
        name="rope",
        context_template="rope",
        regress_template="rope",
        verify_template="rope",
        allowed_prims=frozenset({"PUSH", "DESCEND", "GRASP", "RELEASE"}),
        make_rollout=_rope_rollout,
        parse_rollout=parse_rope_rollout,
        describe_rollout=_describe_rope,
        default_object="rope.",
        default_instruction=("Arrange the rope into a U-shaped curve by dragging its free "
                             "end. Only the free end of the rope may be grasped; the other "
                             "end is fixed."),
        example_trial="examples/rope_real2sim/1102_rope_11",
        build_material="rope",
        build_object_prompt="rope",
    ),
    "dough": TaskSpec(
        name="dough",
        context_template="dough",
        regress_template="dough",
        verify_template="dough",
        # RELEASE is allowed (the context template lists it) but is a sim no-op: the
        # plastic deformation from the squeeze persists after the jaws reopen.
        allowed_prims=frozenset({"PUSH", "ROTATE", "DESCEND", "GRASP", "RELEASE"}),
        make_rollout=_dough_rollout,
        parse_rollout=parse_mpm_rollout,
        describe_rollout=_describe_dough,
        cli=(("--num_steps", dict(type=int, default=200, help="MPM steps per squeeze")),),
        default_object="blue playdoh.",
        default_instruction=("Shape the dough into a square block (roughly equal width and "
                             "depth) by squeezing it from two perpendicular directions."),
        example_trial="examples/dough_real2sim/1104_sand_6",
        build_material="dough",
        build_object_prompt="blue playdoh",
    ),
    "sweep": TaskSpec(
        name="sweep",
        context_template="sweep",
        regress_template="sweep",
        verify_template="sweep",
        allowed_prims=frozenset({"PUSH", "ROTATE", "DESCEND"}),
        make_rollout=_sweep_rollout,
        parse_rollout=parse_mpm_rollout,
        describe_rollout=_describe_sweep,
        make_gate=_sweep_gate,
        cli=(
            ("--num_steps", dict(type=int, default=120, help="MPM steps per segment")),
            ("--min_coverage", dict(type=float, default=0.5,
                                    help="measured gate: fraction of material inside the target")),
        ),
        default_object="black bean pile.",
        default_instruction=("Sweep all the beans together into the taped target region, "
                             "keeping the pusher's flat face square to the push so the "
                             "whole pile moves as one and no beans are left behind."),
        example_trial="examples/sweep_real2sim/0118_sweep_0",
        build_material="sweep",
        build_object_prompt="black bean pile",
        build_bg_prompt="taped target region",
    ),
}
