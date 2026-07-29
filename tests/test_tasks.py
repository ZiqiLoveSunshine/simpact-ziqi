"""The task registry (simpact/tasks.py) is complete and internally consistent.

For every registered task: its three prompt templates load, its allowed
primitives exist in the action schema, and its bundled example trial (when
committed) resolves an initial EE pose. Rope's rollout factory is additionally
constructed for real (CPU-safe). This replaces per-driver template-presence
assertions — the registry IS the contract now.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from simpact.actions import PRIMITIVE_TYPES
from simpact.tasks import TASKS


def _cli_defaults(spec):
    """An argparse-like namespace carrying the driver's common args + the spec's
    per-task defaults — what the factories would receive from scripts/optimize.py."""
    ns = SimpleNamespace(cam=1, instruction=spec.default_instruction,
                         object=spec.default_object, max_iters=1)
    for flag, kw in spec.cli:
        setattr(ns, flag.lstrip("-").replace("-", "_"), kw.get("default"))
    return ns


def test_registry_covers_the_four_release_tasks():
    assert set(TASKS) == {"push", "rope", "dough", "sweep"}
    for key, spec in TASKS.items():
        assert spec.name == key


@pytest.mark.parametrize("spec", TASKS.values(), ids=lambda s: s.name)
def test_templates_load(spec):
    from simpact.generator.propose import load_proposal_template
    from simpact.generator.regress import load_regress_template
    from simpact.generator.templates import load_context_template
    from simpact.generator.verify import load_verify_template
    assert load_context_template(spec.context_template)
    assert load_regress_template(spec.regress_template)
    assert load_verify_template(spec.verify_template)
    assert load_proposal_template("primitive")  # shared proposer template


@pytest.mark.parametrize("spec", TASKS.values(), ids=lambda s: s.name)
def test_allowed_prims_exist_in_schema(spec):
    assert spec.allowed_prims <= set(PRIMITIVE_TYPES)
    assert "PUSH" in spec.allowed_prims  # every task can push


@pytest.mark.parametrize("spec", TASKS.values(), ids=lambda s: s.name)
def test_example_trial_resolves_initial_ee(spec):
    from simpact.generator.context import resolve_initial_ee
    trial = Path(spec.example_trial)
    if not trial.exists():
        pytest.skip(f"missing {trial}")
    ee, src = resolve_initial_ee(trial)
    assert src.endswith("scene.yaml")  # the runtime source, not a fallback


def test_rope_rollout_constructs_from_registry():
    spec = TASKS["rope"]
    if not os.path.exists(f"{spec.example_trial}/sim/segmented_object.ply"):
        pytest.skip("missing rope example")
    roll = spec.make_rollout(Path(spec.example_trial), _cli_defaults(spec))
    assert hasattr(roll, "run")


def test_gate_factories():
    """sweep always gates; push gates only for exactly two objects with an axis."""
    sweep = TASKS["sweep"]
    if not os.path.exists(f"{sweep.example_trial}/sim/target_region.ply"):
        pytest.skip("missing sweep example")
    gate = sweep.make_gate(Path(sweep.example_trial), _cli_defaults(sweep), roll=None)
    assert callable(gate)

    push = TASKS["push"]
    a = _cli_defaults(push)
    fake = SimpleNamespace(object_names=["a", "b"])
    assert callable(push.make_gate(Path("."), a, fake))
    a.align_axis = "none"
    assert push.make_gate(Path("."), a, fake) is None
    a.align_axis = "y"
    assert push.make_gate(Path("."), a, SimpleNamespace(object_names=["solo"])) is None
