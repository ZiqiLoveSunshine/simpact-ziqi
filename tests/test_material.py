"""Tests for VLM material-ID (estimate/clamp/resolve), fully mocked — no API/GPU."""
import json

import numpy as np
import pytest
import yaml

from simpact.generator.material import (
    PHYS_KEYS,
    SOLVER_CONFIG,
    estimate_material,
    load_bands,
    load_material,
    write_material,
)

IMG = "examples/dough_real2sim/1104_sand_6/capture/camera1_rgb.png"


def _fake(E=8000, nu=0.42, yield_stress=1500, density=1200, softness="soft"):
    def gen(contents):
        return json.dumps({"softness": softness, "E": E, "nu": nu,
                           "yield_stress": yield_stress, "density": density, "confidence": 0.8})
    return gen


def test_estimate_returns_physics_only():
    m = estimate_material(IMG, "blue playdoh", "dough", generate_fn=_fake())
    assert set(m) == set(PHYS_KEYS) | {"softness", "confidence", "source"}
    assert m["source"] == "vlm"
    assert "material" not in m and "g" not in m  # solver config is NOT estimated


def test_estimate_clamps_out_of_band():
    bands = load_bands()
    e_hi = max(v[1] for v in bands["E"].values())
    m = estimate_material(IMG, "x", "dough", generate_fn=_fake(E=10_000_000))  # absurd
    assert m["E"] == e_hi                      # clamped to the band max
    lo = min(v[0] for v in bands["E"].values())
    assert estimate_material(IMG, "x", "dough", generate_fn=_fake(E=1))["E"] == lo


def test_load_material_merges_solver_and_physics(tmp_path):
    (tmp_path / "scene.yaml").write_text(yaml.dump(
        {"material": {"E": 7000, "nu": 0.4, "yield_stress": 900, "density": 1100,
                      "source": "vlm"}}))
    p = load_material(tmp_path, "dough")
    assert p["E"] == 7000 and p["yield_stress"] == 900          # VLM physics
    assert p["material"] == "plasticine" and p["friction_angle"] == 35.0  # solver default
    assert p["g"] == SOLVER_CONFIG["dough"]["g"]
    # sweep class -> its own solver config (friction_angle 0)
    (tmp_path / "scene.yaml").write_text(yaml.dump(
        {"material": {"E": 4000, "nu": 0.25, "yield_stress": 400, "density": 700}}))
    assert load_material(tmp_path, "sweep")["friction_angle"] == 0.0


def test_load_material_errors_without_block(tmp_path):
    (tmp_path / "scene.yaml").write_text("object_name: x\n")
    with pytest.raises(KeyError, match="no complete 'material:' block"):
        load_material(tmp_path, "dough")


def test_write_material_roundtrip(tmp_path):
    (tmp_path / "scene.yaml").write_text("object_name: blue playdoh\n")
    m = estimate_material(IMG, "blue playdoh", "dough", generate_fn=_fake())
    write_material(tmp_path, m)
    y = yaml.safe_load((tmp_path / "scene.yaml").read_text())
    assert y["material"]["source"] == "vlm" and y["object_name"] == "blue playdoh"
    assert load_material(tmp_path, "dough")["E"] == m["E"]


@pytest.mark.parametrize("scene,cls", [
    ("examples/dough_real2sim/1104_sand_6", "dough"),
    ("examples/sweep_real2sim/0118_sweep_0", "sweep"),
])
def test_example_scenes_carry_vlm_material(scene, cls):
    import os
    if not os.path.exists(f"{scene}/sim/scene.yaml"):
        pytest.skip(f"missing {scene}")
    y = yaml.safe_load(open(f"{scene}/sim/scene.yaml"))
    assert "material" in y and y["material"]["source"] == "vlm"      # VLM-estimated, not manual
    bands = load_bands()
    for k in PHYS_KEYS:
        lo = min(v[0] for v in bands[k].values()); hi = max(v[1] for v in bands[k].values())
        assert lo <= y["material"][k] <= hi, f"{scene} {k} out of band"
    # resolves to a full solver-ready dict
    p = load_material(scene, cls)
    assert set(("E", "nu", "yield_stress", "density", "material", "friction_angle")) <= set(p)
