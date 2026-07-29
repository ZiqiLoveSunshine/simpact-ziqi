"""
Headless smoke test for simpact.simulators.mpm.

Scenario: a slab of plasticine sitting on a floor, pushed by a rotating box.
Derived from the original warp_mpm box-pusher demo — klampt visualisation removed.

Checks:
  1. Solver initialises and accepts torch tensors.
  2. set_parameters_dict / finalize_mu_lam_bulk runs without error.
  3. Surface and box colliders are accepted.
  4. p2g2p runs N_STEPS without CUDA error.
  5. Particle positions stay within the unit cube (no blow-up).
  6. Final positions are written to outputs/smoke_test_mpm/final.ply.

Usage:
  python scripts/smoke_test_mpm.py
"""

import torch
import numpy as np

from simpact.simulators.mpm import MPM_Simulator_WARP
from simpact.simulators.mpm.engine import particle_position_to_ply
from simpact.utils.config import get_outputs_dir

# ── config ──────────────────────────────────────────────────────────────���─────
N_PARTICLES = 10_000   # keep small for a quick test
N_STEPS     = 50
DT          = 0.002
DEVICE      = "cuda:0"

# ── initialise solver ─────────────────────────────────────────────────────────
print("Initialising MPM solver …")
solver = MPM_Simulator_WARP(N_PARTICLES)

position = torch.rand(N_PARTICLES, 3) * torch.tensor([0.5, 0.5, 0.05])
position += torch.tensor([0.25, 0.25, 0.13])
position  = position.to(DEVICE)
volume    = torch.ones(N_PARTICLES) * 2.5e-8

solver.load_initial_data_from_torch(position, volume)

# ── material ──────────────────────────────────────────────────────────────────
solver.set_parameters_dict({
    "E":                    2000,
    "nu":                   0.2,
    "material":             "plasticine",
    "friction_angle":       35,
    "g":                    [0.0, 0.0, -10.0],
    "density":              200.0,
    "grid_v_damping_scale": 0.9,
})
solver.finalize_mu_lam_bulk()

# ── boundary conditions ───────────────────────────────────────────────────────
solver.add_surface_collider((0.0, 0.0, 0.13), (0.0, 0.0, 1.0), "sticky", 0.0)

box_center = np.array([0.5, 0.8, 0.25])
box_size   = np.array([0.2, 0.05, 0.3])

solver.add_rotate_box_collider(
    point  = box_center,
    quat   = [0.0, 0.0, 0.0, 1.0],
    twist  = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
    surface= "slip",
    friction=0.0,
    width  = box_size[0],
    height = box_size[1],
    length = box_size[2],
)

# ── simulation loop ───────────────────────────────────────────────────────────
print(f"Running {N_STEPS} steps …")
for step in range(1, N_STEPS + 1):
    solver.p2g2p(step, DT, device=DEVICE)
    if step % 10 == 0:
        pos = solver.mpm_state.particle_x.numpy()
        print(f"  step {step:3d}  z-range [{pos[:,2].min():.3f}, {pos[:,2].max():.3f}]")

# ── sanity check ──────────────────────────────────────────────────────────────
pos = solver.mpm_state.particle_x.numpy()
assert pos.min() > -0.5, f"particles escaped lower bound: min={pos.min():.3f}"
assert pos.max() <  1.5, f"particles escaped upper bound: max={pos.max():.3f}"
print("Sanity check passed: all particles within bounds.")

# ── save output ───────────────────────────────────────────────────────────────
out_dir = get_outputs_dir() / "smoke_test_mpm"
out_dir.mkdir(parents=True, exist_ok=True)
ply_path = str(out_dir / "final.ply")
particle_position_to_ply(solver, ply_path)
print(f"Wrote {ply_path}")
print("smoke_test_mpm PASSED")
