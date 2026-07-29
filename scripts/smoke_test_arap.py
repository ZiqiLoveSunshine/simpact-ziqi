"""Headless smoke test for the ARAP embedded deformation graph.

Mirrors the synthetic 5-point chain demo from the upstream
``embed_deform_graph.__main__`` block, but runs without any plotting/display so
it is safe in CI. Optimises a small deformation graph toward two handle targets
and asserts the energy decreases.

Run: python scripts/smoke_test_arap.py
"""
import sys

import numpy as np
import torch

from simpact.simulators.arap import EmbedDeformGraph

if EmbedDeformGraph is None:
    print("smoke_test_arap SKIPPED: arap optional deps not installed "
          "(pip install -e \".[arap]\")")
    sys.exit(0)


def main():
    # The constructor hardcodes NearestNeighbors(n_neighbors=10), so the graph
    # needs >= 10 points. Use a 12-point chain laid out along x at z = 0.
    n = 12
    print(f"Building EmbedDeformGraph ({n}-point chain) …")
    # constructor expects numpy arrays (it calls np.max on rest_pts internally)
    xs = np.linspace(0.0, 1.1, n)
    rest_pts = np.stack([xs, np.zeros(n), np.zeros(n)], axis=1)
    # edges connect consecutive points, both directions
    fwd = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    edges = np.concatenate([fwd, fwd[:, ::-1]], axis=0)

    # vis_pts is required by the current constructor (used for RBF visualization
    # weights); the graph's own rest points are the natural choice for a smoke test.
    ng = EmbedDeformGraph(rest_pts, edges, corotate=True, vis_pts=rest_pts)

    # pull the two endpoints upward in z
    handle_idx = torch.tensor([0, n - 1], dtype=torch.long)
    handle_pts = torch.tensor(
        [[0.0, 0.0, 0.3], [1.1, 0.0, 0.3]], dtype=torch.double
    )

    optimizer = torch.optim.Adam(ng.deform_state.parameters(), lr=0.1)

    print("Optimising 100 steps …")
    energies = []
    for step in range(100):
        energy = ng.energy(handle_idx, handle_pts)
        optimizer.zero_grad()
        energy.backward()
        optimizer.step()
        energies.append(energy.item())
        if (step + 1) % 20 == 0:
            print(f"  step {step + 1:3d}  energy {energy.item():.6f}")

    first, last = energies[0], energies[-1]
    print(f"energy: {first:.6f} -> {last:.6f}")
    assert last < first, f"energy did not decrease ({first} -> {last})"
    print("smoke_test_arap PASSED")


if __name__ == "__main__":
    main()
