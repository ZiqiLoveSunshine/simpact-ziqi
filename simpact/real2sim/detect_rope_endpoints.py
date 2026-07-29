"""Geometric rope endpoint detection (Phase 1 of the VLM endpoint-grounding plan).

Deterministic, no VLM: find the rope's two tips directly in the segmented 3-D cloud.
This replaces the former human ``pick_points_on_point_cloud`` Open3D shift-click (the
manual rope asset-prep) with a geometry pass; a later phase adds a VLM call to
assign which tip is *fixed* (anchored) vs *free* (graspable) — see
``docs/DEFORMABLE_INTEGRATION.md`` §13. This module only localizes the two ends
(sub-problem A); it does NOT decide roles.

Algorithm (robust for a thin curved rope):
  1. voxel-downsample the cloud to graph nodes;
  2. connect nodes within ``connect_radius`` (reusing ``connect_points``);
  3. keep the largest connected component (drop stray specks);
  4. the two tips are the geodesically **farthest-apart** node pair, found by
     double-Dijkstra (farthest node A from an arbitrary start, then farthest node B
     from A — exact for chain/tree-like graphs, which a rope is).
PCA principal-axis extremes are the fallback when the graph is too fragmented to be
a single curve.

The precision bar is lenient: the sim grasps a ``GRIP_RADIUS≈0.03`` (3 cm) blob
around each point, so a tip within ~2-3 cm suffices.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d  # noqa: F401  (import before any coacd; see CLAUDE.md)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from simpact.simulators.arap.pts_utils import connect_points

VOXEL = 0.005          # graph-node spacing (coarser than the rollout's 0.003 -> faster)
CONNECT_RADIUS = 0.013  # ~2.5x VOXEL: links chain neighbours without bridging a U-bend
MIN_COMPONENT_FRAC = 0.5  # largest component must hold this fraction of nodes to trust
TIP_RADIUS = 0.015     # snap each extreme to the centroid of full-res pts within this
                       # ball -> robust to a single stray speck; lands on the rope body


@dataclass
class EndpointResult:
    """Two rope tips (unordered — roles are assigned later) + detection diagnostics."""
    tip_a: np.ndarray          # (3,) 3-D coordinate of one tip
    tip_b: np.ndarray          # (3,) 3-D coordinate of the other tip
    confidence: float          # 0..1 heuristic (see _confidence)
    method: str                # "geodesic" | "pca_fallback"
    n_nodes: int               # graph nodes after downsample
    component_frac: float      # fraction of nodes in the largest component
    geodesic_len: float        # graph distance between the tips (curve length)
    euclidean_len: float       # straight-line distance between the tips

    @property
    def tips(self) -> np.ndarray:
        return np.vstack((self.tip_a, self.tip_b))


def _largest_component(nodes: np.ndarray, radius: float):
    """Return (component_nodes, csr_graph_of_component, frac_of_all_nodes)."""
    edges = connect_points(nodes, radius)  # (M,2) directed pairs, mode='distance'
    if len(edges) == 0:
        return nodes, None, 0.0
    w = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
    n = len(nodes)
    graph = csr_matrix((w, (edges[:, 0], edges[:, 1])), shape=(n, n))
    graph = graph.maximum(graph.T)  # symmetrize (radius graph is already ~symmetric)
    n_comp, labels = connected_components(graph, directed=False)
    if n_comp == 1:
        return nodes, graph, 1.0
    # keep the biggest component
    counts = np.bincount(labels)
    keep = int(counts.argmax())
    idx = np.where(labels == keep)[0]
    sub = graph[idx][:, idx]
    return nodes[idx], sub, float(len(idx)) / n


def _farthest(graph: csr_matrix, src: int):
    """Dijkstra from ``src``; return (farthest_reachable_node, distance_array)."""
    dist = dijkstra(graph, directed=False, indices=src)
    finite = np.where(np.isfinite(dist))[0]
    far = finite[int(np.argmax(dist[finite]))]
    return far, dist


def _confidence(component_frac: float, geodesic_len: float, euclidean_len: float,
                tip_a_isolated: bool, tip_b_isolated: bool) -> float:
    """Heuristic trust score. High when: one clean component, the pair spans a real
    curve (geodesic >= euclidean), and both tips sit at chain ends (low local degree)."""
    span = min(1.0, geodesic_len / 0.05)  # tips at least ~5 cm apart -> full credit
    curviness = 1.0 if euclidean_len < 1e-6 else min(1.0, geodesic_len / euclidean_len)
    ends = 0.5 * (float(tip_a_isolated) + float(tip_b_isolated))
    return float(np.clip(0.45 * component_frac + 0.2 * span
                         + 0.15 * curviness + 0.2 * ends, 0.0, 1.0))


def _pca_extremes(pts: np.ndarray):
    """Fallback: extremes along the principal axis (near-straight rope, broken graph)."""
    c = pts.mean(0)
    axis = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    t = (pts - c) @ axis
    return pts[int(t.argmin())].copy(), pts[int(t.argmax())].copy()


def _snap_to_body(tip: np.ndarray, full_pts: np.ndarray, radius: float) -> np.ndarray:
    """Centroid of full-res cloud points within ``radius`` of the extreme node.

    A rope tip is the geodesic extreme, but the outermost single node can be a stray
    speck; averaging its local neighbourhood on the dense cloud gives a stable
    on-rope point (~0.5-1 cm inward from the extreme). Falls back to the raw tip if
    the ball is empty."""
    near = full_pts[np.linalg.norm(full_pts - tip, axis=1) < radius]
    return near.mean(0) if len(near) else tip


def detect_rope_endpoints(points: np.ndarray, *, voxel: float = VOXEL,
                          connect_radius: float = CONNECT_RADIUS,
                          min_component_frac: float = MIN_COMPONENT_FRAC
                          ) -> EndpointResult:
    """Detect the two rope tips from a segmented point cloud (Nx3, robot frame)."""
    points = np.asarray(points, dtype=float)
    if len(points) < 4:
        raise ValueError(f"too few points for endpoint detection: {len(points)}")
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    nodes = np.asarray(pc.voxel_down_sample(voxel).points)
    if len(nodes) < 4:
        nodes = points

    comp_nodes, graph, frac = _largest_component(nodes, connect_radius)

    if graph is None or frac < min_component_frac:
        # graph too fragmented to be a single curve -> PCA extremes on all nodes
        a, b = _pca_extremes(nodes)
        a, b = _snap_to_body(a, points, TIP_RADIUS), _snap_to_body(b, points, TIP_RADIUS)
        eu = float(np.linalg.norm(a - b))
        return EndpointResult(a, b, _confidence(frac, eu, eu, False, False),
                              "pca_fallback", len(nodes), frac, eu, eu)

    # double-Dijkstra: farthest from an arbitrary node, then farthest from that
    a_idx, _ = _farthest(graph, 0)
    b_idx, dist_a = _farthest(graph, a_idx)
    geo = float(dist_a[b_idx])
    tip_a = _snap_to_body(comp_nodes[a_idx], points, TIP_RADIUS)
    tip_b = _snap_to_body(comp_nodes[b_idx], points, TIP_RADIUS)
    eu = float(np.linalg.norm(tip_a - tip_b))
    # a genuine chain end has few neighbours within the connect radius
    deg = np.asarray((graph > 0).sum(axis=1)).ravel()
    med_deg = float(np.median(deg)) if len(deg) else 0.0
    a_end = deg[a_idx] <= max(1.0, med_deg)
    b_end = deg[b_idx] <= max(1.0, med_deg)
    conf = _confidence(frac, geo, eu, a_end, b_end)
    return EndpointResult(tip_a, tip_b, conf, "geodesic",
                          len(nodes), frac, geo, eu)


def detect_from_ply(ply_path) -> EndpointResult:
    """Convenience: run detection on a ``segmented_object.ply`` file."""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    return detect_rope_endpoints(np.asarray(pcd.points))


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Detect rope tips in a segmented cloud.")
    ap.add_argument("--scene", required=True,
                    help="scene dir containing segmented_object.ply (+ scene.yaml)")
    ap.add_argument("--voxel", type=float, default=VOXEL)
    ap.add_argument("--connect-radius", type=float, default=CONNECT_RADIUS)
    args = ap.parse_args()
    scene = Path(args.scene)
    from simpact.utils.layout import find_scene_file
    r = detect_from_ply(find_scene_file(scene, "segmented_object.ply"))
    print(f"method={r.method}  confidence={r.confidence:.2f}  nodes={r.n_nodes}  "
          f"component_frac={r.component_frac:.2f}")
    print(f"  geodesic_len={r.geodesic_len*100:.1f}cm  euclidean_len={r.euclidean_len*100:.1f}cm")
    print(f"  tip_a = {np.round(r.tip_a, 4).tolist()}")
    print(f"  tip_b = {np.round(r.tip_b, 4).tolist()}")
    yml = find_scene_file(scene, "scene.yaml", required=False)
    if yml is not None:
        import yaml
        y = yaml.safe_load(yml.read_text())
        gt = np.array([y["fixed_point"], y["free_end"]], dtype=float)
        det = r.tips
        # match detected tips to ground-truth pair (unordered) by nearest
        err_direct = np.linalg.norm(det - gt, axis=1).mean()
        err_swap = np.linalg.norm(det[::-1] - gt, axis=1).mean()
        print(f"  vs scene.yaml (fixed_point/free_end): mean tip error = "
              f"{min(err_direct, err_swap)*100:.2f} cm")


if __name__ == "__main__":
    _main()
