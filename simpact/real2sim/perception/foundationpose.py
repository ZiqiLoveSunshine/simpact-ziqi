"""FoundationPose 6-DoF object pose estimation backend.

Wraps the FoundationPose repo. Ported from the original ``real2sim/estimate_pose.py``; the call
sequence is the one verified in ``scripts/run_rigid_pipeline.py`` /
``scripts/smoke_rigid_coexist.py``.

Requires:
* repo: set ``SIMPACT_FOUNDATIONPOSE_DIR`` to a FoundationPose clone
* CUDA: uses ``nvdiffrast`` (RasterizeCudaContext) — needs a GPU
* the repo's bundled weights for ScorePredictor / PoseRefinePredictor
* the built native exts (``mycuda``, ``mycpp``) — see scripts/setup_rigid_env.sh

The model networks (scorer + refiner + rasterization context) are heavy and are
built **once**, lazily, on the first ``estimate()`` call, then reused. Construct
the adapter once and reuse it across objects/frames.
"""
from pathlib import Path

import numpy as np

from simpact.real2sim.perception.base import PoseEstimate, PoseEstimator
from simpact.real2sim.perception.repos import add_repo_to_syspath


class FoundationPoseEstimator(PoseEstimator):
    def __init__(self, est_refine_iter: int = 5, seed: int = 0):
        # resolves SIMPACT_FOUNDATIONPOSE_DIR or raises PerceptionRepoNotFound
        self.repo = add_repo_to_syspath("foundationpose")
        self.est_refine_iter = est_refine_iter
        self.seed = seed
        # heavy, mesh-independent singletons — built on first estimate()
        self._FoundationPose = None
        self._scorer = None
        self._refiner = None
        self._glctx = None
        # per-mesh estimator cache (keyed by mesh path)
        self._est = None
        self._est_key = None

    def _ensure_loaded(self):
        if self._scorer is not None:
            return
        # heavy imports kept out of module top-level so the package stays
        # importable without a GPU / the repo present.
        #
        # estimater.py uses relative imports (`from .Utils import *`), so it must
        # be imported as a *package* submodule (``<repo>.estimater``): put the
        # repo's parent on sys.path so the repo dir is an importable namespace
        # package, while add_repo_to_syspath already put the repo dir itself on
        # sys.path (for its absolute `import mycpp` / `from learning...`).
        import importlib
        import sys

        repo = Path(self.repo)
        if str(repo.parent) not in sys.path:
            sys.path.insert(0, str(repo.parent))
        est = importlib.import_module(f"{repo.name}.estimater")

        est.set_seed(self.seed)
        self._FoundationPose = est.FoundationPose
        self._scorer = est.ScorePredictor()
        self._refiner = est.PoseRefinePredictor()
        self._glctx = est.dr.RasterizeCudaContext()

    def _estimator_for(self, mesh_path: Path):
        """Return a FoundationPose bound to ``mesh_path`` (cached / reset_object)."""
        import trimesh

        key = str(mesh_path)
        if self._est_key == key:
            return self._est
        mesh = trimesh.load(mesh_path, force="mesh")
        if self._est is None:
            self._est = self._FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=self._scorer,
                refiner=self._refiner,
                glctx=self._glctx,
                debug=0,
            )
        else:
            # reuse the loaded networks; just rebind the geometry
            self._est.reset_object(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
            )
        self._est_key = key
        return self._est

    def estimate(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: np.ndarray,
        mesh_path: Path,
        object_name: str,
        camera_id: int = 0,
    ) -> PoseEstimate:
        self._ensure_loaded()

        depth = np.asarray(depth, dtype=np.float32).copy()
        depth[depth < 0.001] = 0  # FoundationPose treats <1mm as invalid
        mask = np.asarray(mask).astype(bool)
        if int(mask.sum()) < 10:
            raise ValueError(
                f"mask for {object_name!r} has {int(mask.sum())} valid pixels; "
                "FoundationPose needs >=10."
            )
        K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)

        est = self._estimator_for(Path(mesh_path))
        pose = est.register(
            K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=self.est_refine_iter
        )
        pose = np.asarray(pose).reshape(4, 4)  # object->camera
        return PoseEstimate(
            pose_cam=pose, object_name=object_name, camera_id=camera_id, score=None
        )

    def draw_pose(self, rgb, intrinsics, mesh_path, pose_cam, axis_scale=0.1,
                  thickness=2):
        """Overlay the estimated 6-DoF pose on an RGB image for validation.

        Draws FoundationPose's posed 3-D bounding box + XYZ axes (the same
        visualization as the original ``estimate_pose.py``). ``pose_cam`` is the
        object->camera 4x4 from ``estimate()``. Returns an RGB uint8 image.
        """
        import importlib

        import numpy as np
        import trimesh

        self._ensure_loaded()
        U = importlib.import_module(f"{Path(self.repo).name}.Utils")
        mesh = trimesh.load(mesh_path, force="mesh")
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        center = np.asarray(pose_cam).reshape(4, 4) @ np.linalg.inv(to_origin)
        vis = U.draw_posed_3d_box(K, img=np.ascontiguousarray(rgb), ob_in_cam=center, bbox=bbox)
        vis = U.draw_xyz_axis(vis, ob_in_cam=center, scale=axis_scale, K=K,
                              thickness=thickness, transparency=0, is_input_rgb=True)
        return vis
