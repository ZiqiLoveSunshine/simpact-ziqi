"""SAM-3D single-image-to-3D reconstruction backend (mesh path).

Wraps the ``sam-3d-objects`` repo. Ported from the original ``real2sim/run_imgto3d.py`` (which
used Hunyuan3D); the call sequence is the one verified in
``scripts/run_rigid_pipeline.py`` / ``scripts/smoke_sam3d.py``.

Requires:
* repo: set ``SIMPACT_SAM3D_DIR`` to a sam-3d-objects clone
* checkpoints: ``<repo>/checkpoints/hf`` (incl. ``slat_decoder_mesh.ckpt``, ~12 GB)
* deps incl. ``kaolin`` for the mesh path — see scripts/setup_rigid_env.sh
* a GPU (loads ~20 GB of weights)

Design notes (see docs/RIGID_ENV_SETUP.md):
* ``LIDRA_SKIP_INIT=true`` skips a Meta-internal ``init`` submodule absent from
  the public checkout; set before importing ``sam3d_objects``.
* The package is imported via ``sys.path`` (NOT pip-installed — its build hook
  would drag a torch-2.5/cu121 kitchen sink in).
* We drive the core pipeline directly, bypassing ``notebook/inference.py`` (which
  hard-imports seaborn/gradio/kaolin-viz and assumes a conda env).

The pipeline (~20 GB) is loaded **once**, lazily, on first ``reconstruct()``.
"""
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from simpact.real2sim.perception.base import ImageTo3DReconstructor, Reconstruction
from simpact.real2sim.perception.repos import add_repo_to_syspath


@contextmanager
def _chdir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class SAM3DReconstructor(ImageTo3DReconstructor):
    def __init__(self, seed: int = 42, checkpoint_tag: str = "hf"):
        os.environ.setdefault("LIDRA_SKIP_INIT", "true")  # before sam3d_objects import
        # resolves SIMPACT_SAM3D_DIR or raises PerceptionRepoNotFound
        self.repo = add_repo_to_syspath("sam3d")
        self.seed = seed
        self.checkpoint_tag = checkpoint_tag
        self._pipe = None

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        import sam3d_objects  # noqa: F401 — needed so hydra _target_ strings resolve
        from sam3d_objects.pipeline.inference_pipeline_pointmap import (  # noqa: F401
            InferencePipelinePointMap,
        )

        cfg_path = f"checkpoints/{self.checkpoint_tag}/pipeline.yaml"
        # ckpt paths in pipeline.yaml are relative to the repo root -> chdir while
        # instantiating, then restore cwd (don't mutate global cwd for the caller).
        with _chdir(self.repo):
            cfg = OmegaConf.load(cfg_path)
            cfg.rendering_engine = "pytorch3d"  # disable the nvdiffrast renderer
            cfg.compile_model = False
            cfg.workspace_dir = f"checkpoints/{self.checkpoint_tag}"
            self._pipe = instantiate(cfg)

    def reconstruct(self, image_path: Path, output_dir: Path) -> Reconstruction:
        """Reconstruct a complete mesh from a single **RGBA** crop.

        ``image_path`` must be an RGBA image with the object mask in the alpha
        channel (the convention produced by mask extraction / the original
        ``*_cropped.png``).
        """
        from PIL import Image

        self._ensure_loaded()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        img = np.asarray(Image.open(image_path))
        if img.ndim != 3 or img.shape[2] != 4:
            raise ValueError(
                f"{image_path} must be RGBA (object mask in alpha); got shape "
                f"{img.shape}. Produce it with mask extraction / *_cropped.png."
            )
        rgba = img.astype(np.uint8)

        out = self._pipe.run(
            rgba, None, self.seed,
            stage1_only=False, with_mesh_postprocess=False,
            with_texture_baking=False, with_layout_postprocess=False,
            use_vertex_color=True, stage1_inference_steps=None, pointmap=None,
        )
        mesh = out["glb"]  # trimesh.Trimesh with vertex colors
        mesh_path = output_dir / f"{Path(image_path).stem}.glb"
        mesh.export(mesh_path)
        n_verts = int(len(mesh.vertices)) if hasattr(mesh, "vertices") else None
        return Reconstruction(
            mesh_path=mesh_path,
            textured=True,  # vertex colors (not a UV texture image)
            metadata={"backend": "sam3d", "seed": self.seed, "vertices": n_verts},
        )
