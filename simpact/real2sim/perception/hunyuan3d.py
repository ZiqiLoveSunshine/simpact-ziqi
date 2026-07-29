"""Hunyuan3D-2.1 single-image-to-3D reconstruction backend.

Wraps the Hunyuan3D-2.1 repo. Port target: the original ``real2sim/run_imgto3d.py``.

Requires:
* repo: set ``SIMPACT_HUNYUAN3D_DIR`` to a Hunyuan3D-2.1 clone
* model: ``tencent/Hunyuan3D-2.1`` (set ``HY3DGEN_MODELS`` to ``<repo>/ckpt``)
* texture: ``<repo>/hy3dpaint/ckpt/RealESRGAN_x4plus.pth`` +
  ``<repo>/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml``
* pip deps: ``hy3dshape``/``hy3dpaint`` (from repo), ``bpy`` (Blender), ``pymeshlab``
"""
import os
from pathlib import Path

from simpact.real2sim.perception.base import ImageTo3DReconstructor, Reconstruction
from simpact.real2sim.perception.repos import add_repo_to_syspath

DEFAULT_MODEL_ID = "tencent/Hunyuan3D-2.1"


class Hunyuan3DReconstructor(ImageTo3DReconstructor):
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, with_texture: bool = True):
        # resolves SIMPACT_HUNYUAN3D_DIR or raises PerceptionRepoNotFound
        self.repo = add_repo_to_syspath("hunyuan3d", "hy3dshape", "hy3dpaint")
        os.environ.setdefault("HY3DGEN_MODELS", str(self.repo / "ckpt"))
        self.model_id = model_id
        self.with_texture = with_texture

    def reconstruct(self, image_path: Path, output_dir: Path) -> Reconstruction:
        raise NotImplementedError(
            "Hunyuan3DReconstructor.reconstruct is pending port "
            "real2sim/run_imgto3d.py (Hunyuan3DDiTFlowMatchingPipeline shape gen "
            "+ Hunyuan3DPaintPipeline texture -> textured GLB)."
        )
