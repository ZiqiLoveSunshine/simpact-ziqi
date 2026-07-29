"""SAM-3D mesh smoke — load the pipeline and reconstruct one mesh from the repo's
kid_box sample, in simpact's uv .venv (torch 2.9+cu128 / numpy 2.x / RTX 5090).

Drives the CORE pipeline directly (no notebook wrapper -> no seaborn/gradio/conda
deps). See docs/RIGID_ENV_SETUP.md.

Run:  .venv/bin/python scripts/smoke_sam3d.py
"""
import os, time
os.environ["LIDRA_SKIP_INIT"] = "true"

import sys
import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM3D = os.environ.get("SIMPACT_SAM3D_DIR", f"{REPO}/external/sam-3d-objects")
sys.path.insert(0, SAM3D)
os.chdir(SAM3D)  # pipeline.yaml ckpt paths are relative to the repo root

from omegaconf import OmegaConf
from hydra.utils import instantiate
import sam3d_objects  # noqa: needed so hydra _target_ strings resolve
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap  # noqa

t0 = time.time()
cfg = OmegaConf.load("checkpoints/hf/pipeline.yaml")
cfg.rendering_engine = "pytorch3d"   # disable nvdiffrast renderer
cfg.compile_model = False
cfg.workspace_dir = "checkpoints/hf"
pipe: InferencePipelinePointMap = instantiate(cfg)
print(f"[load] pipeline ready in {time.time() - t0:.1f}s")

img = np.array(Image.open("notebook/images/kid_box/image.png"))[..., :3].astype(np.uint8)
m = np.array(Image.open("notebook/images/kid_box/14.png"))
m = m[..., 0] if m.ndim == 3 else m
rgba = np.concatenate([img, ((m > 0).astype(np.uint8) * 255)[..., None]], axis=-1)
print(f"[input] image {img.shape} mask {m.shape} -> rgba {rgba.shape}")

torch.cuda.reset_peak_memory_stats()
t1 = time.time()
out = pipe.run(rgba, None, 42, stage1_only=False, with_mesh_postprocess=False,
               with_texture_baking=False, with_layout_postprocess=False,
               use_vertex_color=True, stage1_inference_steps=None, pointmap=None)
print(f"[run] inference {time.time() - t1:.1f}s; keys={list(out.keys())}")

glb = out.get("glb", None)
if glb is not None:
    glb.export("/tmp/sam3d_kid_box.glb")
    nv = len(glb.vertices) if hasattr(glb, "vertices") else None
    nf = len(glb.faces) if hasattr(glb, "faces") else None
    print(f"[mesh] exported /tmp/sam3d_kid_box.glb  verts={nv} faces={nf}")
else:
    print("[mesh] no 'glb' key; available:", list(out.keys()))
print(f"[gpu] peak CUDA mem = {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print("SAM3D_SMOKE_OK")
