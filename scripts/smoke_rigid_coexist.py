"""Phase 0 — single-env runtime coexistence smoke.

Proves SAM-3D (image->3D, mesh) and FoundationPose (6-DoF pose) load their
WEIGHTED models and run inference in ONE process / ONE uv env on
torch 2.9+cu128 / numpy 2.x / RTX 5090. The build spike only showed compilation;
this shows both networks resident and running together, and logs peak GPU memory.

Run:  .venv/bin/python scripts/smoke_rigid_coexist.py
(env: SIMPACT_SAM3D_DIR, SIMPACT_FOUNDATIONPOSE_DIR optional; defaults below)
"""
import os, time
from pathlib import Path
os.environ["LIDRA_SKIP_INIT"] = "true"

import sys
import numpy as np
import torch
from PIL import Image

REPO = str(Path(__file__).resolve().parent.parent)
SAM3D = os.environ.get("SIMPACT_SAM3D_DIR", f"{REPO}/external/sam-3d-objects")
FP = os.environ.get("SIMPACT_FOUNDATIONPOSE_DIR", f"{REPO}/external/FoundationPose")


def gb():
    return torch.cuda.max_memory_allocated() / 1e9


def main():
    print(f"torch {torch.__version__} | numpy {np.__version__} | "
          f"{torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()

    # ---- SAM-3D: load + one reconstruction (mesh) ---------------------------
    sys.path.insert(0, SAM3D)
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    import sam3d_objects  # noqa
    from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap  # noqa
    cwd = os.getcwd(); os.chdir(SAM3D)
    t = time.time()
    cfg = OmegaConf.load("checkpoints/hf/pipeline.yaml")
    cfg.rendering_engine = "pytorch3d"; cfg.compile_model = False
    cfg.workspace_dir = "checkpoints/hf"
    pipe: InferencePipelinePointMap = instantiate(cfg)
    print(f"[sam3d] loaded in {time.time()-t:.1f}s; GPU peak {gb():.2f} GB")

    img = np.array(Image.open("notebook/images/kid_box/image.png"))[..., :3].astype(np.uint8)
    m = np.array(Image.open("notebook/images/kid_box/14.png"))
    m = m[..., 0] if m.ndim == 3 else m
    rgba = np.concatenate([img, ((m > 0).astype(np.uint8) * 255)[..., None]], -1)
    t = time.time()
    out = pipe.run(rgba, None, 42, stage1_only=False, with_mesh_postprocess=False,
                   with_texture_baking=False, with_layout_postprocess=False,
                   use_vertex_color=True, stage1_inference_steps=None, pointmap=None)
    glb = out.get("glb", None)
    nv = len(glb.vertices) if glb is not None and hasattr(glb, "vertices") else None
    print(f"[sam3d] reconstruct {time.time()-t:.1f}s; mesh verts={nv}; GPU peak {gb():.2f} GB")
    os.chdir(cwd)

    # ---- FoundationPose: load + one register (golden mesh + recorded observation)-
    sys.path.insert(0, FP)
    sys.path.insert(0, REPO + "/external")
    from FoundationPose.estimater import (FoundationPose, ScorePredictor,
                                          PoseRefinePredictor, dr, set_seed)
    import trimesh
    set_seed(0)
    # golden reconstructed assets (mesh + mask + RGB-D) from a trial dir — supply your own via
    # env (not bundled). No baked default.
    d = os.environ.get("SIMPACT_RIGID_SMOKE_DIR")
    if not d:
        print("[skip] set SIMPACT_RIGID_SMOKE_DIR to a trial dir with {obj}_scaled.obj + "
              "camera1_rgb/depth.npy + camera1_mask_{obj}.npy + {obj}_6d_cam1.txt")
        return
    obj = os.environ.get("SIMPACT_RIGID_SMOKE_OBJ", "pringles")
    mesh = trimesh.load(f"{d}/{obj}_scaled.obj")
    color = np.load(f"{d}/camera1_rgb.npy")
    depth = np.load(f"{d}/camera1_depth.npy"); depth[(depth < 0.001)] = 0
    K = np.loadtxt(os.environ.get("SIMPACT_SMOKE_K",
                                  f"{REPO}/assets/calibration/0103/cam1_K.txt")).reshape(3, 3)
    mask = np.load(f"{d}/camera1_mask_{obj}.npy").astype(bool)
    t = time.time()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                         glctx=dr.RasterizeCudaContext())
    print(f"[fp] loaded in {time.time()-t:.1f}s; GPU peak {gb():.2f} GB")
    t = time.time()
    pose = np.asarray(est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5)).reshape(4, 4)
    gold = np.loadtxt(f"{d}/{obj}_6d_cam1.txt")
    terr = np.linalg.norm(pose[:3, 3] - gold[:3, 3]) * 1000
    print(f"[fp] register {time.time()-t:.1f}s; trans err vs golden = {terr:.2f} mm")

    print(f"\n[coexist] BOTH models resident + ran in one process. "
          f"peak GPU mem = {gb():.2f} GB / 32 GB")
    print("RIGID_COEXIST_OK")


if __name__ == "__main__":
    main()
