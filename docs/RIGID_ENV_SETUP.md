# Rigid perception env — Grounded-SAM-2 + SAM-3D + FoundationPose in one uv env

This is the **install guide** for running the rigid real2sim perception models —
**Grounded-SAM-2** (open-vocab segmentation from a language prompt), **SAM-3D**
(image→3D mesh) and **FoundationPose** (6-DoF pose) — together in a single
`simpact` uv environment, in-process, on `torch 2.9+cu128 / py3.11`.

Verified on an RTX 5090 (Blackwell, `sm_120`): all three load and run in one
Python process, ~20 GB / 32 GB peak (SAM-3D dominates; Grounded-SAM-2 ≈ 3 GB,
FoundationPose ≈ 1–2 GB).

- How the pipeline works (RGB-D + language → XML): [RIGID_PIPELINE.md](RIGID_PIPELINE.md)
- Design rationale: [PERCEPTION.md](PERCEPTION.md)
- Validation plan + results: [VALIDATION_rigid.md](VALIDATION_rigid.md)
- This file: how to actually install it.

---

## TL;DR

```bash
# 0. clone the two model repos into external/ (or point the env vars elsewhere)
git clone https://github.com/NVlabs/FoundationPose         external/FoundationPose
git clone https://github.com/facebookresearch/sam-3d-objects external/sam-3d-objects

# 1. tell simpact where they are + where their weights live
cp .env.example .env        # edit SIMPACT_FOUNDATIONPOSE_DIR / SIMPACT_SAM3D_DIR

# 2. build the one env (uv sync + the source/git builds that need torch present)
bash scripts/setup_rigid_env.sh

# 3. prove both run in one process
.venv/bin/python scripts/smoke_rigid_coexist.py     # -> RIGID_COEXIST_OK
```

If `scripts/setup_rigid_env.sh` finishes and the smoke prints `RIGID_COEXIST_OK`,
you are done.

---

## Why this needs a script (and isn't just `uv sync`)

`uv sync` installs everything with a PyPI wheel. Three classes of dependency here
have **no usable wheel** for `torch 2.9+cu128 / py3.11` and must be built from
source **with `--no-build-isolation`** so the build sees the already-installed
torch. The env is therefore split into three layers:

| layer | where it's declared | examples |
|---|---|---|
| **wheels** | `pyproject.toml` `[sam3d]` / `[real2sim]` extras | spconv-cu121, timm, lightning, scikit-image, gsplat, optree, … |
| **git/source (need torch at build)** | `scripts/setup_rigid_env.sh`, pinned by commit | pytorch3d, nvdiffrast, utils3d, MoGe, **kaolin** |
| **native C++/CUDA exts** | `scripts/setup_rigid_env.sh` | FoundationPose `mycuda` + `mycpp` |

`scripts/setup_rigid_env.sh` runs all three in order, idempotently (each step is
guarded by an import check, so re-runs are cheap).

> **Do not run `uv sync` without the script afterward.** A bare sync prunes the
> source/git builds (they aren't in `uv.lock`); the script reinstalls them.

---

## Prerequisites

- **GPU + CUDA**: an NVIDIA GPU with ≥ ~24 GB VRAM and a CUDA 12.8 toolkit at
  `/usr/local/cuda-12.8` (`CUDA_HOME`). The stack targets `sm_120` (RTX 5090) but
  works on older archs — set `TORCH_CUDA_ARCH_LIST` for yours.
- **uv** and the pinned interpreter (`.python-version` → 3.11.15).
- **The two model repos** (cloned in step 0). FoundationPose carries the CUDA exts
  built here; SAM-3D is imported via `sys.path`, never pip-installed (see gotchas).
- **Weights / checkpoints** (not redistributed — fetch under each model's license):
  - FoundationPose: `weights/2023-10-28-18-33-37/` (refiner) +
    `weights/2024-01-11-20-02-45/` (scorer), ~1 GB. Point `weights/` at them.
  - SAM-3D: `checkpoints/hf/` (~12 GB, incl. `slat_decoder_mesh.ckpt`). Symlink an
    existing copy or download per the SAM-3D README.
- **kaolin source** (for the mesh path): a local clone at `$KAOLIN_SRC`
  (default `/home/motion/kaolin`), or `git clone
  https://github.com/NVIDIAGameWorks/kaolin`.

---

## What `scripts/setup_rigid_env.sh` does, step by step

1. `uv sync --extra rigid` — core + SAM-3D **wheel** deps. Deliberately **omits
   `mpm`**: FoundationPose pins an older `warp-lang` than the MPM extra, and rigid
   needs no warp, so the conflict simply never arises.
2. **git/source deps** (pinned commits): pytorch3d + nvdiffrast (FoundationPose
   renderer) + utils3d + MoGe.
3. **FoundationPose native exts**:
   - `mycuda` (`common`, `gridencoder`) — built in place. The clone already
     carries the torch-2.9 migration (`-std=c++17`; `.type()→.scalar_type()`).
   - `mycpp` (`cluster_poses`) — **built against the venv's pybind11 (≥ 2.12)**,
     not the system one. See gotcha #1.
4. **kaolin** — built from `$KAOLIN_SRC` (needs `cython`; the PyPI `kaolin` is a
   placeholder). Required by the SAM-3D **mesh** path (gotcha #3).
5. **SAM-3D import check** — confirms `sam3d_objects` imports via `sys.path`
   (it is *not* installed; gotcha #2).

Environment used for the builds: `CUDA_HOME=/usr/local/cuda-12.8`,
`TORCH_CUDA_ARCH_LIST=12.0`.

---

## The gotchas (why naïve installs fail)

**1. FoundationPose `mycpp` segfaults under numpy 2 if built against system
pybind11.** The system pybind11 is often 2.11.x, which predates numpy-2 support;
the resulting `mycpp.so` segfaults in `cluster_poses` the first time
`register()` runs. The script forces the venv pybind11 (≥ 3.0) via
`-Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")`.

**2. Never pip-install SAM-3D, and never install its `requirements.txt`.**
- `requirements.txt` is a `torch 2.5/cu121` kitchen sink (`torchaudio 2.5.1+cu121`,
  `xformers`, `bpy`, …) that would downgrade torch and break the env.
- `pip install -e sam-3d-objects` triggers a hatch `requirements_txt` build hook
  that pulls that same kitchen sink.

  Instead, SAM-3D is imported via `sys.path` (the same pattern as FoundationPose),
  with `LIDRA_SKIP_INIT=true` to skip a Meta-internal `init` submodule absent from
  the public checkout:
  ```python
  import os; os.environ["LIDRA_SKIP_INIT"] = "true"
  import sys; sys.path.insert(0, os.environ["SIMPACT_SAM3D_DIR"])
  import sam3d_objects  # noqa
  from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
  ```

**3. The mesh path requires `kaolin`.** "Mesh-only" skips the gaussian-splat
*output*, but mesh extraction (`flexicubes`) imports `kaolin.utils.testing`, so
kaolin is mandatory. The PyPI `kaolin` is only a placeholder wheel — build the
real one from source.

**4. Drive the core pipeline, not `notebook/inference.py`.** That wrapper hard-imports
`seaborn`/`gradio`/`kaolin.visualize` and assumes a conda env
(`os.environ["CONDA_PREFIX"]`). The pipeline is reached directly:
```python
from omegaconf import OmegaConf
from hydra.utils import instantiate
cfg = OmegaConf.load("checkpoints/hf/pipeline.yaml")
cfg.rendering_engine = "pytorch3d"; cfg.compile_model = False
cfg.workspace_dir = "checkpoints/hf"
pipe = instantiate(cfg)          # loads all decoders to GPU
out = pipe.run(rgba, None, seed, with_mesh_postprocess=False,
               with_texture_baking=False, with_layout_postprocess=False,
               use_vertex_color=True, stage1_inference_steps=None, pointmap=None)
mesh = out["glb"]                # trimesh; also out["gaussian"], out["pointmap"]
```
(`rgba` = HxWx4 uint8 with the object mask in the alpha channel.)

**5. Grounded-SAM-2 needs no repo clone, but mind the transformers 5.x API.**
GroundingDINO loads from `transformers` (HF `IDEA-Research/grounding-dino-tiny`)
and SAM2 + its configs (`configs/sam2.1/sam2.1_hiera_l.yaml`) ship inside the
`sam2` wheel — so `SIMPACT_GROUNDED_SAM2_DIR` is optional (only the SAM2
checkpoint `sam2.1_hiera_large.pt`, ~900 MB, is an external download:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
# then set SIMPACT_SAM2_CHECKPOINT=/path/to/sam2.1_hiera_large.pt in .env
```

— the official release asset; see the `facebookresearch/sam2` README for
alternatives). On
transformers 5.x, `post_process_grounded_object_detection` renamed
`box_threshold`→`threshold` and returns string labels under `text_labels` (not
`labels`). Also scope `torch.autocast(dtype=bfloat16)` to the segment call — the
legacy script entered it process-wide, which would leak into SAM-3D/FoundationPose.

---

## Verify

```bash
.venv/bin/python scripts/smoke_gsam2.py          # Grounded-SAM-2 segmentation (recorded scene)
.venv/bin/python scripts/smoke_sam3d.py          # SAM-3D mesh only (kid_box sample)
.venv/bin/python scripts/smoke_rigid_coexist.py  # SAM-3D + FoundationPose in one process
```

Expected (RTX 5090):

```
[sam3d] loaded in ~14s; reconstruct ~18s; mesh verts≈600k
[fp]    loaded in ~2s; register ~0.6s; trans err vs golden ≈ 0.6 mm
[coexist] BOTH models resident + ran in one process. peak GPU mem ≈ 19.9 GB / 32 GB
RIGID_COEXIST_OK
```

---

## Troubleshooting

| symptom | cause → fix |
|---|---|
| `Segmentation fault` in `mycpp.cluster_poses` | mycpp built against system pybind11 < 2.12 → rebuild with venv `pybind11_DIR` (gotcha #1). |
| `ImportError: kaolin placeholder wheel` | mesh path needs real kaolin → build from `$KAOLIN_SRC` (gotcha #3). |
| editable SAM-3D install pulls `torchaudio==2.5.1+cu121` / `xformers` | you ran `pip install -e` on SAM-3D → don't; use `sys.path` (gotcha #2). |
| `KeyError: 'CONDA_PREFIX'` | you imported `notebook/inference.py` → drive the core pipeline instead (gotcha #4). |
| `ModuleNotFoundError: sam3d_objects.init` | `LIDRA_SKIP_INIT` not set → `export LIDRA_SKIP_INIT=true`. |
| torch silently downgraded off `2.9.0+cu128` | a dep dragged a CPU/other-CUDA torch → reinstall torch from the cu128 index (see `pyproject.toml` `[tool.uv.sources]`). |
