# Perception backends — integration & release design

How simpact integrates the external perception models the real2sim pipeline
depends on — **Grounded-SAM-2** (segmentation, stage 2), **SAM-3D /
Hunyuan3D** (image→3D, stage 4), and **FoundationPose** (6-DoF pose, stage 6) —
so the repo can be released and others can run these models inside it.

> Status: **design** (not yet implemented). The offline pipeline (stages 3/7/8)
> is done. The
> single-env finding below was verified by a build spike (2026-06-21).

## The governing constraint (and what the spike changed)

Both target models expose clean, importable Python APIs that map directly onto
the adapter interfaces already in `simpact/real2sim/perception/base.py`:

| model | entry API | maps to |
|---|---|---|
| FoundationPose | `estimater.py::FoundationPose` (rgb+depth+mask+K+mesh → 4×4) | `PoseEstimator.estimate()` |
| SAM-3D | `inference.Inference(cfg)(image, mask, seed)` → mesh | `ImageTo3DReconstructor.reconstruct()` |
| Grounded-SAM-2 | `sam2` + GroundingDINO (image+prompt → masks/boxes) | `Segmenter.segment()` |

The interface was never the problem; the worry was that each model needs heavy
native deps (FoundationPose: a built CUDA C++ ext + `nvdiffrast`; SAM-3D:
`pytorch3d`, `spconv-cu121`, `xformers`, optional `bpy`/`MoGe`) that might not
build against simpact's `torch 2.9+cu128 / py3.11`.

**A build spike settled this: a single uv env IS sufficient to run both
FoundationPose and SAM-3D** (plus Grounded-SAM-2), in-process, on
`torch 2.9+cu128 / py3.11` / RTX 5090. Verified by compiling every native dep
against that torch on the 5090:

| component | result on torch 2.9+cu128 |
|---|---|
| FoundationPose `bundlesdf/mycuda` (`common`+`gridencoder`) | ✅ builds + imports after a 2-line patch (below) |
| `nvdiffrast` (`RasterizeCudaContext`) | ✅ builds + runs (`uv pip install --no-build-isolation`) |
| `pytorch3d` (shared by FP + SAM-3D) | ✅ a working cu128/torch-2.9/cp311 build exists (the `sam3d-objects` env) |
| Grounded-SAM-2 / `sam2` | ✅ pip-resolves (`torch>=2.5.1`), no native build |
| open3d / mujoco / scipy / trimesh | ✅ co-resolve (FP's `scipy==1.12.0` / `trimesh==4.2.2` satisfy all) |

The feared torch C++ API drift was just two mechanical patches to FoundationPose
(see *Build steps* below). No GPU-arch problem — nvcc 12.8 compiled `sm_120`
cleanly. (Spike confirmed **build + import**, not a full weighted inference run.)

## Core design: adapters with a selectable execution mode

Keep `base.py`'s ABCs as the stable seam. Each backend has **one** adapter that
runs the model in one of two modes, selected by config:

- **`inprocess`** — import the model directly in simpact's interpreter. **Now the
  default for the rigid pipeline: Grounded-SAM-2 + SAM-3D + FoundationPose all
  run in one uv env** (`torch 2.9+cu128 / py3.11`), per the spike above.
- **`subprocess`** — adapter writes inputs to a temp dir, runs
  `<backend_python> runner.py --in in.json --out out.json` in a separate env,
  reads the result back. Retained as a **fallback** for backends that genuinely
  cannot share the env (e.g. Hunyuan3D's `bpy`/own CUDA, or any future model
  pinned to an incompatible torch), and for running a model on a different GPU.

**Recommended default: single in-process uv env for the rigid pipeline**
(GSAM2 + SAM-3D + FoundationPose), with subprocess kept available per-backend for
the incompatible exceptions. Two practical guards:
- **Do not install the `mpm` extra** in this env — FoundationPose pins
  `warp-lang==1.0.2` vs the MPM extra's `1.10.0`; rigid needs no warp, so the
  conflict simply doesn't arise.
- **SAM-3D's optional `bpy`** (texture bake) is the one unverified dep — it was
  absent from the working `sam3d-objects` env, so the core mesh path likely
  doesn't need it; confirm if you enable texturing.

### IPC contract (subprocess mode)

JSON in / JSON out, file paths for large arrays:

```
foundationpose_runner:  in {rgb, depth, mask, K, mesh, est_refine_iter}
                        out {pose: [[..4×4..]], score}
sam3d_runner:           in {image, mask, seed, checkpoint_tag}
                        out {mesh_path}
gsam2_runner:           in {image, text_prompt, box_threshold, sam2_checkpoint}
                        out {masks_path, labels, scores, boxes}
```

The in-process adapter stays in core (light); the **runner** imports the heavy
model and only ever executes under the backend interpreter.

## Repo structure (target)

```
simpact/real2sim/perception/
  base.py              # ABCs + dataclasses (exists)
  config.py            # NEW typed config (dataclass + dacite/yaml — already deps)
  registry.py          # NEW build_segmenter / build_reconstructor / build_pose_estimator
  backends/            # thin adapters (in-proc or shell-out)
    grounded_sam2.py   foundationpose.py   sam3d.py   hunyuan3d.py
  runners/             # NEW — run INSIDE each model's env, JSON in/out
    foundationpose_runner.py   sam3d_runner.py   gsam2_runner.py
configs/
  perception/{grounded_sam2,sam3d,foundationpose}.yaml   # committed defaults
  pipeline/{rigid,rope}.yaml                             # backends + params per pipeline
envs/                  # NEW per-backend environment manifests (reproduce each env)
  foundationpose.environment.yaml   sam3d.environment.yaml
scripts/
  run_real2sim.py      # gains --online to invoke backends
  setup_perception.py  # NEW bootstrap: clone repos@pin, build envs, fetch weights
.env.example           # paths / interpreters / devices / secrets
```

## Configuration — three layers, no new dependencies

`dacite` + `pyyaml` already ship in the `real2sim` extra, so typed-YAML config
is free.

1. **`.env`** — machine-specific, never committed. Repo dirs, per-backend
   interpreters, devices, API keys:
   ```
   SIMPACT_FOUNDATIONPOSE_DIR=/home/motion/FoundationPose
   SIMPACT_FOUNDATIONPOSE_PYTHON=/home/.../envs/foundationpose/bin/python
   SIMPACT_SAM3D_DIR=/home/motion/SPARCS/dependencies/sam-3d-objects
   SIMPACT_SAM3D_PYTHON=/home/.../envs/sam3d-objects/bin/python
   ```
2. **`configs/perception/*.yaml`** — committed, shareable defaults with
   `${ENV}` interpolation so secrets/paths stay in `.env`:
   ```yaml
   # configs/perception/sam3d.yaml
   backend: sam3d
   mode: inprocess            # auto-falls back to subprocess off-env
   repo_dir: ${SIMPACT_SAM3D_DIR}
   python:   ${SIMPACT_SAM3D_PYTHON}
   checkpoint_tag: sam-3d-objects-v1
   device: cuda:0
   params: { seed: 42 }
   ```
   ```yaml
   # configs/perception/foundationpose.yaml
   backend: foundationpose
   mode: inprocess            # verified to share simpact's cu128 env (see spike)
   repo_dir: ${SIMPACT_FOUNDATIONPOSE_DIR}
   python:   ${SIMPACT_FOUNDATIONPOSE_PYTHON}   # only used if mode: subprocess
   params: { est_refine_iter: 5, track_refine_iter: 2 }
   ```
3. **Typed objects + registry** — one validated entry point:
   ```python
   @dataclass
   class BackendConfig:
       backend: str; repo_dir: Path; python: Path
       mode: str = "subprocess"; device: str = "cuda:0"
       params: dict = field(default_factory=dict)

   seg   = build_segmenter(load_cfg("configs/perception/grounded_sam2.yaml"))
   recon = build_reconstructor(load_cfg("configs/perception/sam3d.yaml"))
   ```
   `pipeline/*.yaml` names which backend config each stage uses, so swapping
   Hunyuan3D → SAM-3D is a one-line config change, not a code change.

## Build steps — rigid single env (verified)

> **Concrete install guide: [RIGID_ENV_SETUP.md](RIGID_ENV_SETUP.md)** — the
> implemented, step-by-step version of this section (`scripts/setup_rigid_env.sh`
> + `scripts/smoke_rigid_coexist.py`), with the four gotchas and troubleshooting.
> The sketch below is the original design.

One uv env on `torch 2.9+cu128 / py3.11` running Grounded-SAM-2 + SAM-3D +
FoundationPose. This is **not** pure `uv sync`: three deps are git/source builds
(no PyPI wheels) and need `--no-build-isolation` so the build sees torch. Encode
these in `scripts/setup_rigid_env.sh`:

```bash
uv sync --extra real2sim            # core: torch 2.9+cu128, open3d, mujoco, ...
                                    # (do NOT add --extra mpm: warp-lang conflict)
uv pip install sam2                 # stage 2 (torch>=2.5.1; no build)
# pytorch3d — shared by FP + SAM-3D (source/FAIR build for cu128/torch-2.9/cp311)
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
# nvdiffrast — FoundationPose renderer (JIT CUDA plugin)
uv pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
# FoundationPose mycuda — apply the torch-2.9 patch, then build in place
git -C "$SIMPACT_FOUNDATIONPOSE_DIR" apply patches/foundationpose_torch29.patch
uv pip install --no-build-isolation -e "$SIMPACT_FOUNDATIONPOSE_DIR/bundlesdf/mycuda"
```

The **`foundationpose_torch29.patch`** (vendored in `patches/`, applied at setup
so the upstream clone is untouched) is two mechanical torch-API migrations:

1. `bundlesdf/mycuda/setup.py`: `-std=c++14` → `-std=c++17` (torch 2.9 headers
   require C++17).
2. `bundlesdf/mycuda/common.cu`: `.type()` → `.scalar_type()` in the three
   `AT_DISPATCH_FLOATING_TYPES(...)` calls (deprecated implicit
   `DeprecatedTypeProperties → ScalarType` conversion was removed).

Build env: `CUDA_HOME=/usr/local/cuda-12.8`, `TORCH_CUDA_ARCH_LIST=12.0` (5090 /
`sm_120`). Because these are source/local installs they are **not** captured in
`uv.lock` like wheels — pin them by git commit in the setup script.

## Release mechanics

- **Core stays light & installable.** The offline real2sim + sim path works with
  zero models; perception is opt-in.
- **Do not vendor model code or weights.** Size aside, **licensing** forbids it:
  FoundationPose is NVIDIA non-commercial; SAM-3D and Hunyuan3D each carry their
  own terms. The release ships *adapters + runners + env manifests + a bootstrap
  script*; users fetch the models under their own licenses. State this in a
  top-level `PERCEPTION_SETUP.md`.
- **`scripts/setup_perception.py`** clones each repo at a pinned commit, builds
  its env from `envs/*.yaml`, downloads checkpoints, and writes `.env` — so
  "use it inside this repo" is one command per backend.
- **CI** runs the hermetic offline tests (already built); model-backed tests
  stay opt-in / env-gated like `test_golden_trial`.
- **Pin everything**: repo commits in the bootstrap, checkpoint tags in config,
  env manifests committed. Reproducibility is the point.

## Suggested build order

1. `config.py` + `registry.py` + the `${ENV}`-interpolating loader — no models
   needed, unit-testable immediately. The seam everything plugs into.
2. `scripts/setup_rigid_env.sh` + `patches/foundationpose_torch29.patch` — the
   verified single rigid env (above).
3. SAM-3D adapter **in-process** → validates the `reconstruct()` seam.
4. FoundationPose adapter **in-process** → validates `estimate()` in the same
   env (subprocess runner kept as the fallback path).
5. Grounded-SAM-2 adapter (sam2 + GroundingDINO).
6. `setup_perception.py` + `envs/*.yaml` + `PERCEPTION_SETUP.md` for release
   (env manifests/runners remain for the subprocess-only backends, e.g.
   Hunyuan3D).
