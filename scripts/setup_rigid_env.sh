#!/usr/bin/env bash
# Build the single "rigid perception" uv env: SAM-3D (image->3D, mesh path) +
# FoundationPose (6-DoF pose) in ONE uv env on torch 2.9+cu128 / py3.11 / RTX 5090.
#
# This is intentionally NOT pure `uv sync`: several deps are git/source builds
# with no usable PyPI wheel and must be installed with --no-build-isolation so the
# build sees the already-installed torch. Wheel-installable deps live in the
# `rigid` extra in pyproject.toml; the source/git ones are pinned by commit here.
#
# Mesh-only: kaolin / gsplat / xformers / flash_attn / bpy are deliberately
# skipped (the gaussian-splat + texture-bake paths). The sync includes `dev`
# (mpm/generator/arap + pytest) so ONE env runs both the rigid perception AND
# the deformable/planning reproduce: the feared warp-lang conflict is moot —
# kaolin's warp-lang requirement is unpinned (1.10.0 satisfies it) and
# FoundationPose's requirements.txt (the old 1.0.2 pin) is never installed,
# only its native exts are built in place.
#
# Idempotent: every step is guarded by an import check, so re-runs are cheap.
set -euo pipefail
cd "$(dirname "$0")/.."                     # repo root

# Load .env (SIMPACT_* model paths) like reproduce_all.sh does; drop unresolvable
# /path/to placeholders so the in-repo defaults below take effect instead.
[ -f .env ] && set -a && . ./.env && set +a
for _v in SIMPACT_SAM3D_DIR SIMPACT_FOUNDATIONPOSE_DIR SIMPACT_SAM2_CHECKPOINT SIMPACT_GROUNDED_SAM2_DIR; do
  _p="${!_v:-}"; [ -n "$_p" ] && [ ! -e "$_p" ] && unset "$_v"
done

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}   # RTX 5090 = sm_120

FP_DIR=${SIMPACT_FOUNDATIONPOSE_DIR:-$PWD/external/FoundationPose}
SAM3D_DIR=${SIMPACT_SAM3D_DIR:-$PWD/external/sam-3d-objects}

# Preflight: the model repos are NOT shipped with this repo (their code + weights
# are fetched under their own licenses). Clone them first, or point the env vars
# at existing clones. See docs/RIGID_ENV_SETUP.md ("TL;DR", step 0).
missing=0
if [ ! -d "$FP_DIR" ]; then
  echo "ERROR: FoundationPose clone not found at: $FP_DIR"
  echo "  git clone https://github.com/NVlabs/FoundationPose external/FoundationPose"
  echo "  (+ its weights/ — see docs/RIGID_ENV_SETUP.md), or set"
  echo "  SIMPACT_FOUNDATIONPOSE_DIR=/path/to/existing/FoundationPose in .env"
  missing=1
fi
if [ ! -d "$SAM3D_DIR" ]; then
  echo "ERROR: SAM-3D clone not found at: $SAM3D_DIR"
  echo "  git clone https://github.com/facebookresearch/sam-3d-objects external/sam-3d-objects"
  echo "  (+ its checkpoints/hf/ weights — see docs/RIGID_ENV_SETUP.md), or set"
  echo "  SIMPACT_SAM3D_DIR=/path/to/existing/sam-3d-objects in .env"
  missing=1
fi
[ $missing = 1 ] && exit 1

py()   { uv run --no-sync python "$@"; }
have() { py -c "import $1" 2>/dev/null; }
log()  { printf '\n=== %s ===\n' "$1"; }

# 1. Core + SAM-3D wheel deps (real2sim + sam3d) + the dev extras (mpm/warp,
#    generator, arap, pytest) so this one env also runs reproduce_all.sh fully.
#    NOTE: any later bare `uv sync` prunes the source builds below — always
#    sync with BOTH extras and re-run this script (idempotent) afterward.
log "uv sync --extra rigid --extra dev"
uv sync --extra rigid --extra dev

# 2. Shared / git source deps (no PyPI wheel for this torch).  Pinned by commit.
log "git/source deps (pytorch3d, nvdiffrast, utils3d, moge)"
have pytorch3d  || uv pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
have nvdiffrast || uv pip install --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git"
have utils3d    || uv pip install --no-build-isolation \
    "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"
have moge       || uv pip install --no-build-isolation \
    "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"

# 3. FoundationPose native extensions.
#  3a. mycuda (common + gridencoder). A FRESH NVlabs clone needs the 2-line
#      torch-2.9 patch first (-std=c++14 -> -std=c++17 in bundlesdf/mycuda/setup.py;
#      .type() -> .scalar_type() at the three call sites in common.cu) — see
#      docs/VALIDATION_rigid.md "FoundationPose mycuda torch-2.9 patch".
log "FoundationPose mycuda (build in place)"
have common || ( cd "$FP_DIR/bundlesdf/mycuda" && py setup.py build_ext --inplace )
#  3b. mycpp (cluster_poses): MUST build against the venv's pybind11 (>=2.12).
#      The system pybind11 (2.11.x) predates numpy-2 support and segfaults under
#      numpy>=2. See external/FoundationPose/build_all.sh comment.
log "FoundationPose mycpp (venv pybind11)"
if ! py -c "import sys; sys.path.insert(0,'$FP_DIR/mycpp/build'); import mycpp" 2>/dev/null; then
    PYBIND_DIR=$(py -c "import pybind11; print(pybind11.get_cmake_dir())")
    ( cd "$FP_DIR/mycpp" && rm -rf build && mkdir build && cd build \
        && cmake .. -DPYTHON_EXECUTABLE="$(uv run --no-sync which python)" \
                    -Dpybind11_DIR="$PYBIND_DIR" \
        && make -j"$(nproc)" )
fi

# 4. kaolin — required by the SAM-3D MESH path (flexicubes surface extraction
#    imports kaolin.utils.testing). The PyPI `kaolin` is a placeholder; build the
#    real one from source against this torch. KAOLIN_SRC defaults to a local
#    checkout; clone https://github.com/NVIDIAGameWorks/kaolin if absent.
log "kaolin (mesh path: flexicubes)"
KAOLIN_SRC=${KAOLIN_SRC:-$HOME/kaolin}
if ! have kaolin; then
    # kaolin's setup.py imports pkg_resources: uv venvs ship without setuptools,
    # and setuptools >= 82 removed pkg_resources — pin the last version with it
    have pkg_resources || uv pip install "setuptools<82"
    uv pip install "cython>=0.29.37"
    IGNORE_TORCH_VER=1 uv pip install --no-build-isolation "$KAOLIN_SRC"
fi

# 5. SAM-3D itself is NOT pip-installed. Its pyproject uses a hatch
#    `requirements_txt` hook that would drag in a torch-2.5/cu121 kitchen sink and
#    break this env. Instead it is imported via sys.path (same pattern as
#    FoundationPose), with LIDRA_SKIP_INIT=true to skip the Meta-internal `init`
#    submodule. The adapter/harness sets:
#        os.environ["LIDRA_SKIP_INIT"] = "true"
#        sys.path.insert(0, $SIMPACT_SAM3D_DIR)            # import sam3d_objects
#        sys.path.insert(0, $SIMPACT_SAM3D_DIR/notebook)   # from inference import Inference
#    Verify the core mesh pipeline imports:
log "SAM-3D import check (sys.path, no install)"
LIDRA_SKIP_INIT=true py -c "import sys; sys.path.insert(0,'$SAM3D_DIR'); \
import sam3d_objects; \
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap; \
print('OK: SAM-3D core mesh pipeline imports')"

# 6. Grounded-SAM-2 (segmentation, stage 2). transformers + supervision came from
#    the `gsam2` extra above; `sam2` has no wheel for this torch and builds from
#    sdist here. NO Grounded-SAM-2 repo clone needed (GroundingDINO = transformers
#    HF model; SAM2 configs ship inside the sam2 wheel). The SAM2 checkpoint
#    (sam2.1_hiera_large.pt) is a separate ~900 MB download — point the adapter at
#    it via SIMPACT_GROUNDED_SAM2_DIR/checkpoints or a direct path.
log "Grounded-SAM-2 (sam2 sdist build + import check)"
have sam2 || uv pip install sam2
py -c "from sam2.build_sam import build_sam2; from sam2.sam2_image_predictor import SAM2ImagePredictor; \
from transformers import AutoModelForZeroShotObjectDetection; import supervision; \
print('OK: Grounded-SAM-2 (sam2 + transformers GroundingDINO + supervision) imports')"

# 7. Checkpoints (SAM-3D mesh path). Symlink the local 12 GB blob if present;
#    otherwise fetch per the SAM-3D README into $SAM3D_DIR/checkpoints/hf.
log "checkpoints"
if [ ! -e "$SAM3D_DIR/checkpoints/hf/slat_decoder_mesh.ckpt" ]; then
    echo "WARNING: $SAM3D_DIR/checkpoints/hf/slat_decoder_mesh.ckpt missing."
    echo "  Symlink an existing checkpoints/hf, or download per SAM-3D README."
fi

log "rigid env ready"
py - <<'EOF'
import torch
print("torch    :", torch.__version__, "cuda", torch.version.cuda)
print("device   :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
import numpy, pytorch3d  # noqa
print("numpy    :", numpy.__version__)
print("OK: core imports resolve")
EOF
