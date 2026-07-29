#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduce simpact real2sim + action-planning results for ALL examples.
#
# It auto-detects prerequisites and runs what it can, SKIPPING (never failing)
# the steps whose inputs are missing:
#   - GOOGLE_API_KEY  -> required for the closed-loop planning demos (Gemini)
#   - CUDA + warp     -> required for the MPM dough/sweep rollouts
#   - external models -> SAM-3D + FoundationPose + SAM2 ckpt; when detected, the push
#                        perception REBUILDS the scene from the committed capture/ and is
#                        VERIFIED against the golden reconstruction bundled in sim/
#                        (planning uses the committed sim/ unless --full-real2sim or a
#                        user-supplied PUSH_DATA)
#   - SAM2 ckpt + GPU + API key -> the deformable real2sim stage: each
#                        rope/dough/sweep scene is REBUILT from its capture/ (RGB-D + EE
#                        record) and verified against the committed sim/ (tolerated VLM
#                        variation). Planning still uses the committed reference unless
#                        --full-real2sim, which plans on the rebuilt scene instead — the
#                        end-user capture-only chain.
#
# Outputs go to a scratch dir by default (does NOT touch the committed
# examples/*/<trial>/runs unless you pass --overwrite-examples). The default scratch root is
# TIMESTAMPED per run, so a new run never overwrites a previous run's outputs. If you
# pin --out-dir (or OUT_DIR) to an existing non-empty dir, the run aborts unless --force.
#
# Usage:
#   scripts/reproduce_all.sh [options]
#     --out-dir DIR          scratch output root  (default /tmp/simpact_reproduce/run_<ts>)
#     --force                reuse a non-empty --out-dir (else the run aborts to protect
#                            existing outputs)
#     --overwrite-examples   write into examples/*/<trial>/runs instead of the scratch dir
#     --tasks LIST           comma list of task cases to run (default: all four)
#                            choices: push,rope,dough,sweep   e.g. --tasks rope,dough
#     --quick                fewer iters/steps for a fast smoke
#     --skip-tests           skip the pytest + smoke sanity checks
#     --skip-real2sim        skip the real2sim reconstruction stage
#     --skip-planning        skip the closed-loop planning demos
#     --full-perception      FORCE the push perception (GSAM2->SAM-3D->FoundationPose) even
#                            if model auto-detection fails. It already runs by default when
#                            the models are found; rebuilds the scene from the example
#                            RGB-D and verifies it against the committed sim/.
#     --skip-perception      never run push perception (planning uses the committed sim/)
#     --full-real2sim        plan every task on the scene REBUILT from capture/
#                            (default: rebuild is verify-only; planning uses committed sim/)
#     -h | --help
#
#   Ctrl-C aborts the whole run (prints the summary so far).
#
#   Env overrides: OUT_DIR (a fixed root; still guarded against overwrite unless --force),
#                  PUSH_DATA (OPTIONAL override: plan push on your own assets dir),
#                  PUSH_TRIAL, MAX_ITERS, NUM_STEPS_DOUGH, NUM_STEPS_SWEEP, MUJOCO_GL
#
# This script reads inputs ONLY from the repo's examples/ (and an optional user-supplied
# PUSH_DATA); it never reaches into any external checkout.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---- config (env-overridable) ----
# Default out root is timestamped so successive runs never overwrite each other. A fixed
# OUT_DIR (env) or --out-dir is honored verbatim, but guarded below unless --force.
OUT_DIR="${OUT_DIR:-/tmp/simpact_reproduce/run_$(date +%Y%m%d_%H%M%S)}"
PUSH_TRIAL="${PUSH_TRIAL:-0103_push_0}"   # which committed push example scene to use
# The committed push trial bundles the golden reconstruction (textured {obj}_scaled.obj
# + {obj}_6d_cam{cam}.txt in sim/), so planning runs from examples/ alone. PUSH_DATA is
# an OPTIONAL override: point it at your own reconstructed-assets dir to plan on that
# instead (a fresh perception build is adopted automatically under --full-real2sim).
PUSH_DATA="${PUSH_DATA:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
MAX_ITERS="${MAX_ITERS:-3}"
NUM_STEPS_DOUGH="${NUM_STEPS_DOUGH:-200}"
NUM_STEPS_SWEEP="${NUM_STEPS_SWEEP:-120}"
PY="uv run python"
# Which task cases to run (comma list). Default: all four. --tasks overrides. Each still
# needs its capability gate (dough/sweep->CUDA+warp, planning->API key).
TASKS="${TASKS:-push,rope,dough,sweep}"
want() { case ",$TASKS," in *,"$1",*) return 0;; *) return 1;; esac; }

DO_TESTS=1; DO_REAL2SIM=1; DO_PLANNING=1; OVERWRITE_EXAMPLES=0; FORCE=0
# Deformable real2sim mode: verify = rebuild each scene from capture/ and CHECK it
# against the committed sim/ (planning still uses the committed reference);
# full = the rebuilt scene also REPLACES the committed one for planning
# (the end-user capture-only chain). --full-real2sim selects full.
R2S_MODE=verify
# Push perception (RGB-D -> meshes/poses -> scene): auto = run if the external models are
# detected; force = run regardless (--full-perception); off = never (--skip-perception).
DO_PERCEPTION=auto

while [ $# -gt 0 ]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2;;
    --tasks) TASKS="$2"; shift 2;;
    --force) FORCE=1; shift;;
    --overwrite-examples) OVERWRITE_EXAMPLES=1; shift;;
    --quick) MAX_ITERS=1; NUM_STEPS_DOUGH=120; NUM_STEPS_SWEEP=80; shift;;
    --skip-tests) DO_TESTS=0; shift;;
    --skip-real2sim) DO_REAL2SIM=0; shift;;
    --skip-planning) DO_PLANNING=0; shift;;
    --full-perception) DO_PERCEPTION=force; shift;;
    --skip-perception) DO_PERCEPTION=off; shift;;
    --full-real2sim) R2S_MODE=full; shift;;
    -h|--help) sed -n '2,48p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

# Validate --tasks against the known task cases.
for t in ${TASKS//,/ }; do
  case "$t" in push|rope|dough|sweep) ;;
    *) echo "unknown --tasks value: '$t'  (valid: push rope dough sweep)" >&2; exit 2;; esac
done

# Don't clobber existing outputs: refuse a non-empty out root unless --force. The default
# root is timestamped (won't pre-exist); this guards a pinned --out-dir / OUT_DIR that
# already holds results from a previous run.
if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ] && [ "$FORCE" = 0 ]; then
  echo "ERROR: output dir already exists and is non-empty: $OUT_DIR" >&2
  echo "       pass --out-dir <new-dir> for a fresh location, or --force to reuse it." >&2
  exit 1
fi
mkdir -p "$OUT_DIR"
RESULTS=()   # "STATUS | step | detail"
record() { RESULTS+=("$1 | $2 | ${3:-}"); }
skip_step() { echo; echo "-------------------- SKIP: $1 --------------------"; echo "  reason: $2"; record "SKIP" "$1" "$2"; }

summarize() {
  echo; echo "==================== SUMMARY ===================="
  local pass=0 fail=0 skip=0 intr=0 r
  for r in "${RESULTS[@]:-}"; do
    [ -z "$r" ] && continue
    echo "  $r"
    case "$r" in PASS*) pass=$((pass+1));; FAIL*) fail=$((fail+1));;
                 SKIP*) skip=$((skip+1));; INT*) intr=$((intr+1));; esac
  done
  echo "-------------------------------------------------"
  echo "  $pass passed, $fail failed, $skip skipped$([ $intr -gt 0 ] && echo ", $intr interrupted")   (outputs under $OUT_DIR)"
  return $fail
}

# run a labelled step; a user interrupt (Ctrl-C -> exit 130) aborts the whole run.
run_step() {
  local name="$1"; shift
  echo; echo "==================== $name ===================="
  local rc=0; "$@" || rc=$?
  if   [ $rc -eq 0 ];   then record "PASS" "$name"
  elif [ $rc -eq 130 ]; then record "INT" "$name" "user interrupt"; echo; echo "Interrupted — aborting."; summarize; exit 130
  else record "FAIL" "$name" "exit $rc"; fi
}

# ---- capability detection ----
[ -f .env ] && set -a && . ./.env && set +a   # load GOOGLE_API_KEY etc. if present
# .env may carry placeholder model paths (e.g. /path/to/...); drop any that don't resolve so
# run_rigid_pipeline's own defaults (the real local checkpoints) take effect.
for _v in SIMPACT_SAM3D_DIR SIMPACT_FOUNDATIONPOSE_DIR SIMPACT_SAM2_CHECKPOINT SIMPACT_GROUNDED_SAM2_DIR; do
  _p="${!_v:-}"; [ -n "$_p" ] && [ ! -e "$_p" ] && unset "$_v"
done
HAVE_KEY=0; [ -n "${GOOGLE_API_KEY:-}" ] && HAVE_KEY=1
HAVE_GPU=0
$PY - <<'PY' >/dev/null 2>&1 && HAVE_GPU=1
import torch, warp  # noqa
assert torch.cuda.is_available()
PY
PUSH_SCENE="$ROOT/examples/push_real2sim/$PUSH_TRIAL"   # committed trial (capture/ + sim/)
PUSH_BUILD="$OUT_DIR/push_build"                               # full-perception reconstruction
PUSH_K="$ROOT/assets/calibration/0103/cam1_K.txt"              # registry intrinsics (profile 0103)
HAVE_PUSH=0
[ -n "$PUSH_DATA" ] && [ -d "$PUSH_DATA" ] && HAVE_PUSH=1
PUSH_OBJECTS="$($PY -c "from simpact.tasks import TASKS; print(TASKS['push'].build_object_prompt)")"

# Push perception needs the heavy external models (not bundleable): SAM-3D + FoundationPose
# + the SAM2 checkpoint. Detect them so perception can run BY DEFAULT where available.
SAM3D_DIR="${SIMPACT_SAM3D_DIR:-$ROOT/external/sam-3d-objects}"
FP_DIR="${SIMPACT_FOUNDATIONPOSE_DIR:-$ROOT/external/FoundationPose}"
# SAM2 checkpoint: env if it resolves, else the pipeline's default (matches run_rigid_pipeline.py)
SAM2_CKPT="${SIMPACT_SAM2_CHECKPOINT:-}"
[ -f "$SAM2_CKPT" ] || SAM2_CKPT="$HOME/sam2/checkpoints/sam2.1_hiera_large.pt"
HAVE_PERCEPTION=0
[ $HAVE_GPU = 1 ] && [ -f "$SAM3D_DIR/checkpoints/hf/slat_decoder_mesh.ckpt" ] \
  && [ -d "$FP_DIR" ] && [ -f "$SAM2_CKPT" ] && HAVE_PERCEPTION=1
# Deformable real2sim (capture/ -> sim/) needs only segmentation + the VLM — a strict
# subset of the push stack. The segmenter runs from the sam2 wheel + transformers
# (a Grounded-SAM-2 repo clone is OPTIONAL); the only artifact is the SAM2 checkpoint.
HAVE_GSAM2=0
[ $HAVE_GPU = 1 ] && [ -f "$SAM2_CKPT" ] && HAVE_GSAM2=1
# resolve the auto/force/off policy into a boolean for this run
RUN_PERCEPTION=0
case "$DO_PERCEPTION" in
  force) RUN_PERCEPTION=1;;
  auto)  [ $HAVE_PERCEPTION = 1 ] && RUN_PERCEPTION=1;;
esac

echo "simpact reproduce — prerequisites"
echo "  GOOGLE_API_KEY : $([ $HAVE_KEY = 1 ] && echo yes || echo 'NO  (planning demos will skip)')"
echo "  CUDA + warp    : $([ $HAVE_GPU = 1 ] && echo yes || echo 'NO  (dough/sweep will skip)')"
echo "  push percept.  : $([ $RUN_PERCEPTION = 1 ] && echo "yes ($DO_PERCEPTION)" || echo "no ($DO_PERCEPTION; models $([ $HAVE_PERCEPTION = 1 ] && echo found || echo 'not found'))")"
echo "  push assets    : $([ $HAVE_PUSH = 1 ] && echo "override: $PUSH_DATA" || echo 'committed sim/ (golden reconstruction)')"
echo "  deform real2sim: $([ $HAVE_GSAM2 = 1 ] && echo "yes ($R2S_MODE)" || echo "no (needs SAM2 ckpt + GPU)")"
echo "  tasks          : $TASKS"
echo "  out root       : $OUT_DIR   (overwrite-examples=$OVERWRITE_EXAMPLES, force=$FORCE)"
echo "  MUJOCO_GL=$MUJOCO_GL  MAX_ITERS=$MAX_ITERS"

# Stage a working COPY of the push assets (PUSH_DATA — user-supplied or produced by
# perception below) into the scratch dir so the source stays pristine. Idempotent + lazy:
# called right before the step that needs it, after perception may have set PUSH_DATA.
PUSH_WORK=""
stage_push() {
  [ -n "$PUSH_WORK" ] && return 0
  # Keep the REAL leaf name ($PUSH_TRIAL) — the planning demo falls back to the bundled
  # examples/.../<trial>/ (scene.yaml initial_ee_pose) keyed on data_dir.name; a renamed
  # dir silently falls back to the generic home pose (wrong initial gripper).
  PUSH_WORK="$OUT_DIR/push_work/$PUSH_TRIAL"
  echo "  staging push assets -> $PUSH_WORK  (PUSH_DATA left untouched)"
  rm -rf "$PUSH_WORK"; mkdir -p "$(dirname "$PUSH_WORK")"; cp -r "$PUSH_DATA" "$PUSH_WORK"
  # Overlay the committed example's sim config + observation. The staged scene.yaml
  # carries the runtime sources (camera profile ref + initial_ee_pose); the planning
  # loop also reads the scene RGB (propose image).
  # Calibration/EE always come from the example's scene.yaml; the rest is copied only
  # when the assets dir doesn't already provide it (so a user PUSH_DATA wins).
  if [ -f "$PUSH_SCENE/sim/scene.yaml" ]; then cp "$PUSH_SCENE/sim/scene.yaml" "$PUSH_WORK/scene.yaml"
  else printf 'camera: {profile: "0103", cam: 1}\n' > "$PUSH_WORK/scene.yaml"; fi
  for f in camera1_rgb.png camera1_depth.npy; do
    [ -e "$PUSH_WORK/$f" ] || { [ -e "$PUSH_SCENE/capture/$f" ] && cp "$PUSH_SCENE/capture/$f" "$PUSH_WORK/$f"; }
  done
}

# eval output dir for an example: committed dir if --overwrite-examples, else scratch
eval_out() { # $1 = example family, $2 = eval name
  if [ "$OVERWRITE_EXAMPLES" = 1 ]; then echo "$ROOT/examples/$1/$2"; else echo "$OUT_DIR/$1/$2"; fi
}

############################################################
# 1. SANITY CHECKS
############################################################
if [ $DO_TESTS = 1 ]; then
  run_step "imports" $PY -c "from simpact.simulators.mpm import MPM_Simulator_WARP; \
from simpact.executor.mpm_rollout import MPMRollout, SweepRollout; \
from simpact.executor.rope_rollout import ARAPRollout; print('imports OK')"
  run_step "pytest (CPU-safe suite)" uv run pytest tests/ -q
  [ $HAVE_GPU = 1 ] && run_step "smoke_test_mpm (GPU)" $PY scripts/smoke_test_mpm.py \
                    || skip_step "smoke_test_mpm (GPU)" "no CUDA+warp"
  run_step "smoke_test_arap" $PY scripts/smoke_test_arap.py
else
  skip_step "sanity checks" "--skip-tests"
fi

############################################################
# 2. REAL2SIM
############################################################
if [ $DO_REAL2SIM = 1 ]; then
  # Push real2sim mirrors the deformable stage: the committed trial carries a golden
  # reconstruction in sim/, so perception (Grounded-SAM-2 -> SAM-3D -> FoundationPose)
  # REBUILDS from capture/ and is VERIFIED against it. Planning uses the committed sim/
  # unless --full-real2sim (or a user PUSH_DATA) supplies the assets instead.
  _rgb_ok=0; { [ -f "$PUSH_SCENE/capture/camera1_rgb.png" ] || [ -f "$PUSH_SCENE/capture/camera1_rgb.npy" ]; } && _rgb_ok=1
  if ! want push; then
    skip_step "real2sim push (full perception)" "not selected (--tasks=$TASKS)"
  elif [ $RUN_PERCEPTION = 1 ] && [ $_rgb_ok = 1 ] && [ -f "$PUSH_K" ]; then
    run_step "real2sim push: rebuild $PUSH_TRIAL from capture/ (perception)" \
      $PY scripts/run_rigid_pipeline.py --data_dir "$PUSH_SCENE" \
        --objects "$PUSH_OBJECTS" --cam 1 --K "$PUSH_K" \
        --out_dir "$PUSH_BUILD" --xml "$PUSH_BUILD/scene.xml"
    run_step "real2sim push: verify $PUSH_TRIAL vs reference" \
      $PY scripts/verify_scene_build.py --built "$PUSH_BUILD" --reference "$PUSH_SCENE" --material push
    # --full-real2sim plans on the fresh reconstruction (unless the user pinned PUSH_DATA)
    if [ "$R2S_MODE" = full ] && [ -z "$PUSH_DATA" ] && [ -d "$PUSH_BUILD" ]; then
      PUSH_DATA="$PUSH_BUILD"; HAVE_PUSH=1
    fi
  elif [ "$DO_PERCEPTION" = off ]; then
    skip_step "real2sim push (full perception)" "--skip-perception (planning uses the committed sim/)"
  else
    skip_step "real2sim push (full perception)" "external perception models not found (SAM-3D + FoundationPose + SAM2 ckpt; see docs/RIGID_ENV_SETUP.md); planning uses the committed sim/"
  fi
  # Deformable real2sim: rebuild each selected scene from its capture/ (RGB-D + EE
  # record) via build_scene — GSAM2 segmentation + VLM grounding/material-ID — and
  # VERIFY it against the committed sim/ (tolerances absorb VLM variation). With
  # --full-real2sim the rebuilt scene also replaces the committed one for planning.
  build_one() { # $1 task, $2 family, $3 trial
    local ref="$ROOT/examples/$2/$3" build="$OUT_DIR/${1}_build/$3"
    # The SAM2 checkpoint is passed EXPLICITLY (like the push stage does): .env may
    # carry placeholder paths that simpact's dotenv load re-injects into children.
    run_step "real2sim $1: rebuild $3 from capture/" $PY - "$1" "$ref" "$build" "$SAM2_CKPT" <<'PYEOF'
import sys, yaml
from pathlib import Path
from simpact.real2sim.build_scene import build_scene
from simpact.real2sim.perception.grounded_sam2 import GroundedSAM2Segmenter
from simpact.tasks import TASKS
from simpact.utils.layout import find_scene_file
task, ref, build = TASKS[sys.argv[1]], Path(sys.argv[2]), Path(sys.argv[3])
profile = (yaml.safe_load(find_scene_file(ref, "scene.yaml").read_text())
           .get("camera") or {}).get("profile")
seg = GroundedSAM2Segmenter(device="cuda", sam2_checkpoint=sys.argv[4])
r = build_scene(ref / "capture", build / "sim", task.build_material,
                task.build_object_prompt, cam=1, bg_prompt=task.build_bg_prompt,
                ee_pose_path=str(ref / "capture" / "initial_ee_pose.txt"),
                profile=profile, segmenter=seg)
print(f"built {r.scene_yaml} (cloud: {r.cloud_path.name})")
PYEOF
    run_step "real2sim $1: verify $3 vs reference" \
      $PY scripts/verify_scene_build.py --built "$build/sim" --reference "$ref" --material "$1"
    if [ "$R2S_MODE" = full ]; then  # stage capture + rebuilt sim as the planning trial
      local work="$OUT_DIR/${1}_work/$3"
      rm -rf "$work"; mkdir -p "$work"
      cp -r "$ref/capture" "$work/capture"; cp -r "$build/sim" "$work/sim"
    fi
  }
  if [ $HAVE_GSAM2 = 1 ] && [ $HAVE_KEY = 1 ]; then
    want rope  && for s in 11 8; do build_one rope rope_real2sim "1102_rope_$s"; done
    want dough && build_one dough dough_real2sim 1104_sand_6
    want sweep && build_one sweep sweep_real2sim 0118_sweep_0
  else
    skip_step "real2sim deformable (rebuild from capture)" \
      "needs the SAM2 checkpoint + GPU + GOOGLE_API_KEY (planning uses the committed sim/)"
  fi
else
  skip_step "real2sim" "--skip-real2sim"
fi

# Planning scene per trial: the committed reference, unless --full-real2sim staged a
# capture-only rebuild above (missing/failed rebuilds fall back to the reference).
scene_for() { # $1 task, $2 family, $3 trial
  if [ "$R2S_MODE" = full ] && [ -f "$OUT_DIR/$1_work/$3/sim/scene.yaml" ]; then
    echo "$OUT_DIR/$1_work/$3"
  else
    echo "$ROOT/examples/$2/$3"
  fi
}

############################################################
# 3. ACTION PLANNING (closed loop) — needs GOOGLE_API_KEY
############################################################
if [ $DO_PLANNING = 1 ] && [ $HAVE_KEY = 1 ]; then

  # One call per task case — task specifics (rollout class, templates, gates)
  # live in the task registry (simpact/tasks.py). Extra args land AFTER the
  # common ones, so a per-task flag (e.g. push's --max_iters 5) overrides.
  plan_one() { # $1 step label, $2 task, $3 scene, $4 out_dir, extra args...
    run_step "$1" $PY scripts/optimize.py --task "$2" --scene "$3" \
      --max_iters "$MAX_ITERS" --out_dir "$4" "${@:5}"; }

  # 3a. PUSH — the committed trial (sim/ carries the golden reconstruction) unless
  # --full-real2sim adopted the fresh perception build or the user pinned PUSH_DATA.
  if ! want push; then
    skip_step "plan: push" "not selected (--tasks=$TASKS)"
  else
    if [ $HAVE_PUSH = 1 ]; then stage_push; PUSH_PLAN_SCENE="$PUSH_WORK"
    else PUSH_PLAN_SCENE="$PUSH_SCENE"; fi
    plan_one "plan: push" push "$PUSH_PLAN_SCENE" "$(eval_out push_real2sim 0103_push_0/runs)" \
      --align_axis y --align_tol 0.02 --max_iters 5
  fi

  # 3b. ROPE (repo-reproducible, CPU-ok) — both bundled scenes
  if want rope; then
    for s in 11 8; do
      plan_one "plan: rope $s" rope "$(scene_for rope rope_real2sim 1102_rope_$s)" \
        "$(eval_out rope_real2sim 1102_rope_$s/runs)"
    done
  else
    skip_step "plan: rope" "not selected (--tasks=$TASKS)"
  fi

  # 3c. DOUGH (needs CUDA+warp)
  if ! want dough; then
    skip_step "plan: dough" "not selected (--tasks=$TASKS)"
  elif [ $HAVE_GPU = 1 ]; then
    plan_one "plan: dough (multi-grasp square)" dough "$(scene_for dough dough_real2sim 1104_sand_6)" \
      "$(eval_out dough_real2sim 1104_sand_6/runs)" --num_steps "$NUM_STEPS_DOUGH"
  else
    skip_step "plan: dough" "no CUDA+warp"
  fi

  # 3d. SWEEP (needs CUDA+warp)
  if ! want sweep; then
    skip_step "plan: sweep" "not selected (--tasks=$TASKS)"
  elif [ $HAVE_GPU = 1 ]; then
    plan_one "plan: sweep (coverage gate)" sweep "$(scene_for sweep sweep_real2sim 0118_sweep_0)" \
      "$(eval_out sweep_real2sim 0118_sweep_0/runs)" \
      --num_steps "$NUM_STEPS_SWEEP" --min_coverage 0.5
  else
    skip_step "plan: sweep" "no CUDA+warp"
  fi

elif [ $DO_PLANNING = 1 ]; then
  skip_step "action planning (all)" "GOOGLE_API_KEY not set"
else
  skip_step "action planning" "--skip-planning"
fi

############################################################
# SUMMARY
############################################################
summarize
