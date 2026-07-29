# Validation plan — SAM-3D completion → FoundationPose pose, single uv env

Goal: prove that **one uv env** (`torch 2.9+cu128 / py3.11`, RTX 5090) can run
**3D object completion (SAM-3D) from an RGB-D image → 6-DoF pose (FoundationPose)**
end to end, with correctness checked against the recorded golden trials in
the recorded-trial archive (110 trials carry both a golden `_scaled.obj` mesh and a
golden `_6d_cam{id}.txt` pose).

> Status: **executed** (2026-06-22) — single env confirmed, full chain runs end to
> end and writes a MuJoCo scene + a 6-DoF validation overlay. See **Results**
> below. Builds on [PERCEPTION.md](PERCEPTION.md) and
> the worked example is
> [RIGID_PIPELINE.md](RIGID_PIPELINE.md) + `examples/push_real2sim/`.

## Prerequisite gate — FoundationPose weights (to be provided)

FoundationPose's `weights/` dir is **empty**. Inference needs the refiner
(`weights/2023-10-28-18-33-37/model_best.pth` + `config.yml`) and the scorer
network (~1 GB total, from FoundationPose's public release). The build spike only
exercised headers; weighted inference cannot run until these exist.
**Action: user provides the weights or a local path.** Everything else is present:

- SAM-3D checkpoints: `checkpoints/hf/` incl. `slat_decoder_mesh.ckpt` → SAM-3D
  emits a **mesh** (not just a gaussian splat).
- Recorded data is FoundationPose-ready: `camera{id}_depth.npy` is **float metres**
  (no `/1e3`), `camera{id}_mask_{obj}.npy` masks, `cam{id}_K.txt` intrinsics, and
  golden `{obj}_6d_cam{id}.txt` 4×4 poses.

## Why the validation is split (a correctness subtlety)

The golden poses were produced by FoundationPose **using Hunyuan3D meshes**
(`_scaled.obj`). Feeding a *SAM-3D* mesh to FoundationPose yields a pose in a
different canonical frame/scale, so it will **not** equal the golden pose. Hence:

- **FoundationPose correctness** → test with the recorded **golden mesh**
  (apples-to-apples vs the golden pose).
- **Full SAM-3D → FP chain** → validate by **geometric consistency with the
  observation** (silhouette / depth alignment), not by exact golden-pose match.

## FoundationPose call contract (from `run_demo.py` / `estimater.py`)

```python
est  = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                      mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx)
pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)  # 4×4 ob_in_cam
```
- `rgb` uint8 HxWx3; `depth` float **metres** (values <0.001 treated invalid);
  `mask` bool; `K` 3×3 (`cam1_K.txt`); `glctx` = nvdiffrast `RasterizeCudaContext`.
- Recorded depth is already metres → feed directly (do **not** divide by 1e3).
- Use camera **1** (the original `test.sh` used `--cam_id 1`): `camera1_*` + `cam1_K.txt`
  + `_6d_cam1.txt`.

## Phases — harness `scripts/validate_rigid_pipeline.py`

### Phase 0 — single-env build + runtime coexistence smoke (the core proof)
- Build the rigid env (`scripts/setup_rigid_env.sh` from PERCEPTION.md: simpact
  [real2sim] + sam2 + pytorch3d + nvdiffrast + patched mycuda) **plus** SAM-3D
  deps (spconv-cu121, timm, einops; `xformers` flagged — absent from the working
  sam3d env, likely optional) **plus** FP inference deps.
- Smoke: in **one** Python process, instantiate *both* `FoundationPose`
  (refiner+scorer loaded) and SAM-3D `Inference(checkpoints/hf)`, run one tiny
  inference each.
- **Pass:** both load + run in one process; log peak GPU memory (must fit 32 GB).
- This is what the build spike could not show — *weighted* models coexisting at
  runtime, not just compiling.

### Phase 1 — FoundationPose correctness (exact-ish, isolated)
- Inputs: recorded `rgb`, `depth`(m), `mask`, `cam1_K.txt`, **golden
  `{obj}_scaled.obj`** → `est.register(..., iteration=5)`.
- Compare to golden `{obj}_6d_cam1.txt`.
- **Metrics / pass:** ADD-S (symmetric closest-point error) < 0.1·diameter; and
  translation < 2 cm, rotation < 10°. Fix seeds (FP is mildly stochastic).

### Phase 2 — SAM-3D completion sanity (isolated)
- Inputs: recorded `camera1_mask_{obj}_cropped.png` (RGBA) → SAM-3D
  `inference(image, mask, seed=42)` → mesh (mesh decoder) → export `.obj`.
- **Metrics / pass:** non-empty, closed/manifold mesh, plausible bbox aspect;
  log runtime + GPU memory.

### Phase 3 — end-to-end chain (the real goal)
- Chain: `rgb+depth+mask` → SAM-3D mesh → `estimate_scale` (metricise via
  depth+mask+K) → FoundationPose `register` → pose **P**.
- **Geometric consistency** of (SAM-3D mesh, P) vs observation:
  - posed-mesh silhouette **IoU vs mask > 0.7**
  - median |rendered_depth − observed_depth| over mask **< 1.5 cm**
  - chamfer(posed mesh, back-projected masked depth cloud) median **< ~5 mm**
- Soft reference: pose delta vs golden (expected larger than Phase 1).
- **Pass:** chain runs in one env **and** the posed reconstruction aligns with
  the RGB-D observation.

### Phase 4 — coverage + resources
- Run Phases 1 & 3 over ~20–30 of the 110 rigid trials (push/obstacle/pivot/
  stack). Report: FP success rate + ADD-S distribution, chain IoU/chamfer
  distribution, per-stage latency, **peak GPU memory with both models resident**.
- Answers "single env" at scale, not on one example.

## FoundationPose `mycuda` torch-2.9 patch (needed before Phase 0)

`patches/foundationpose_torch29.patch` (applied at setup; upstream clone
untouched) — two mechanical torch-API migrations validated in the spike:
1. `bundlesdf/mycuda/setup.py`: `-std=c++14` → `-std=c++17`.
2. `bundlesdf/mycuda/common.cu`: `.type()` → `.scalar_type()` in the three
   `AT_DISPATCH_FLOATING_TYPES(...)` calls.
Build env: `CUDA_HOME=/usr/local/cuda-12.8`, `TORCH_CUDA_ARCH_LIST=12.0`.

## Deliverables
- `scripts/validate_rigid_pipeline.py` (harness → JSON + markdown report) + opt-in
  pytest gated on `SIMPACT_RIGID_VALIDATION=1` (like `test_golden_trial`).
- `scripts/setup_rigid_env.sh` + `patches/foundationpose_torch29.patch`.
- Results appended here.

## Risks / caveats
- FP needs a **metric** mesh → SAM-3D mesh scaled before FP (golden `_scaled.obj`
  is already metric for Phase 1).
- FP mildly stochastic → fixed seeds, report variance.
- SAM-3D canonical frame differs from Hunyuan3D → Phase 3 uses observation
  consistency, not golden-pose equality.
- Env friction candidates: `xformers` / `bpy` for SAM-3D (both absent from the
  working sam3d env → probably optional) — Phase 0 surfaces them.

---

## Results (executed 2026-06-22, RTX 5090)

Single uv env: `torch 2.9.0+cu128 / numpy 2.4.6 / py3.11`. The end-to-end driver
[`scripts/run_rigid_pipeline.py`](../scripts/run_rigid_pipeline.py) drives the
perception adapters (`GroundedSAM2Segmenter` → `SAM3DReconstructor` →
`FoundationPoseEstimator`). Worked example + bundled data: `examples/push_real2sim/`.

### Phase 0 — single-env coexistence (the core proof)
All three weighted models load and run **in one process**:
- Grounded-SAM-2 ≈ 3 GB, SAM-3D ≈ 20 GB, FoundationPose ≈ 1–2 GB.
- Full pipeline **peak 18.7 GB / 32 GB** (SAM-3D and FoundationPose run in two
  phases — reconstruct all → free SAM-3D → pose all — so they are never
  peak-resident together; loading both at once OOMs the card).
- Smokes: `scripts/smoke_gsam2.py`, `scripts/smoke_sam3d.py`,
  `scripts/smoke_rigid_coexist.py` (all pass).

### Phase 1 — FoundationPose correctness (vs golden mesh + golden pose)
Using the recorded **golden** `_scaled.obj` (apples-to-apples), FoundationPose
reproduces the golden `_6d_cam1.txt` poses essentially exactly — e.g.
`1030_push_8/pringles`: **translation 0.61 mm, rotation 0.25°, ADD-S 0.2 % of
diameter** (PASS). Deterministic across reruns (`set_seed(0)`). Some **cam0**
goldens are themselves bad registrations (object floating ~2 m); a depth-fit check
shows the simpact reproduction is the correct one there.

### Phase 2 — SAM-3D mesh sanity
SAM-3D emits a closed ~200–600k-vertex mesh per object on the mesh decoder path
(`slat_decoder_mesh.ckpt`); ~14 s load + ~18 s/object on the 5090.

### Phase 3 — full SAM-3D → scale → FoundationPose chain (the real goal)
On recorded trial `1030_push_8` (prompt `"pringles. white coconut milk carton."`,
cam1; the committed runnable example is now `0103_push_0`):

| object | detect | SAM-3D mesh | metric scale | FP pose t (cam) |
|---|---|---|---|---|
| white coconut milk carton | 0.71 | 217k verts | 26.5 cm | `[0.024 0.022 0.612]` |
| pringles | 0.50 | 213k verts | 26.2 cm | `[0.114 -0.034 0.718]` |

The chain runs in one env and writes a MuJoCo scene (`build/scene.xml`) that
**loads and steps in MuJoCo** (4 bodies / 2 meshes, no crash).

### Validation by point-cloud overlay (pose + shape)
`scripts/visualize_poses.py` back-projects the RGB-D frame to a 3-D cloud and
overlays each object's completed mesh at its pose + the observed masked points,
with a printed (extrinsic-invariant) mean observed→mesh fit. It defaults to the
**robot base frame** using the aligned packaged extrinsic (`get_camera_to_robot`),
so it also reveals world placement. On `0103_push_0` (recorded with the committed
0103 calibration): blue carton 15.0 mm, white carton 20.2 mm, both upright (<2°)
and co-planar on the table. Usage + example in
[RIGID_PIPELINE.md](RIGID_PIPELINE.md#validating-perception-pose--shape--scriptsvisualize_posespy).

### Validation by 6-DoF overlay
The driver writes **`build/pose_overlay.png`** — each estimated pose drawn back on
the input RGB as FoundationPose's posed 3-D box + XYZ axes (committed reference:
`examples/push_real2sim/pose_overlay_example.png`). A tight box ⇒ correct
pose+scale; an oversized box ⇒ scale error. This overlay caught a real bug.

### Scale fix (found via the overlay)
The depth-bbox-ratio scale initially over-scaled the carton to **64 cm** (≈3×)
because of mask-edge depth bleed onto the table. Removing statistical outliers
from the masked depth cloud (open3d `remove_statistical_outlier`, before measuring
the bbox) corrected it to **26.5 cm**; pringles 27.3 → 26.2 cm. Scale remains a
heuristic — for exact dimensions use an ICP fit to the depth cloud or known sizes.
Objects are also "settled" onto the fitted table plane (lowest vertex at z=0) so
the scene stays physically valid even if scale is slightly off.

### Known limitations
- **Scale** is approximate (above) — validate with the overlay.
- **World frame**: the packaged aligned camera→robot extrinsic (committed 0103
  `optimized_transform*`) gives true robot-frame placement for trials captured with
  that calibration (e.g. `0103_push_0`). Trials recorded earlier (e.g.
  `1111_push_0`) predate it — their stored `_mujoco_cam` poses are ~15–18 cm off
  and tilt/penetrate the table (objects topple in sim); recompute via
  `transform_6d.transform_to_robot_frame` and use a calibration-matched trial.
- **Texture**: SAM-3D meshes use vertex colors, not a UV texture; the XML uses a
  per-object mean color (no `_scaled_0.png` bake yet).
