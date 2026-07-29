# Bundled examples — one recorded trial per task

Four recorded trials, one per release task, each runnable end to end with the
unified driver:

| task | trial | simulator | success signal | needs |
|---|---|---|---|---|
| push  | [`push_real2sim/0103_push_0`](push_real2sim/0103_push_0/) | MuJoCo (`PushSceneRollout`) | measured `alignment_gate` AND VLM validity | GL, API key |
| rope  | [`rope_real2sim/1102_rope_{8,11}`](rope_real2sim/) | ARAP (`ARAPRollout`) | VLM-only | CPU ok, API key |
| dough | [`dough_real2sim/1104_sand_6`](dough_real2sim/1104_sand_6/) | MPM (`MPMRollout`, multi-grasp) | VLM-only | CUDA + warp, API key |
| sweep | [`sweep_real2sim/0118_sweep_0`](sweep_real2sim/0118_sweep_0/) | MPM (`SweepRollout`) | measured `coverage_gate` AND VLM validity | CUDA + warp, API key |

Task specifics (rollout class, prompts, allowed primitives, gates, build prompts)
live in the task registry, [`simpact/tasks.py`](../simpact/tasks.py).

## The trial layout

Every trial separates its files by role (resolved by `simpact/utils/layout.py`;
external flat trial dirs also work everywhere):

```
<trial>/
  capture/   the raw recording — the self-sufficient entry point
    camera1_rgb.png       real photo (propose image + projection reference)
    camera1_depth.npy     raw depth, metres (real2sim rebuild input)
    initial_ee_pose.txt   recorded EE pose at capture (4x4, robot frame)
  sim/       simulation assets — all a rollout ever reads
    scene.yaml            THE runtime config: camera: {profile} -> assets/calibration/
                          registry, initial_ee_pose (4x4), and per-material fields
    <geometry>            segmented_object.ply (rope) / mpm_points.npy (dough) /
                          beans_mpm_points.npy + target_region.ply (sweep) /
                          {obj}_scaled.obj + texture + {obj}_6d_cam1.txt +
                          {obj}_mujoco_cam1.txt (push — the golden reconstruction)
  runs/      recorded closed-loop outputs (the committed reference results)
    propose.json          initial VLM proposals
    rollouts/             every attempt (rollout_NN.json + renders) with its verdict
    refined_plan.json     the chosen plan
    final_rollout/        the chosen plan re-rolled + INDEPENDENTLY re-verified
    run_log.txt           full console log of the recorded run
```

Simulation never reads `capture/` (enforced by `tests/test_scene_schema.py`, which
builds a rollout from a trial copy with `capture/` deleted). `capture/` is
provenance + the rebuild input; `sim/` is the runtime source.

## The lifecycle: capture → sim → runs

**1. Build `sim/` from `capture/`** (what an end user runs on a fresh recording):

```bash
# deformables — segmentation + back-projection + VLM grounding/material-ID
uv run python -m simpact.real2sim.build_scene \
  --raw-dir <trial>/capture --out-dir <trial>/sim \
  --material rope|dough|sweep --object "<segmentation prompt>" \
  [--bg "<target-region prompt>"] --profile <calibration profile>

# push — the rigid perception pipeline (Grounded-SAM-2 -> SAM-3D -> FoundationPose)
uv run python scripts/run_rigid_pipeline.py \
  --data_dir examples/push_real2sim/0103_push_0 \
  --objects "white coconut milk carton. blue milk carton." --cam 1 \
  --K assets/calibration/0103/cam1_K.txt \
  --out_dir examples/push_real2sim/0103_push_0/build   # gitignored; ~1.5 min on a 5090
```

Every trial's `sim/` is committed — including push's golden reconstruction
(SAM-3D textured meshes + FoundationPose 6-DoF poses, ~9 MB) — so **all four
loops run from the repo alone** (step 1 is only needed for fresh captures).
`reproduce_all.sh` treats every build step as **rebuild + verify**: when the
models are detected it re-runs perception/`build_scene` from `capture/` and
checks the result against the committed `sim/` (`scripts/verify_scene_build.py`).

**2. Run the closed loop** (propose → rollout → verify ↔ regress → confirm):

```bash
uv run python scripts/optimize.py --task rope \
  --scene examples/rope_real2sim/1102_rope_11 --out_dir /tmp/rope_loop
MUJOCO_GL=egl uv run python scripts/optimize.py --task dough \
  --scene examples/dough_real2sim/1104_sand_6 --out_dir /tmp/dough_loop
MUJOCO_GL=egl uv run python scripts/optimize.py --task sweep \
  --scene examples/sweep_real2sim/0118_sweep_0 --out_dir /tmp/sweep_loop
MUJOCO_GL=egl uv run python scripts/optimize.py --task push \
  --scene examples/push_real2sim/0103_push_0 --out_dir /tmp/push_loop
```

**3. Reproduce everything** — `bash scripts/reproduce_all.sh` runs the tests, the
push perception, a **rebuild of every deformable scene from its `capture/`**
(verified against the committed `sim/` by `scripts/verify_scene_build.py` —
tolerances absorb VLM variation), and all four loops. `--full-real2sim` plans on
the rebuilt scenes instead of the committed references — the end-user
capture-only chain, end to end.

Calibration resolves **per scene** from `sim/scene.yaml`'s `camera: {profile}`
through the registry in [`assets/calibration/`](../assets/calibration/) — never a
code default. The rebuild+verify stage guards the assignment: it caught this
repo's one latent calibration bug (see the sweep note below).

---

## Per-task notes

### push — `0103_push_0` (white + blue milk cartons)

The goal is **horizontal alignment**: push the white carton side by side with the
blue one, `|Δy| ≤ 2 cm`, without disturbing blue. Success = **measured
`alignment_gate` AND VLM validity** — the top-down camera foreshortens depth so
the VLM eyeballs alignment wrong (the gate measures it), while the VLM catches
what the gate can't see (topples, collateral motion, catastrophes).

- Object world poses are recomputed from the **aligned per-scene extrinsic**
  (`get_camera_to_robot @ 6d_cam`), *not* the trial's stale `_mujoco_cam` files;
  each base is snapped onto the table. This trial was recorded with the committed
  `0103` calibration, so both cartons land upright (<2°) — pre-0103 trials are
  ~15–18 cm off and topple (docs/VALIDATION_rigid.md).
- The gripper starts at the trial's **real recorded EE pose** (right beside the
  white carton — what makes a single push reach it) and is **teleported there at
  init** (`snap_to_mocap`); without the teleport, the mocap weld drags the hand
  across the scene and knocks the carton over before any action runs.
- Perception validation: the build writes `pose_overlay.png` (posed 3-D boxes +
  axes on the RGB) — a tight box means pose AND metric scale are right. Committed
  golden: [`push_real2sim/pose_overlay_example.png`](push_real2sim/pose_overlay_example.png).
- The recorded run ([`runs/`](push_real2sim/0103_push_0/runs/)) shows both
  feedback channels working: two candidates *reached alignment but toppled the
  carton* (VLM veto of a gate pass), a refined plan stayed upright but stopped
  3.5 cm short (gate veto with measured feedback), and the next refinement
  **SOLVED** it (|Δy| = 1.3 cm, upright, blue undisturbed) — independently
  re-verified → `DEMO_OK`.

### rope — `1102_rope_{8,11}` (ARAP, quasi-static)

Goal: drag the rope's free end into a **U-shaped curve** (a single smooth bend —
not a *symmetric* U, which free-end dragging cannot produce; the verifier judges
accordingly). Success is VLM-only (no measured shape gate this release).

- **Free-end-only grasp**: the fixed end is anchored, so `ARAPRollout` pins the
  grasp to `scene.yaml`'s `free_end` — the VLM effectively only chooses where to
  drag. Endpoints are grounded by the VLM pipeline (`simpact/generator/ground.py`)
  or the original human picks (`endpoint_source` records which).
- Recorded runs: `1102_rope_11` **SOLVED in 0 regress iterations**;
  `1102_rope_8` **SOLVED in 2** — its candidates made genuine S-curves (correctly
  rejected: *"inflection point, two opposite bends"*), and the optimizer worked
  through an L-shape to a clean U.

### dough — `1104_sand_6` (MPM, multi-grasp squeeze)

Goal: a **square block** (dx ≈ dy footprint) via squeezes from **two
perpendicular directions** — each squeeze flattens the side pair perpendicular to
the jaw yaw, so yaw 0 then yaw ≈ 90° flattens both. `MPMRollout` applies 1..N
squeezes in ONE continuous sim (jaws repositioned per grasp; deformation
persists). Success is VLM-only.

- Material physics (E/ν/ρ/yield) is **VLM-estimated per scene** at build time and
  stored in `scene.yaml`'s `material:` block — never a hand-assigned default.
- An earlier verifier also demanded a *clean, regular* block; across 15 attempts,
  11 reached near-square dimensions but 0 passed — off-centre squeezing leaves a
  lumpy blob, a structural ceiling more VLM iterations can't fix. That regularity
  bar was removed; the criterion is square **proportions**.
- Recorded run: **SOLVED in 0 regress iterations** — two perpendicular squeezes,
  bbox 0.049 × 0.050 (dx/dy = 0.97), conf 1.00 → `DEMO_OK`.
- Rollouts are ~1–2 s each after warp's one-time ~30 s kernel compile.

### sweep — `0118_sweep_0` (MPM, pusher + the first measured deformable gate)

Goal: sweep the bean pile into the taped target region. Success = **VLM-valid AND
`coverage_gate`** — the fraction of final particles whose (x, y) falls inside the
target region's hull, computed in robot frame (immune to the render viewpoint).
The beans start at 0% coverage.

- **Blade orientation is prompt-enforced**: a flat pusher only moves the pile as
  one mass **broadside**; edge-on it *knifes through* and splits it — and the
  sticky sim collider still drags beans in, so coverage alone can't catch it. The
  regress prompt ties blade yaw to push direction; the verify prompt's `together`
  check rejects knifed-through sweeps (in one run it correctly rejected all three
  83–87%-coverage candidates as *"smeared into streaks"*).
- **Calibration note**: this January-18 trial originally referenced the October
  `1026` extrinsic (picked by a 2-D mask-centroid match — blind to the depth
  axis). Rebuilding from `capture/` exposed a systematic ~6–7 cm z-bias on beans
  AND tape; the January-3 **`0103`** profile reproduces the committed clouds to
  <0.5 cm, so `scene.yaml` now references `0103`. The verify stage guards this
  from recurring.
- Recorded run: **SOLVED in 0 regress iterations** — a broadside `+y` sweep,
  **100% coverage**, VLM `together` pass → `DEMO_OK`.

---

## Rendering & rollout videos

Rope renders as a matplotlib projection overlay on the real photo; dough/sweep
render headless via **PyVista** (`simpact/executor/render_deformable.py`): 3-D
shaded sphere glyphs with the jaws/blade depth-composited at the real camera1
pose, image size from the calibration profile; push renders overhead MuJoCo
screenshots. **Every rollout also writes a full-simulation `rollout_NN.mp4`** —
a debugging artifact the VLM never sees (it gets only the before/after PNGs).

To regenerate any recorded plan's video **without replanning** (no VLM calls):

```bash
MUJOCO_GL=egl uv run python scripts/replay_rollout.py --task push \
  --scene examples/push_real2sim/0103_push_0 \
  --plan  examples/push_real2sim/0103_push_0/runs/refined_plan.json \
  --out_dir /tmp/push_replay \
  --check examples/push_real2sim/0103_push_0/runs/final_rollout/rollout_00.json
```

The MuJoCo push replay is exact (`--check` asserts the final object positions
match the recorded rollout); MPM/ARAP replays reproduce the recorded outcome to
sub-centimetre. The committed `final_rollout/rollout_00.mp4` files were produced
this way.
