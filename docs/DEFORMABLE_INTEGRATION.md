# Integrating deformable objects into the closed loop

Design record for extending simpact's rigid closed-loop optimizer (propose →
rollout → verify ↔ regress, see [EVALUATION.md](EVALUATION.md)) to **deformable
manipulation**: rope (ARAP), and sand / dough / bean-sweep (MPM). Written as the
implementation plan and kept as the design rationale; dated status notes record
when each slice landed. Grounded in the original real-robot experiments this
release reproduces offline.

---

> **Status — rope closed loop runnable (2026-07-06).** The full rope path is wired
> end to end and runs: `scripts/optimize.py --task rope` composes the shared loop
> (`VLMProposer` → `ARAPRollout` → `TaskVerifier` ↔ `RegressOptimizer` inside
> `OptimizationLoop`). New pieces: `simpact/executor/rope_rollout.py` (`ARAPRollout` —
> accepts both propose-primitive and `move`/`gripper_control` plans, solves ARAP,
> renders, emits the rollout-JSON envelope), `regress.parse_rope_rollout`,
> `prompts/{regress,verify}/rope.txt`. A recorded run is at
> `examples/rope_real2sim/1102_rope_11/runs/`: the verifier gives shape-aware feedback
> ("asymmetric J", "S-curve", "shallow C — grasp the midpoint and drag +x") and the
> optimizer incorporates it, ending best-effort (rope is VLM-only — §5 — and a single
> grasp-drag rarely yields a perfectly symmetric U). Run:
> `MUJOCO_GL=egl python scripts/optimize.py --task rope --max_iters 3`.
> (`gemini_generate` also gained a per-request timeout so transient TLS flakiness
> fails fast instead of the client's retry looping for minutes.)

## 1. The loop is already generic — the gap is narrow

simpact's rigid loop is parameterized exactly where deformables need to differ, and
**both deformable simulators are already ported and smoke-tested**. Every
deformable-specific behaviour plugs into an existing seam:

| seam (simpact) | deformable use |
|---|---|
| `OptimizationLoop(rollout_fn=, parse_rollout=)` (`generator/optimize_loop.py`) | inject a deformable rollout + parser; **loop core unchanged** |
| `RegressOptimizer(prompt_template=, parse_rollout=)` (`generator/regress.py`) | rope/sand regress template + deformable rollout parser |
| `TaskVerifier(prompt_template=, success_gate=)` (`generator/verify.py`) | deformable validity prompt + (optional) measured gate |
| `VLMProposer(prompt_template=).propose(…, allowed_types=)` (`generator/propose.py`) | **shared with rigid** — only the context template differs |
| `build_context(…)` rope/MPM branches (`generator/context.py:151-172`) | **already ported** |

### Ready vs missing

**Ready (ported, smoke-tested):**
- Simulators: `simpact/simulators/arap/` (`EmbedDeformGraph.solve_global_local`) and
  `simpact/simulators/mpm/` (`MPM_Simulator_WARP`, warp-guarded).
- Asset-prep: `real2sim/prepare_dough_asset.py` (its `sample_between_surface_and_table_columns`
  column-fill is what `build_scene` uses for MPM points). The old standalone
  `prepare_rope_asset.py` / `sample_mpm_points.py` were removed (superseded by
  `detect_rope_endpoints` + `generator/ground.py`, and by the column-fill above).
- Context branches + templates: `context.py` rope (`fixed_point`/`free_end`) and MPM
  (`init_mpm_center` + `bg_pcd_path`→target-centre); `prompts/contexts/rope.txt`,
  `dough.txt`, `sweep.txt`.
- Propose stage (shared with rigid) + the whole loop core, final-confirmation,
  memory accumulation, verdict feedback.
- Back-projection + extrinsics (`scripts/visualize_poses.py::backproject`,
  `transform_6d.get_camera_to_robot`) and segmentation adapters
  (`real2sim/perception/grounded_sam2.py`, `sam3d.py`).

**Missing (the actual work):**
- Deformable **rollout drivers** in `simpact/executor/` (only rigid `MuJoCoRollout`
  exists).
- A **rendering path** for the before/after screenshots (see §4 — the original used Klampt).
- **Rope/sand regress + verify templates** and deformable **rollout parsers**
  (`prompts/regress/` has only `primitive.txt`; `parse_rigid_rollout` is rigid-only).
- An **optional measured success gate** (the original had none — see §5).

---

## 2. The rollout-JSON contract (how deformables reuse verify/regress unchanged)

`verify.py` / `regress.py` read `snapshots[*].{screenshot, objects, gripper}` +
`object_names`; `alignment_gate` / `rollout_displacements` read
`objects[<name>].position`. Deformable rollouts should write the **same envelope** so
the before/after screenshots flow through the VLM path unchanged, and store the
deformable state as:
- `objects[<name>].position` = **cloud centroid** (keeps `rollout_displacements`
  working), plus
- a material block: `final_keypoints` (rope) or `final_points_path` + bounding box
  (MPM).

A custom `parse_rollout` builds the regress text from that block; any future
`success_gate` reads it. This keeps the loop, verifier, and regressor code untouched.

---

## 3. Per-material rollout drivers (the real new code)

### Rope — `ARAPRollout` (quasi-static; recommended first slice)
Wrap `EmbedDeformGraph`:
- Build the graph once (voxel-downsample nodes, radius edges, `set_handle_idx`).
  Handles = fixed blob (nodes within `grip_radius≈0.03` of `scene.yaml` `fixed_point`)
  + grasp blob (nodes near the plan's grasp point).
- "Run" a plan = translate the grasp blob by (place − grasp) (+ optional yaw) →
  `solve_global_local(...)` to convergence. **No time loop.**
- Record `final_keypoints` + centroid; render before/after (RBF-skinned vis cloud +
  table; gripper not drawn).
- New dep: **`pypose`** (ARAP local step) — add to the deformable extra.

### Sand / dough / sweep — `MPMRollout` (time-stepped, GPU)
Wrap `MPM_Simulator_WARP`:
- Load particles from `_mpm_points.npy`, apply the **`+0.5` y grid-shift** (undo on
  save), `add_surface_collider` (table), `set_parameters_dict` (material params —
  externalize the per-script magic numbers into a config).
- Tool = one or two boxes as `add_rotate_box_collider` (sticky BC): two fingers
  (squeeze) or one pusher (sweep). Drive with the per-waypoint twist; step
  `p2g2p` ~100–500×.
- Record final particle cloud + bbox + centroid; render before(tool)/after(tool
  removed) (+ `bg_pcd` overlay for sweep). Multi-step = several waypoints in **one
  continuous rollout** (state persists; not re-planning).
- Dep: **warp-lang 1.10.0** (already pinned; `warp.torch` needed), CUDA mandatory.

Both live beside `executor/rollout.py`, expose the same `run(plan,…) -> RolloutResult`
+ `.save()`, and are wired by a `do_rollout` closure in the driver (since unified
into `scripts/optimize.py --task {rope,dough}`).

---

## 4. Design decision — rendering (Klampt vs open3d vs **PyVista**)

The VLM's *entire* success signal for a deformable rollout is one final screenshot,
so render fidelity matters more here than in the rigid loop (which had a measured
alignment gate). The original rendered both deformables with **Klampt `vis.screenshot()`** +
hardcoded `*_viewport.txt` camera files.

**All three candidate renderers are already simpact dependencies** (`klampt`,
`open3d`, `pyvista` in `pyproject.toml`) — so this is *not* a dependency decision. It
is about headless-offscreen reliability, fidelity, and code reuse:

| option | reuse | fidelity | headless offscreen |
|---|---|---|---|
| **PyVista** (VTK) | the original already had PyVista sand renderers (`vis_sand_grasp.py`, `vis_sand_sweep.py`) to adapt | **highest** — particles as sphere glyphs, tool as a mesh, lighting | needs `pv.start_xvfb()` (Xvfb) or an EGL/OSMesa VTK build — the one real risk |
| open3d | none (but used everywhere in-repo) | flat point splatting | `OffscreenRenderer` needs an EGL context; also headless-finicky |
| Klampt | most faithful port (the original system's actual path) | fine | GUI-first; offscreen is the least pleasant of the three |

**Verdict — PyVista is the preferred option**, contingent on validating headless
offscreen early. Rationale: it is already a dependency with existing PyVista
particle-render code to adapt, and its glyph/mesh fidelity best serves the
single-screenshot success signal. `pv.Plotter(off_screen=True).screenshot()` returns
a numpy `HxWx3` array directly, integrating cleanly into the rollout's frame list, and
camera control (`plotter.camera_position = [pos, focal, up]`) reproduces a canonical
viewport without the original system's Klampt viewport files.

**The one thing to de-risk first:** VTK offscreen on the headless RTX 5090 box. Keep
the renderer behind a `render_deformable(points, colors, tool_boxes, camera) -> uint8`
abstraction so open3d remains a drop-in fallback if VTK headless proves troublesome.

> **Pilot result (rope visualization pilot, 2026-07-05; the one-off pilot script
> was removed in the release cleanup).** De-risked on recorded
> rope examples (bundled at `examples/rope_real2sim/scene_1102_rope_{8,11}`). An ARAP
> rollout of each folder's existing `propose.json` was rendered three ways, and **all
> worked headless on the 5090**: (1) a matplotlib projection overlay on
> `camera1_rgb.png` — the initial rope projects *exactly* onto the real rope
> (validates the `cam1→robot` extrinsic + `cam1_K`); (2) **open3d `OffscreenRenderer`
> via EGL directly** (no Xvfb) at the exact camera pose; (3) **pyvista `off_screen`** —
> visibly higher fidelity (sphere glyphs read as a real rope). pyvista wins on
> fidelity, open3d on exact-camera-pose simplicity + zero Xvfb.
>
> Two lessons that generalize to every deformable rollout: **(a)** the proposal deltas
> are relative to the EE pose in **`context.txt`** (what the VLM was shown), *not*
> `scene.yaml`'s `init_gripper_pose` — using the context pose, `1102_rope_11`'s
> proposal grasps the actual `free_end` and forms a clean symmetric U (energy 13 vs a
> snapped 67); **(b)** the renderer camera **must be driven by the scene's real
> extrinsic+intrinsics**, not a hand-set view angle — an arbitrary `view_vector` made
> the first pyvista render appear rotated relative to the real observation.

> **Implemented — PyVista is the MPM renderer (2026-07-06).** `simpact/executor/
> render_deformable.py` renders the material as **3D shaded sphere glyphs**
> (`render_points_as_spheres`, uniform color — no per-point colormap) and the tool
> (jaws / pusher) as a translucent 3D box, **depth-composited** by VTK so the tool's
> depth relative to the material is correct (the earlier flat 2-D matplotlib projection
> drew the tool on top regardless of depth). The camera is built from the scene's real
> `K` + `cam_to_robot` (position, optical axis, up, vertical FOV from `fy`, principal
> point via window-centre). Each rollout also writes a **full-simulation `rollout_NN.mp4`**
> (cv2/mp4v — the jaws closing / the pusher sweeping) as a debugging artifact that is
> **not** fed to the VLM loop (which sees only the before/after PNGs). Rope still uses its
> own matplotlib projection; migrating it to this shared renderer is optional cleanup.

---

## 5. Design decision — goal / success signal

**the original system had no measured success metric for any deformable task** (confirmed across the
whole tree — no chamfer / EMD / IoU / coverage anywhere). Deformable "success" is
Gemini judging one final screenshot against a natural-language instruction. So:

- **Faithful first pass:** `TaskVerifier(success_gate=None)` → verdict = VLM validity
  only (material moved, tool didn't wreck the scene, shape plausibly matches). The
  loop still runs and iterates on VLM feedback.
- **The measured-gate seam is already open** (`SuccessGate = Callable[[path],
  (bool, str)]` in `verify.py`) — this is where the rigid `alignment_gate` plugs in.
  For deformables:
  - **Sweep is the natural first measured gate**: its `bg_pcd` target region already
    exists, so *coverage* (fraction of final beans inside the target region) or a
    chamfer to the region is computable — a real closed-loop metric.
  - **Shaping tasks (T / square)** have **no target shape in the original system**. A measured gate
    there is net-new and requires introducing a target representation (a target
    particle cloud, or deriving target descriptors from the instruction). This is a
    genuine research extension, out of scope for a faithful first port, but the seam
    is designed for it.

Recommendation: ship VLM-only first (faithful), then add the sweep coverage gate as
the first measured deformable success signal.

---

## 6. Design decision — rope endpoint grounding (fixed / free points)

the original set a rope's `fixed_point` (anchor) and `free_end` (graspable) via an
**interactive Open3D shift-click** on the 3-D cloud (`prepare_rope_asset.py`, two
human clicks). **Proposal: replace this with automated grounding in image space** —
feasibility is **HIGH**, and it removes the only human-in-the-loop step in rope
scene setup.

Why it is tractable in simpact specifically:
- **2-D → 3-D lifting is already solved in-repo.** A pixel `(u,v)` →
  `backproject` at `(u,v)` with depth + `K` → camera frame → `get_camera_to_robot`
  → robot frame. All machinery exists (`scripts/visualize_poses.py`, `transform_6d`).
- **The precision bar is lenient.** The sim uses a `grip_radius≈0.03` (3 cm) blob
  around each point, so a point within ~2–3 cm suffices — far easier than a precise
  6-DoF grasp.
- **Rope endpoints are salient** — "the two ends of the rope" is an easy grounding
  target for a modern VLM (Gemini, already used).

**Recommended approach — hybrid grounding (geometry for *where*, VLM for *which*):**
1. Segment the rope with the already-present **Grounded-SAM-2** adapter
   (`perception/grounded_sam2.py`) → binary mask.
2. **Skeletonize** the mask (`skimage.morphology.skeletonize`) → the two skeleton
   endpoints are the geometric rope ends (robust, no VLM localization noise).
3. Assign **fixed vs free** semantically — from the instruction ("grab the free
   end…") or a one-shot VLM call. This is a binary choice the VLM handles well.
4. Back-project both endpoints (masked-neighbourhood **median** depth — a thin rope's
   single-pixel depth is noisy) → `fixed_point`, `free_end`; cache into `scene.yaml`
   exactly like the current picker output (keeps runs reproducible).

Pure VLM pixel-click (skip step 2) is also feasible given the 3 cm tolerance, but is
noisier and more fragile at thin-rope depth edges; the hybrid is more robust for the
same effort and **reuses existing perception**.

**Caveats / risks:**
- Depth at a thin-rope endpoint is often missing / edge-bleed → masked-neighbourhood
  median (the existing masked `backproject` already supports this).
- Self-crossing / looped rope makes skeleton endpoints ambiguous; the VLM can
  disambiguate, or fall back to the human picker for pathological scenes.
- Which end is "fixed" is genuinely task/scene-dependent (the original had a human decide);
  automating requires the instruction or a physical cue (clamp) to define it.
- VLM output is stochastic — but within the 3 cm blob tolerance that is fine, and
  caching to `scene.yaml` fixes it per scene.

**Upside beyond asset-prep:** the same "propose a point in image space → back-project"
pattern is a cleaner action parameterization for the rope **grasp/place points**
themselves (the original encoded them as `PUSH` deltas from the initial EE pose). A
click-based rope action space would unify asset-prep grounding and grasp proposal, and
map directly onto the ARAP handle mechanism.

---

## 7. Action-schema translation (a load-bearing gotcha)

Both deformable pipelines split the schema between propose and regress; the rollout
driver must accept **both** (as `proposal_to_waypoints` was extended to accept
`Move`/`GripperControl` for rigid):
- **Rope:** propose emits `PUSH/DESCEND/GRASP/RELEASE` (→ 2-D grasp point + 2-D place
  point + yaw; z / width / descend are sim no-ops); regress emits
  `move`/`gripper_control`.
- **Sand:** propose emits primitives; regress emits a single `"gripping action"`
  object with parallel `gripper_centers` / `gripper_yaws` / `gripper_widths` arrays.

Plan a per-material `plan → tool-transform-sequence` parser, and port it with tests —
the original system's coupling is filename/array-index fragile.

---

## 8. Port hygiene (must-do on any copy from the original experiments)
- **Strip hardcoded `AIzaSy…` keys** from every `regress_gemini_*` / `propose_gemini_*`
  before porting (Phase-0 `git grep "AIzaSy"` gate); route through `get_gemini_client`.
- **Remove `import pdb; pdb.set_trace()`** left in `prepare_dough_asset.py`,
  `shape_kinetic_sand.py`, and all regress scripts (they hard-block automation).
- **Guard `franky`** in asset-prep (it instantiates `Robot(host)` only to read
  `O_T_EE`) — make EE-pose capture offline/injectable.
- **Eliminate `/home/ydu/haowen/…`** in argparse defaults, `scene.yaml` paths, and
  viewport files → `get_data_dir()`.
- Keep `warp_mpm/mpm_solver_warp.py` and the sand/rope templates **verbatim**
  (do-not-touch list); copy-don't-modify.

---

## 9. Recommended sequencing
1. **Rope vertical slice first** — smallest, quasi-static (no warp/GPU step loop), one
   ARAP solve per proposal. Proves the whole deformable path (rollout driver +
   renderer + parser + regress template) end-to-end through the *unchanged* loop,
   VLM-only success. Deliverable: a recorded rope `runs/` output like the rigid example.
   Bundle the rope endpoint grounding (§6) here.
2. **Shared `render_deformable` (PyVista)** — harden headless offscreen as part of
   slice 1; reuse for MPM.
3. **MPM sand/dough squeeze** — two-box collider + time-stepped `p2g2p`.
4. **Sweep** — single pusher; add the **first measured deformable gate** (coverage /
   chamfer vs `bg_pcd`), demonstrating the closed loop with a real metric.

---

## 10. Open questions / risks
- ~~**Headless VTK offscreen** (§4) — the top de-risking item for PyVista.~~
  **Resolved** by the rope pilot (§4): open3d (EGL) *and* pyvista (`off_screen`) both
  render headless on the 5090. The endpoint-grounding insight from the pilot also
  confirms §6: the VLM proposal's "rope centre" was the straight-line midpoint of the
  endpoints, **8.3 cm off the curved rope** — so a grasp point must be snapped to the
  actual rope (nearest node / skeleton), not taken as the endpoint midpoint.
- **No measured signal for shaping tasks** (§5) — rope/sand shaping stays VLM-judged
  unless a target representation is introduced.
- **MPM cost** — ~500 `p2g2p` steps × up to 8 rollouts × N regress iterations is
  GPU-heavy; use fewer substeps in the loop than in a render/export pass.
- **Schema-translation correctness** (§7) — port with tests.

---

## 11. Rope loop refinements — free-end grasp, goal wording, shape-readable render

Findings from running the rope loop on `1102_rope_{8,11}` (`1102_rope_11/runs/`):

- **The world-frame description is correct.** Projecting each axis into the camera1
  image confirms the context's `+x: down / out of screen, +y: right, +z: up` exactly
  (+x→+30px down, +y→+42px right, +z→−31px up), and the sim render uses the *same*
  camera1 pose. The VLM reasons about the frame correctly — the frame is **not** the bug.
- The real failure modes are: (i) the VLM grasps the **midpoint** (geometrically
  sound for a symmetric U, but not graspable on the real robot) or drags the free end
  **onto the fixed end** (collapsing the rope to a line); and (ii) the verifier
  **misreads the shape** from the 53°-oblique render (a single-bend arch read as an S,
  see §3/§5).

Three coupled changes (implement together):

**(b) Free-end-only grasp — structural (recommended).** On the real robot the fixed
end is anchored; only the free end is graspable — this is the original system's task ("grab the *free
end* and arrange the rope to a U"). So reformulate the action from "pick a grasp point
+ a drag" to **"decide where the free end goes."** The grasp is *defined* as the
scene's `free_end`; the VLM supplies only the drag/destination. In `ARAPRollout`:
override `grasp = free_end` and apply the plan's post-grasp drag delta to it
(`place = free_end + (parsed_place − parsed_grasp)`), gated by a `free_end_only=True`
flag. This makes midpoint / off-rope grasps **structurally impossible** and removes the
grasp-snapping hack — the free end is always a real rope point.

**(a) Belt-and-suspenders rule (prompt).** State it explicitly in the instruction and
the rope regress/verify prompts: *"Only the free end of the rope (at `free_end`) may
be grasped; the other end is fixed and cannot be grasped."* A soft backstop to (b).

**Goal wording.** Free-end-only grasping **cannot** make a perfectly *symmetric* U
(that needs a midpoint pull), so keeping "symmetric U" guarantees the verifier
(correctly) rejects every attempt for asymmetry. Change the goal from "symmetric U"
to a **"U-shaped curve"** — a single smooth bend; asymmetry is acceptable — in the
default instruction and the verifier's SHAPE check.

**Pair with a shape-readable render (§3/§5).** So the verifier can actually judge the
result: render **top-down** (bird's-eye XY of the table plane) instead of the oblique
camera, draw the rope as a **connected curve** (not scattered points), mark the fixed
vs free end, and (robust) pass measured shape descriptors — **inflection count** (U≈0,
S≈1), symmetry, end-to-end/arc-length ratio — as text evidence, en route to a measured
shape gate (§5). The oblique→top-down re-render of `rollout_02` turned the VLM's "S"
into an obvious single-bend arch, so this is the highest-leverage render fix.

Order: land (b) + (a) + goal wording first (cheap, in `ARAPRollout` + the rope
prompts), then the top-down render, then measured shape evidence/gate.

---

## 12. MPM dough / sand slice — grounded implementation plan

Concrete plan for the **next** deformable slice (§9 step 3): dough manipulation via
MPM, mirroring the landed rope slice. Recipe below is grounded in a read of the original
`executor/shape_kinetic_sand.py` (not a summary) so the port is deterministic.

> **Status — dough squeeze slice landed (2026-07-06); unified multi-grasp (2026-07-07).**
> The dough rollout runs the shared loop (`scripts/optimize.py --task dough`:
> `VLMProposer` → `MPMRollout` → `TaskVerifier` ↔ `RegressOptimizer`) and is **multi-grasp
> by default**: `MPMRollout` (in `simpact/executor/mpm_rollout.py`) applies a *list* of
> 1..N squeezes (`grasps_from_plan`) in ONE continuous MPM sim — the two jaw colliders are
> added once then repositioned per grasp via `set_collision_params`, and the dough state
> persists across grasps. **A single squeeze is just the N=1 case — there is no separate
> single-squeeze path** (the original single `MPMRollout` + `gripper_transform_from_plan`
> were removed and the multi-step class promoted to `MPMRollout`). Recipe ported verbatim
> from `shape_kinetic_sand_multi_step.py`: the `+[0,0.5,0]` grid shift, coherent plasticine
> (E=5000, ρ=1200, yield 1000), a 15 cm sticky jaw box, every grasp at a fixed
> `grasp_height`. The §7 schema split is handled by `grasps_from_plan`, which turns **both**
> propose-primitives (N× PUSH/ROTATE/GRASP) and the regress output (one
> `move`+`gripper_control` pair per squeeze) into the same ordered grasp list — so simpact
> keeps its universal schema instead of the original system's bespoke `gripper_centers[]/yaws[]/widths[]`
> arrays. Pieces: `render_deformable.py` (PyVista sphere renderer + mp4),
> `parse_mpm_rollout`, `prompts/{regress,verify}/dough.txt`. The default dough goal is a
> **square block via perpendicular squeezes** — the regress prompt tells the VLM to
> alternate the jaw yaw ~90° between squeezes and equalize dx/dy, and the verifier's SHAPE
> check passes only for a roughly-square footprint (dx ≈ dy, flat sides). A smoke test
> confirms two perpendicular squeezes take the footprint from dx/dy ≈ 2.25 (elongated)
> toward ≈ 1 (square), and the recorded `examples/dough_real2sim/` run converges to a
> 1.04:1 square in 1 regress iteration. After warp's one-time compile (~30 s) each rollout
> is **~1–2 s**. Tests: `tests/test_mpm_rollout.py`
> (grasp-list parser + GPU-gated N=1 and N=2 rollouts). **Remaining in §12:** none — all
> both MPM rollouts (dough multi-grasp + sweep) are in.

> **Status — sweep slice + first measured gate landed (2026-07-06).** Slice 3 (step 3
> below) is wired: `SweepRollout` (same module) ports `sweep_sand_multi_step.py` — a
> single thin pusher box moved segment by segment with a linear twist velocity in one
> continuous MPM sim, shoving a coherent plasticine pile (E=10000, ρ=600, yield 4000)
> toward a target region. `sweep_segments_from_plan` parses both schemas (PUSH/DESCEND/
> ROTATE and `move`) and floors horizontal pushes to the table height so a sweep always
> contacts the pile. **The headline is the first measured deformable success gate:**
> `verify.coverage_gate(target_region)` = fraction of final particles inside the target
> `bg_pcd`'s (x,y) convex hull, read straight from the cloud in robot frame; the verifier
> success becomes **VLM-valid AND coverage ≥ goal** (this is where the rigid
> `alignment_gate` plugs in — §5). New pieces: `prompts/{regress,verify}/sweep.txt`, a
> `parse_mpm_rollout` sweep branch, `scripts/optimize.py --task sweep`, bundled
> `examples/sweep_real2sim/0118_sweep_0/` (beans start at 0% coverage). Smoke:
> a +y sweep moves the pile 19.6 cm into the cage → coverage 100%. Tests extend
> `tests/test_mpm_rollout.py` (segment parser + hermetic coverage-gate + a GPU-gated
> sweep-with-gate). **§12 (dough multi-grasp + sweep) is complete** — remaining deformable
> work is a shape gate for rope/dough (§5), render hardening, and real-robot execution.

### The gap is one rollout driver
Same conclusion as §1: the loop core, propose stage, verify/regress seams, waypoint
bridge (`executor/waypoints.py` already accepts propose-primitives *and* the
optimizer's `move`/`gripper_control`), MPM solver, asset-prep, and the MPM context
branch (`context.py:163-172` + `dough.txt`) are **all ported**. The
only new code is a per-material rollout driver + its prompts/parser/demo — exactly the
shape of the rope slice.

**New files (mirror rope):**
- `simpact/executor/mpm_rollout.py` — `MPMRollout` (analogue of `ARAPRollout`).
- `simpact/executor/render_deformable.py` — shared headless renderer (also
  retrofits rope; see §4). open3d `OffscreenRenderer` (EGL, no Xvfb — pilot-proven
  on the 5090) as the safe default; PyVista sphere-glyphs as the higher-fidelity option
  the original already had render code for. Behind `render_deformable(points, tool_boxes,
  camera) -> uint8`. **Do not** port the original system's Klampt `vis.screenshot()` + `*_viewport.txt`.
- `simpact/generator/regress.py::parse_mpm_rollout` — builds regress text from the MPM
  block (bbox + centroid + `final_points_path`).
- `prompts/{regress,verify}/dough.txt`.
- `scripts/optimize.py --task dough` — `do_rollout` closure wiring `MPMRollout` into
  the shared `OptimizationLoop`, mirroring `optimize.py --task rope`.
- `examples/dough_real2sim/` — a recorded dough trial + a recorded `runs/` output.

### MPM setup recipe (verified against `shape_kinetic_sand.py`)
Encapsulate in `MPMRollout`; externalize the magic numbers into a config (the solver
itself is do-not-touch — copy-don't-modify):
- Load `scene.yaml raw_pcd_path` (`*_mpm_points.npy`) → apply the **`+[0, 0.5, 0]`
  grid shift** (the original `# TODO: fix grid bounds` hack; **undo by subtracting on save** —
  this is the load-bearing gotcha), `volume = 2.5e-8` per particle,
  `load_initial_data_from_torch`.
- `set_parameters_dict({E:2000, nu:0.4, material:"plasticine", g:[0,0,-10],
  density:200, grid_v_damping_scale:0.9, yield_stress:50})` → `finalize_mu_lam_bulk()`.
  (Sand variant: `material:"sand", nu:0.2, friction_angle:35`.)
- `add_surface_collider((0,0,0.13),(0,0,1),"sticky")` — table.
- Two `add_rotate_box_collider` fingers: box `[0.06,0.02,0.5]`, local centers
  `±0.01773615 y / 0.07102775 z`, spread by `init_width/2`, twist world-frame velocity
  `±(init_w − final_w)/(2·num_steps·dt)` from the gripper pose. Gripper pose = plan
  applied to the **`context.txt` EE pose** (same lesson the rigid + rope slices hit —
  not `scene.yaml init_gripper_pose`).
- Step `p2g2p` ~500× at `dt=0.002`; record final cloud (un-shifted) + bbox + centroid.
- Sweep variant = a single pusher box with a linear twist instead of two closing jaws.

### Rollout-JSON envelope (unchanged loop)
Emit the §2 envelope: `objects["dough"].position = centroid` (keeps
`rollout_displacements` working) + an MPM block (`final_points_path` + bbox).
`parse_mpm_rollout` reads that block; verifier/regressor/loop untouched.

### Success signal
VLM-only first (faithful — the original had **no** measured dough metric, §5). The measured-gate
seam is open; the first *measured* deformable gate belongs to **sweep coverage vs
`bg_pcd`**, not shaping — so it lands with the sweep variant, not the squeeze.

### Sequencing (refines §9 step 3-4)
1. ✅ **Multi-grasp dough squeeze** — `MPMRollout` (a *list* of 1..N squeezes via
   `grasps_from_plan`; single squeeze = N=1, no separate path) + shared `render_deformable`
   + `parse_mpm_rollout` + `sand.txt` prompts (regress emits one `move`+`gripper_control`
   pair per squeeze, §7 schema-translation) + demo → recorded
   `examples/dough_real2sim/*/runs`, VLM-only. State persists across grasps in one
   continuous rollout (not re-planning).
2. ✅ Sweep (single pusher) + **first measured deformable gate** — `SweepRollout` +
   `coverage_gate` (fraction of the pile inside the `bg_pcd` target region). Success =
   VLM-valid AND coverage ≥ goal.

### Port hygiene (§8 — confirmed present in source)
Strip `import pdb; pdb.set_trace()` (`shape_kinetic_sand.py:162`), `AIzaSy…` keys in
`regress_gemini_sand*`, `/home/ydu/haowen/…` paths + viewport files; guard `franky` in
asset prep.

### Cost / platform caveat
~500 `p2g2p` × up to 8 candidates × N regress iters is GPU-heavy vs rope's single
quasi-static ARAP solve — use fewer substeps inside the loop than in a final render
pass, and it needs **CUDA** (rope's slice is CPU-runnable; dough is not).

---

## 13. Automating rope endpoint selection with the VLM — implementation plan

Concrete, code-grounded plan for the **"rope endpoint grounding"** item (now shipped in
this release — see the Status note at the end of this section). This refines the §6
*design decision* into an implementable spec against the **current** refactored code, and
it does not touch the closed-loop optimizer — it only replaces the human click that seeds
`scene.yaml`.

### Where endpoint selection happens today (fully manual)
The two rope endpoints are the *only* human-clicked inputs in the whole rope pipeline:
1. `prepare_rope_asset.py:148-161` — a person Shift+Clicks two points in an Open3D
   window (`pick_points_on_point_cloud`); click 0 → `fixed_point`, click 1 →
   `free_end`, both written to `scene.yaml`.
2. `context.py:181-196` (`_object_pose_block`) — reads them back into the VLM context
   as text ("rope free end position…", "rope fixed end position…").
3. `rope_rollout.py:152-154` — with `free_end_only=True` the grasp is *overridden* to
   `scene.yaml`'s `free_end`; the `fixed_point` blob becomes the Dirichlet anchor
   (`fixed_idx`, `rope_rollout.py:147`).

So the VLM currently chooses only the *drag*, never the endpoints. Automating means
replacing the manual click in step 1; steps 2–3 stay unchanged if we write the same
`scene.yaml` schema.

### Two sub-problems (do not conflate)
| | problem | best tool |
|---|---|---|
| **A. Tip localization** | *where* are the rope's two ends in 3D? | **geometry** — a curve-topology problem; a VLM can't emit reliable 3D coords |
| **B. Role assignment** | which tip is *fixed/anchored* vs *free/graspable*? | **VLM** — semantic (which end is clamped/held/near a fixture, or which to grasp for the goal) |

The failure mode of "ask the VLM for two 3D points" is hallucinated coords off the
cloud. Robust design: **geometry proposes candidates, the VLM disambiguates roles on
an annotated image.** This matches §6's "geometry for *where*, VLM for *which*".

### Refinement over §6 — detect tips in 3D, not from the 2D mask
§6 proposed skeletonizing the **2-D mask** then back-projecting (noisy at thin-rope
depth edges — §6's own top caveat). But the current pipeline **already has the
segmented 3-D cloud** (`segmented_object.ply`) and already builds a rope graph from it
(`connect_points`, `rope_rollout.py:136`). So detect the tips **directly in 3-D** and
skip back-projection entirely — strictly more robust, and reuses in-repo machinery.

### Phased plan
**Phase 1 — geometric tip candidates (deterministic, no VLM).**
New `simpact/real2sim/detect_rope_endpoints.py`:
- Build the rope graph via `connect_points(pts, r)` (as `rope_rollout.py:136` does).
- Endpoints = the geodesically **farthest-apart pair** on that graph (all-pairs on a
  downsampled node set), or the degree-1 nodes when the graph is a clean chain; PCA
  principal-axis extremes as a cheap fallback for a near-straight rope.
- Return two 3-D tips + a confidence (graph connected? two clear extremes?).

**Phase 2 — annotate the image.**
Project both candidates to the camera image with the existing `project()`
(`render_deformable.py:27`) and draw labeled markers **"A"/"B"** on the RGB. Keep the
3-D coord behind each label — **no deprojection needed** (this is what makes the
approach robust).

**Phase 3 — VLM role call (the "VLM endpoint selection").**
New `prompts/grounding/rope_endpoints.txt` + a thin `simpact/generator/ground.py`
(mirrors `propose.py`'s image+prompt call, `propose.py:82-86`). Send the annotated
image + short context; force structured output:
```json
{"fixed": "A", "free": "B", "are_valid_tips": true, "confidence": 0.9, "reasoning": "..."}
```
Because the VLM only *labels* pre-computed candidates, its answer maps back to exact
3-D coords — no hallucinated geometry.

**Phase 4 — write-back.**
Map chosen labels → the 3-D tips → write `fixed_point`/`free_end` to `scene.yaml`
(same schema the rest of the pipeline reads), so **zero downstream changes** in
`context.py` / `rope_rollout.py`. Add `endpoint_source: vlm|manual` +
`endpoint_confidence` for provenance/reproducibility.

**Phase 5 — validation gates + fallback.**
- Reject if the two tips are closer than a min separation (not two distinct ends), or
  `are_valid_tips=false`, or `confidence < τ` → fall back to manual `pick_points`,
  printing why.
- Assert both chosen tips lie on the cloud (nearest-node distance < voxel), reusing
  the BC-assertion pattern already used for the handle checks.

### Alternative (secondary, not lead)
Let the VLM emit raw 2-D pixel clicks (Gemini can output points) and snap each ray to
the nearest cloud point. Removes the geometry pre-pass but needs a deproject/ray-snap
helper and is far likelier to land mid-rope — keep only if Phase-1 geometry proves
unreliable on curved/self-crossing ropes (§6's ambiguity caveat).

### Files touched
- **new** `simpact/real2sim/detect_rope_endpoints.py` (Phases 1–2, 4–5)
- **new** `simpact/generator/ground.py` + `prompts/grounding/rope_endpoints.txt` (Phase 3)
- ~~**edit** `prepare_rope_asset.py` — add `--auto-endpoints` routing~~ → done differently:
  grounding fully replaced the manual picker in `build_scene` (`ground_rope_endpoints`),
  and the legacy `prepare_rope_asset.py` was removed as superseded (no fallback edit).
- **tests** — Phase-1 detector on the two committed clouds (tips ≈ the hand-clicked
  `scene.yaml` values, ±~1 cm); a mocked-VLM role-assignment test (no API in CI, like
  the existing generator tests).

### Effort & fit
~Phase 1 half day (geometry is the crux), Phases 2–4 half day (grounding mirrors
`propose.py`), validation/tests half day. Slots into the "rope endpoint
grounding" item without disturbing the optimizer. Precision bar is lenient — the sim
uses a `GRIP_RADIUS≈0.03` (3 cm) blob (`rope_rollout.py:31`), so a tip within ~2–3 cm
suffices (§6). Start with **Phase 1** (highest-risk, validates the whole approach
against the two committed clouds before any VLM wiring).

> **Status: IMPLEMENTED.** Phase 1 = `simpact/real2sim/detect_rope_endpoints.py`;
> Phases 2–3 = `simpact/generator/ground.py` + `prompts/grounding/rope_endpoints.txt`
> (tests `test_rope_endpoints.py`, `test_ground.py`). On the two committed clouds the
> detector lands on the true tips and the VLM reads the clamp as the fixed end,
> matching the human roles. Wired into the offline generator in §14.

---

## 14. `scene.yaml` schema + offline scene generation (RGB-D → scene)

Two connected pieces: what a `scene.yaml` must actually contain, and a builder that
produces a whole scene from a recorded RGB-D bundle without a robot or manual clicks.

### 14.1 Minimal `scene.yaml` schema (field audit)

Only load-bearing fields are kept; everything else was removed (see the commit
`refactor(scene): trim scene.yaml…`).

| field | material | consumed by | status |
|---|---|---|---|
| `camera` `{profile, cam}` | all | `camera_calibration.load_camera` (§14.5) | **required** — calibration registry reference |
| `material` `{E, nu, yield_stress, density, …}` | MPM | `material.load_material` (§15) | **required (MPM)** — VLM-estimated per scene |
| `fixed_point`, `free_end` | rope | `rope_rollout.py`, `context.py` | **required** (VLM-groundable) |
| `endpoint_source` / `endpoint_confidence` | rope | provenance | optional |
| `raw_pcd_path` | MPM | `mpm_rollout.py`, `context.py` | **required** (relative filename) |
| `initial_ee_pose` (4×4) | MPM | `mpm_rollout.py`, demos | **required** |
| `bg_pcd_path` | MPM | `mpm_rollout.py`, `context.py` (both `.get()`) | **optional** — target region for coverage-gated MPM tasks; sweep sets it, any MPM scene may. Read via `.get()`, so absent-when-unused (no null placeholder). |
| `object_name` | MPM | `mpm_rollout.py` (label) | optional |
| `init_mpm_center` | MPM | `context.py` fallback | **optional** — centre is computed live from the cloud |
| ~~`init_gripper_pose`~~ | all | nobody | **removed** (dead; init pose comes from `context.txt` / `initial_ee_pose`) |
| ~~rope `raw_pcd_path`~~ | rope | nobody | **removed** (rope reads `segmented_object.ply` directly; the old value baked an absolute `/home/…` path) |
| ~~`cam{id}_K.txt` / `cam{id}_to_robot.txt`~~ | all | — | **no longer embedded** — replaced by the `camera:` registry reference (§14.5); embed only for a portable bundle |

The initial EE pose is embedded in every material's `scene.yaml` (`initial_ee_pose`, the
runtime source resolved by `context.resolve_initial_ee`; `capture/initial_ee_pose.txt` is
the raw record and `context.txt` a VLM-facing copy). `context.py`'s
`_mpm_cloud_center` now treats `init_mpm_center` as an optional fallback and raises a
clear error if the cloud is unavailable *and* no fallback is set. Camera calibration is a
**registry reference** now, not embedded per-scene files — see §14.5.

### 14.2 `build_scene.py` — the offline generator

`simpact/real2sim/build_scene.py` turns a recorded RGB-D bundle into a scene dir +
minimal `scene.yaml`, mirroring the original `prepare_*_asset.py` procedure
(`get_stream` → `run_gsam2` → `mask_extraction` → back-project) but offline. It is the
deformable sibling of `run_real2sim.py` (which does the same for the rigid/MuJoCo path).

Pipeline (single selected camera, original stage order):
1. load `camera{cam}_rgb.{npy,png}` + `camera{cam}_depth.npy` + intrinsics
   (`cam{cam}_intrinsics.txt` or `cam{cam}_K.txt`) + `cam{cam}_to_robot.txt`;
2. **segment** the object (`perception/grounded_sam2.py`) → union its masks;
3. zero depth outside the mask → `create_point_cloud_from_rgbd` → `.transform(cam_to_robot)`
   → robot-frame cloud (`estimate_scale.create_point_cloud_from_rgbd`, reused);
4. per material:
   - **rope** → save `segmented_object.ply`, then `ground_rope_endpoints` (§13, VLM,
     replaces `pick_points`) → `fixed_point`/`free_end`; emit `context.txt` (§14.3);
   - **dough/sweep** → `sample_between_surface_and_table_columns` → `mpm_points.npy`
     (sweep also segments the target region → `target_region.ply`);
5. write the minimal `scene.yaml` + copy `camera{cam}_rgb.png` / `cam{cam}_K.txt` /
   `cam{cam}_to_robot.txt` into the scene dir.

Four deliberate deviations from the original system's live pipeline (all confirmed):
- **offline** — recorded bundle, no `get_stream` live capture;
- rope `pick_points` (human) → **VLM endpoint grounding**;
- live `robot.state.O_T_EE` → a **recorded EE-pose file** (§14.3);
- **config'd calibration** (per-scene extrinsic) instead of the original system's hardcoded
  date-stamped `./cam_utils/optimized_transform*` (a hard-coding audit finding).

The `segmenter` and VLM `generate_fn` are **injectable**, so the whole builder is
unit-tested with fakes on a synthetic RGB-D bundle — no GPU / API / external repo
(`tests/test_build_scene.py`). A **live** run needs the Grounded-SAM-2 env + a raw
bundle with `.npy` RGB/depth (the committed examples ship processed clouds, not raw
RGB-D, so live is exercised only with the real perception stack).

```
python -m simpact.real2sim.build_scene --raw-dir <bundle> --out-dir <scene> \
    --material rope --object "rope" --ee-pose <bundle>/initial_ee_pose.txt
```

### 14.3 The EE-pose contract (the one non-RGB-D input)

The initial end-effector pose is robot **proprioception** — it is *not* in the RGB-D,
and the original system's capture (`get_stream.py`) never saved it (it writes only RGB-D and its
`--host` arg is unused); the original read it **live** from the robot. Offline it must be
**loaded from a recorded file**, never hardcoded per scene. `build_scene` resolves it:

1. explicit `--ee-pose PATH` — a pose file (`EEPose.from_file`: 4×4 / 16-vec /
   `x y z qx qy qz qw`) **or** an existing `context.txt` (`EEPose.from_context_file`);
2. auto-discover in the bundle: `initial_ee_pose.txt` / `robot_state.txt` / `context.txt`;
3. missing → **hard error** (fidelity-first); opt-in `--allow-home-pose` falls back to
   the original system's recorded home pose (`rollout.py HOME_GRIPPER_*`) with a loud warning.

The resolved pose feeds **both** outputs: rope's `context.txt` (position/orientation/yaw
lines) and MPM's `scene.yaml initial_ee_pose`. Gripper dimensions and the world-frame /
task-rule text are fixed config, not observed data.

**Capture side (option B):** rather than port the original system's hardware `get_stream`, a small
franky-guarded helper `save_ee_pose_from_robot(host, out)` (CLI: `scripts/save_ee_pose.py`)
records `initial_ee_pose.txt` next to the RGB-D at capture, making new bundles
self-contained. Every committed example ships an `initial_ee_pose.txt` (derived from its
`context.txt` / `scene.yaml`) as the canonical recorded-pose artifact for offline use.

### 14.4 `context.txt` vs `scene.yaml` (the split)

`scene.yaml` = machine-consumed geometry (endpoints, cloud path, MPM `initial_ee_pose`).
`context.txt` = the recorded real-world context in the original system's human-readable form — most
importantly the **rope** init EE pose (`EEPose.from_context_file`), plus the object key
points, world-frame convention, and task notes. `build_context()` re-renders the
*VLM-facing* context string at runtime from a template + `scene.yaml` + the parsed EE
pose, so the static `context.txt` is primarily the EE-pose record. For a fresh rope
scene, `build_scene` emits `context.txt`; MPM needs none (its EE pose is in `scene.yaml`).

### 14.5 Camera-calibration registry (keyed profiles + per-scene reference)

Camera params split by natural unit — intrinsics `K` are **per-camera**, extrinsics
`cam_to_robot` are **per-calibration-session** — so neither a single frozen global (the
old `transform_6d._EXTRINSIC_BY_CAMERA`, wrong for other sessions — audit
item B) nor per-scene duplication fits. Calibration lives in a **keyed registry** and a
scene records which profile it used.

- **Registry** (rig data OUT of the code package, repo `assets/`):
  `assets/calibration/<profile>/cam{id}_{K,to_robot}.txt` + `profile.yaml`
  (`image_size`, date). Seeded profiles: `1026` (rope/dough/sweep), `0103` (rigid).
  Resolved via `simpact.utils.config.get_calibration_dir()` = `$SIMPACT_ASSETS_DIR`/`assets`
  `/calibration`.
- **Reference**: `scene.yaml` carries `camera: {profile: "1026", cam: 1}`.
- **Resolver** `simpact/real2sim/camera_calibration.load_camera(scene_dir, cam, profile)`,
  precedence: **embedded** `scene_dir/cam{id}_{K,to_robot}.txt` (portable bundle) →
  scene.yaml `camera:` ref → explicit `profile` arg → registry → **clear error** (never a
  silent frozen default). All consumers (`rope_rollout`, `mpm_rollout`, `ground`,
  `transform_6d`) go through it; `transform_6d` defaults raw rigid trials to the `0103`
  profile, overridable per-scene.
- **`build_scene` modes**: `--profile P` writes the `camera:{profile:P}` reference (the
  default, reusable — matches the bundled examples); `--embed-calibration` materializes
  `cam{id}_{K,to_robot}.txt` into the scene for a portable, registry-independent bundle.
- **The examples reference the registry** (no embedded cam files). Verified
  behavior-preserving (registry-resolved `K`/extrinsic == the old embedded files
  bit-for-bit) and geometrically correct (each scene's cloud projects 100% in-bounds —
  durable test `test_example_cloud_projects_in_bounds`, which catches a wrong profile
  assignment).

Robot/gripper models now live in `assets/robot/` alongside the calibration registry
(moved out of the code package); `simpact/` is pure code, all rig data under `assets/`.

---

## 15. Material parameters — VLM material-ID (per-scene, real2sim)

MPM physical parameters (`E`, `nu`, `yield_stress`, `density`) were **hardcoded
per-material recipes** (`mpm_rollout.DEFAULT_MATERIAL`/`SWEEP_MATERIAL`) — fixed defaults,
never identified from the observation (an audit finding). They are now **VLM-estimated
per scene during real2sim** and written into `scene.yaml` — physics inferred from the
observation, exactly like geometry (cloud, pose, rope endpoints).

**Not a shared registry** (unlike calibration §14.5). Calibration is a fixed property of
the *rig* → a shared registry is right; **material is a property of the object *instance
in each scene*** → its natural unit is **per-scene**. A predefined material profile just
re-hardcodes the tuned defaults under a nicer name, which defeats the point — so material
is *estimated per scene*, not looked up.

### 15.1 Motivating findings (sensitivity + VLM behaviour)
Measured on the actual example scenes with the final optimized action:
- **Rigid is ~insensitive.** Sweeping the pushed carton's mass (60×) and friction (10×)
  on the real `MuJoCoRollout` + final plan moved the outcome **< 1 %** (Δ ≈ 0.1 cm on an
  18.7 cm push): the kinematic mocap gripper carries the object, so mass/friction barely
  enter. → material-ID is low-value for rigid.
- **Deformable IS sensitive.** Same dough plan, varying `E`/`yield`: final height swings
  ~25 % (2.7→3.4 cm), depth ~20 %. → material-ID genuinely changes the MPM result.
- **VLM *absolute* SI prediction is unusable** where it matters: Young's modulus came back
  10× (dough) to **50× (beans)** off the tuned defaults, with **CoV up to 164 %** (bean `E`
  spanned 10⁵–10⁷ Pa across 5 runs). Bounded/everyday quantities (`nu`, density, mass,
  friction) were stable and within ~2×.
- **An anchored/reference-grounded prompt fixes it.** Telling the VLM these are *effective
  MPM parameters* (not real-world constants) + giving softness→range **bands** + asking for
  a *classification* collapsed variance to **≈0 %** and pulled `E` into the sim regime
  (dough/beans → ~8000 Pa, **0.8–1.6× the defaults**). One consistent residual disagreement
  (bean `yield` 10× below default) is a real physical judgment, not noise.
- **The chosen VLM values change the closed-loop outcome (A/B, no sweep).** Running the
  *committed final plan* with the VLM material vs the old tuned default — same plan, same
  seed, same cloud, only the physics differ — the final shape moves measurably: dough
  Δbbox **0.48 cm** / per-particle RMS **1.9 mm** (stiffer VLM `E` 8000 vs 5000 holds a
  taller block), sweep Δbbox **0.80 cm** / RMS **3.5 mm** (softer, low-`yield` VLM beans
  spread further). So per-scene material-ID is load-bearing, not cosmetic.

Conclusion: estimate via a **classification-into-bands** prompt, not raw SI; **clamp** to
the bands as a safety net; focus on MPM.

### 15.2 Design (implemented for MPM)
- **Estimator in real2sim** — `simpact/generator/material.py::estimate_material(image,
  object_name, material_class, *, generate_fn=gemini_generate)`: the anchored prompt
  (`prompts/material/mpm_params.txt`, bands rendered from `bands.yaml`) → VLM estimates the
  **object physics** `{E, nu, yield_stress, density}` → **clamp to the bands** (safety net)
  → returns with `source: vlm`. `generate_fn` injectable → mocked tests.
- **Written per scene** — `build_scene` (MPM branch) calls it and stamps a `material:`
  block (`{E, nu, yield_stress, density, softness, confidence, source: vlm}`) into
  `scene.yaml`, exactly as `ground_rope_endpoints` writes rope endpoints.
- **Physics vs solver split** — the VLM estimates only the four **object physical
  properties**. The non-physical MPM setup — constitutive model (`plasticine`), gravity,
  numerical damping, and the plasticity `friction_angle` — is a per-class **default** in
  `material.SOLVER_CONFIG` (dough vs sweep), *not* VLM-estimated.
- **Resolver** — `load_material(scene_dir, material_class)` returns the full MPM dict =
  `SOLVER_CONFIG[class]` merged with the scene's `material:` physics. **No manual physical
  fallback**: a scene without a complete `material:` block raises a clear error (pass
  `material_params` explicitly to the rollout to bypass, e.g. sensitivity sweeps).
- **Grounding data** — `assets/materials/bands.yaml` (softness class → E/yield/nu/density
  ranges) is the single source for both the prompt text and the clamp; resolved via
  `config.get_materials_dir()` (env-overridable). It is estimator *grounding*, not a
  catalog of per-object answers.
- **Rollouts** — `MPMRollout`/`SweepRollout` default `material_params` to
  `load_material(scene)`; the old `DEFAULT_MATERIAL`/`SWEEP_MATERIAL` constants are gone.

### 15.3 Status
**IMPLEMENTED for MPM.** Committed example scenes carry VLM-estimated values (real run,
no manual defaults): dough `E=8000, nu=0.45, yield=1500, density=1200` (`softness: soft`);
sweep `E=4000, nu=0.25, yield=400, density=700` (`very_soft`) — both differ from the old
tuned recipes (sweep `yield` dropped 10×: the VLM reads loose beans as low-cohesion).
Tests: `test_material.py` (estimate/clamp/merge/error + example-scenes-in-band) + the
`build_scene` MPM tests (mocked material VLM) + `test_mpm_rollout.py::
test_material_id_changes_outcome` (GPU-gated A/B: the committed VLM material vs the old
default on the final plan must yield a measurably different final shape).

**Rejected design:** an earlier plan mirrored calibration with a *registry of predefined
material profiles* (`playdoh`/`black_beans`) referenced from `scene.yaml`. Dropped — that
re-hardcodes tuned defaults under a name instead of inferring physics per scene, which is
the paper-aligned goal (§15 intro).

**Not done:** rigid `mass`/`friction` estimation (deprioritized — the kinematic rigid push
is ~insensitive, §15.1).

### 15.4 Why this shape
Material is estimated where geometry is (real2sim / `build_scene`) and written per scene,
reusing the same **VLM-writer + validation-clamp** pattern as rope-endpoint grounding. The
clamp-to-bands stops a hallucinated `E` (VLM absolute SI is 10–50× off, §15.1) from
destabilising the sim, while the softness-classification prompt keeps estimates consistent
and in the effective-MPM regime. Effort is concentrated on MPM, where the outcome is
actually parameter-sensitive.

> The one-off VLM-query experiment behind §15.1 was removed in the release cleanup
> (its anchored mode called the shipped `estimate_material`); the outcome-closure A/B
> survives as the GPU-gated regression
> `test_mpm_rollout.py::test_material_id_changes_outcome`. The material-ID step itself is
> wired into the pipeline (see §15.3).
