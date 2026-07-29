# Action evaluation — physics rollouts (no evaluator)

How a proposed action plan is **evaluated** by rolling it out in physics. This is
the simpact port of the original `executor/push_6d.py` (+ `parse_waypoints.py`), the
consumer of the proposals from [ACTIONS.md](ACTIONS.md).

> **There is no cost / ranking score.** the original system never scored rollouts: the rollout's
> only job is to **produce the outcome a VLM reads** — an overhead screenshot + the
> executed pose/gripper trace. simpact adds one thing the original lacked: a **VLM
> task-completion verifier** that turns the open-loop optimizer into a closed loop
> (judge *done / not-done* per rollout, feed *why-not* back, iterate). That is a
> boolean completion check for loop control, **not** a scalar the optimizer
> minimizes — see [The closed loop](#the-closed-loop-propose--rollout--verify--regress-scriptsoptimizepy---task-push).

```
propose (N) ─▶ rollout each ─▶ verify each ─▶ (none done) regress ─▶ rollout ─▶ verify ─▶ … ─▶ best plan
              └─────────── rollout JSON + screenshots ───────────┘   (loop: accumulating memory)
```

## The rollout

`simpact/executor/rollout.py::MuJoCoRollout`:
1. Builds a **gripper scene** via the unified generator
   `simpact/real2sim/scene.py::build_mujoco_scene(objects, ..., with_gripper=True)` —
   objects (free-jointed mesh bodies) + table + the mocap Franka gripper
   (`assets/robot/franka_mujoco/franka_gripper.xml`, resolved via
   `paths.get_assets_dir()`) + an overhead `top` camera.
2. Converts the proposal to absolute gripper waypoints
   (`simpact/executor/waypoints.py::proposal_to_waypoints`): `PUSH→x,y`,
   `LIFT→+z`, `DESCEND→−|z|`, `ROTATE→yaw`, `ROLL→roll`, `FLICK→x,y,z`,
   `GRASP→width`, `RELEASE→open`; clipped to `z_min` / workspace bounds. (Reads
   the typed `Grasp.width`, so the LLM's `grasp_width` is honored — the original system's executor
   read `'width'` and silently dropped it.)
3. **Snaps the gripper to its start pose** before the first frame
   (`FloatingGripperController.snap_to_mocap`): the `hand` is a freejoint body
   *welded* to `gripper_mocap`, and `set_gripper_pose` only moves the mocap target,
   so without this the weld would drag the hand in from its XML spawn (the origin)
   over the first steps — sweeping across the scene and knocking objects over before
   any action runs. Teleporting the hand's qpos to the mocap pose (identity weld
   relframe) makes it start *at* its pose.
4. Drives the mocap gripper (`mocap_pos`/`mocap_quat` + `left_finger`/`right_finger`
   actuators) through the waypoints with lerp/slerp interpolation, stepping
   `mj_step`; records a snapshot per waypoint.

CPU-OK and deterministic. Offscreen rendering needs a GL context
(`MUJOCO_GL=egl`) and **degrades gracefully** — without GL the JSON (poses) is
still written, screenshots are `null`.

## What gets saved (per proposal)

```
<out_dir>/
  rollout_00.json          # proposal 0
  rollout_00_0.png         # snapshot 0 (initial / "before")
  rollout_00_1.png         # after waypoint 1
  rollout_00_2.png         # settled final / "after"
  rollout_01.json ...
```

Rollout JSON (keys match what a `regress_gemini`-style reader expects):

```json
{
  "timestamp": "20260622_171530",
  "proposal_index": 0,
  "instruction": "Push the orange bottle right, avoiding the box.",
  "object_names": ["orange_bottle", "brown_purple_box"],
  "waypoints": [{"position":[x,y,z], "orientation":[w,x,y,z], "gripper_width":f, "duration":f}, ...],
  "snapshots": [
    {"waypoint_index": 0,
     "gripper": {"position":[x,y,z], "orientation":[w,x,y,z], "width": f},
     "objects": {"orange_bottle": {"position":[x,y,z], "orientation":[w,x,y,z]}, ...},
     "screenshot": "rollout_00_0.png"}, ...
  ]
}
```

Conventions (faithful to the original):
- `snapshots[0]` = "before", `snapshots[-1]` = settled "after".
- `gripper.width` is the **per-finger** actuator value (0–0.04); the optimizer
  doubles it for the total opening.
- `orientation` is `[w,x,y,z]`; screenshots are **480×640** from the overhead
  `top` camera.

`RolloutResult.save(out_dir, index, instruction)` writes this; `RolloutResult`
also exposes a convenience `displacement(name)` and `metrics` (init→final object
motion) — *not* used for scoring, just diagnostics.

## Verified
`tests/test_rollout.py` (headless, `render=False`): the waypoint bridge
(primitive accumulation, `z_min` clamp, `RELEASE`, `grasp_width` alias), a full
MuJoCo rollout where a `PUSH` **physically moves a box** (dx > 2 cm), and the
recorded-format `save()`. The rendering path (overhead screenshots showing the gripper
displacing the object) is verified separately on a GL-capable host.

## The closed loop: propose → rollout → **verify ↔ regress** (`scripts/optimize.py --task push`)

the original regress is **open-loop**: analyze N candidate rollouts once, emit one plan,
stop — with no check that the plan actually works (the one-shot demo could regress a
*working* push into a worse one and never notice). simpact closes the loop with a
**VLM task-completion verifier** and an accumulating rollout memory:

```
propose (N, VLM) ─▶ rollout each (MuJoCo) ─▶ VERIFY each ──(any success?)──▶ return it
                                                  │ none
                                                  ▼
              ┌──────── all rollouts-so-far (each carrying its verdict) ◀── append failed attempt
              ▼                                                                      │
        regress (VLM) ─▶ refined plan ─▶ rollout ─▶ VERIFY ──(success?)── no, iter<max ┘
              ▲                                          │ yes / out of iters
              └──────────────────────────────────────── return best plan
```

```bash
# defaults to trial 0103_push_0 (committed 0103 calibration) + the standard push
MUJOCO_GL=egl GOOGLE_API_KEY=... \
  python scripts/optimize.py --task push --out_dir /tmp/loop_demo --view top_view --max_iters 5
```

What each stage does:
1. **propose** — `VLMProposer` sees the real `camera1_rgb` photo + a context built
   from the estimated object poses (`build_context`) + the instruction, and emits
   candidate primitive plans (PUSH/DESCEND/LIFT/…).
2. **rollout** — each candidate runs through `MuJoCoRollout` on the real textured
   meshes. Object **world poses are recomputed from the aligned camera→robot
   extrinsic** (`transform_to_robot_frame` = `get_camera_to_robot(cam) @ 6d_cam`),
   *not* the trial's `_mujoco_cam` files, and each base is snapped onto the table
   top — so the cartons start upright and co-planar instead of toppling. (Hence the
   default trial `0103_push_0`; pre-0103 trials place objects ~15–18 cm off.) The
   gripper starts at the trial's **real recorded end-effector pose**, resolved by
   `resolve_initial_ee` (scene.yaml `initial_ee_pose`; legacy trials fall back to
   `context.txt`, the same file the original `push_6d.py` reads) and converted EE→mocap by
   the 0.105 m tool offset — exactly as the original executor initialized the sim — so the start
   matches the real arm at capture (`0103_push_0`: EE `[0.486, -0.232, 0.251]`,
   right beside the white carton, so a push can actually reach it). The VLM context
   advertises the EE-frame pose. Without any recorded pose the sim falls back to
   the original system's generic Franka home (`HOME_GRIPPER_*` in `executor/rollout.py`).
3. **verify** — `TaskVerifier` (`simpact/generator/verify.py`) decides success as
   **a measured geometric gate AND a VLM validity check**:
   - **Measured goal gate** (`success_gate`, e.g. `alignment_gate`) — the part that
     must be exact is read straight from the rollout's final object positions, *not*
     eyeballed. For this push task the goal is **horizontal alignment**: the two
     cartons end at the same front-back depth, `|y_white − y_blue| ≤ 2 cm`. The
     top-down camera foreshortens depth, so a VLM routinely mis-judges this — the
     measurement cannot.
   - **VLM validity** (`prompts/verify/push.txt`, strict, default-fail) — the
     VLM judges the qualitative safety checks it *is* good at from the before/after
     images + per-object displacement: right rough **direction**, **collateral**
     objects undisturbed (< 3 cm), all objects **upright/on-table** (not toppled),
     and **no catastrophe** (target not flung off).

   Final `success = gate_passed AND vlm_valid`; a failing gate feeds "not aligned yet:
   |Δy| = … cm" back as actionable feedback. This kills false positives from either
   side (a carton vaguely toward the target that isn't actually aligned; or an aligned
   result that toppled/shoved the landmark). **If a candidate already passes, the loop
   returns it** (no regress).
4. **regress ↔ verify loop** (`OptimizationLoop`) — if none pass, `RegressOptimizer`
   reads **all** rollouts in the folder, each now annotated with its verdict
   (`parse_rigid_rollout` surfaces `VERIFIER OUTCOME: FAILURE — …` + `STILL NEEDED:
   …`), and returns one refined `move`/`gripper_control` plan. That plan is rolled
   out (via the same `proposal_to_waypoints`, which accepts `Move`/`GripperControl`),
   verified, and — if it fails — **appended to the rollouts folder as new memory** for
   the next iteration. Repeats until a verdict is success or `--max_iters` (default 5).
5. **confirm** — the demo rolls the chosen plan out **once more** into
   `final_rollout/` and verifies it independently. The loop's in-flight verdict can
   be a false positive, so this final pass is the authoritative "did the returned
   plan *actually* solve the task?" — the demo reports that verdict (`DEMO_OK` only
   when the final confirmation itself passes).

Result: the chosen plan (first candidate that verified, else the most-informed
refined plan) → `refined_plan.json`, re-rolled + independently verified in
`final_rollout/`. Every attempt (candidates + refined) lands in
`rollouts/rollout_NN.json` with its verdict + before→after PNGs.

> The verifier is a **completion check for loop control**, still *not* a cost/ranking
> over rollouts — there's no scalar score the optimizer minimizes. It only answers
> "is the task done?" to decide whether to stop/accept, and feeds *why-not* back as text.

Requirements: `GOOGLE_API_KEY` (propose + regress + verify) and a GL context
(`MUJOCO_GL=egl`) for the rollout screenshots.

### Recorded example outputs (committed)

A real run is checked in at
[`examples/push_real2sim/0103_push_0/runs/`](../examples/push_real2sim/0103_push_0/runs/)
— `run_log.txt`, `rollouts/rollout_NN.json` (each with its `verdict`) + before→after
PNGs, and `refined_plan.json` — runnable via
`scripts/optimize.py --task push` (see
[examples/README.md](../examples/README.md) for the
walk-through of that run.

## Status
- ✅ unified scene generator (`with_gripper`)
- ✅ primitive→waypoint bridge (primitives **+** `Move`/`GripperControl` plan actions)
- ✅ `MuJoCoRollout` → recorded-format rollout JSON + overhead screenshots
- ✅ **VLM optimizer** (`RegressOptimizer`, `regress_gemini` port) → refined
  `move`/`gripper_control` plan
- ✅ **VLM task verifier** (`TaskVerifier`) — completion check, VLM + motion evidence
- ✅ **closed loop** (`OptimizationLoop`) — verify ↔ regress with accumulating
  rollout memory + verdict feedback; short-circuits on a successful candidate

Next: **MPM** (sand/dough) and **ARAP** (rope) rollouts so the loop covers
deformables, then the executor (Phase 5).
