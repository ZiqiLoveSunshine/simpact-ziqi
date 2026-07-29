# Action sampling — proposals for the manipulation loop

How simpact generates candidate robot actions: the schema, the two samplers
(random + vision-LLM), the scene-context builder, and the driver. This is the
first piece of the **propose → simulate → optimize → execute** loop, ported from
the original proposal generators (`propose*`, `generate_random_proposals`,
`generate_context`).

## The unit of work: a proposal

A **proposal** is one candidate plan — an ordered sequence of coarse manipulation
primitives. A sampler returns a **`ProposalSet`** (several proposals). The JSON is
byte-compatible with the original system's generator output, so recorded `proposal*.json` files
round-trip:

```json
{"action_proposals": [
  {"description": "push then grasp",
   "action_sequence": [
     {"type": "PUSH",  "delta_x": 0.1, "delta_y": 0.0, "reasoning": "approach"},
     {"type": "GRASP", "grasp_width": 0.03, "reasoning": "close on the edge"}]}]}
```

## Primitives (`simpact/actions/primitives.py`)

8 primitives; deltas are in the absolute world frame, rotations are relative.

| primitive | params | meaning |
|---|---|---|
| `PUSH` | `delta_x, delta_y` | move horizontally at current height |
| `LIFT` | `delta_z` | move up |
| `DESCEND` | `delta_z` | move down |
| `GRASP` | `width` (0=closed .. 0.1=open) | set gripper width |
| `RELEASE` | — | open fully (= GRASP 0.1) |
| `ROTATE` | `delta_yaw` (rad) | yaw, relative |
| `ROLL` | `delta_roll` (rad) | roll, relative (roll template only; CoR at the wrist) |
| `FLICK` | `delta_x, delta_y, delta_z` | one quick combined move |

Each carries an optional free-form `reasoning` string. Typed dataclasses with
`to_dict`/`from_dict`; `ActionProposal` and `ProposalSet` wrap sequences and sets
(`ProposalSet.to_json`/`from_json`, `.validate(allowed_types, ranges)`).

> **GRASP key normalization (important).** the original pipeline is internally inconsistent: the LLM
> emits `grasp_width` (all recorded proposals use it), while the random sampler,
> the prompt's output-format block, and the executor used `width` — so the
> executor silently dropped the LLM's grasp. simpact normalizes: the canonical
> attribute is `Grasp.width`; `from_dict` accepts **either** `grasp_width` or
> `width`; `to_dict` emits `grasp_width` (matching the model + recorded data). The
> ported prompt templates were also fixed to use `grasp_width` consistently.

### Optimizer-output plan actions (`Move` / `GripperControl`)

The **propose** stage emits the primitives above; the **regress** optimizer
(`generator/regress.py`, see [EVALUATION.md](EVALUATION.md)) emits a refined plan in
a lower-level, legacy-`WaypointParser`-compatible format — two extra action types on
the same `ProposalSet` schema:

| plan action | params | meaning |
|---|---|---|
| `Move` | `delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw` | one accumulated EE move (position + orientation deltas) |
| `GripperControl` | `width` | set the gripper opening (total, metres) |

`proposal_to_waypoints` (`executor/waypoints.py`) accepts **both** the primitives
and these plan actions, so a proposal and a refined plan roll out through the same
bridge (`MOVE → x,y,z + roll/pitch/yaw`; `GRIPPER_CONTROL → width`). `PLAN_ACTION_TYPES
= ("move", "gripper_control")`.

## Samplers

Two backends produce the same `ProposalSet`:

### Random — `simpact/generator/sampling.py`
`RandomProposer(seed=...)` samples sequences from per-primitive ranges
(`DEFAULT_RANGES`, e.g. `PUSH dx,dy∈[-0.5,0.5]`, `GRASP width∈[0,0.1]`,
`ROTATE dyaw∈[-π,π]`). **Seeded** for reproducibility (the original system's was unseeded); values
rounded to 4 decimals as in the original system. This is also the building block for the future
CEM optimizer (sample around an evolving mean/std instead of uniform ranges).

### Vision-LLM — `simpact/generator/propose.py`
`VLMProposer` builds a prompt from a proposal template + scene image + context +
instruction, calls a vision-LLM, strips markdown fences, parses JSON to a
`ProposalSet`, and **retries** on malformed/invalid output.
- **Provider-agnostic**: the model call is a pluggable `generate_fn(image, prompt)
  -> str`. Default is the secure Gemini client (`GOOGLE_API_KEY` from env — the original scripts'
  hardcoded keys are dropped). Inject a different provider or a test stub without
  touching parse/validate logic.
- Prompt template: [prompts/proposals/primitive.txt](../prompts/proposals/primitive.txt)
  (7 primitives; the roll-variant template was removed with the unused pivot
  task — the `ROLL` primitive stays in the schema for round-tripping recorded proposals).

## Scene context — `simpact/generator/context.py`

`build_context(object_string, data_dir, template, ee_pose, cam_id)` fills the
`{ee_pose}` + `{object_poses}` placeholders of a `prompts/contexts/*.txt` template
(reusing `load_context_template`), reproducing the original system's strings:

- rigid object → `{name}_mujoco_cam{cam}.txt` (4×4 object→world pose),
- rope → `scene.yaml` `fixed_point` / `free_end`,
- MPM (sand/dough) → `scene.yaml` `init_mpm_center` (+ `bg_pcd_path` → target center).

**Decoupled from the robot:** the original read the end-effector pose live from a Franka.
Here the EE pose is an `EEPose` from any of: an explicit value, a file (`4×4`
matrix or `x y z qx qy qz qw`), or — guarded, optional — a live robot
(`EEPose.from_robot(host)`, needs `franky`). So context builds offline from
recorded trials with no robot/GPU.

## Driver — `scripts/propose_actions.py`

The simpact equivalent of the original `propose.sh`.

```bash
# random only — no GPU / LLM / robot
python scripts/propose_actions.py --backend random --task obstacle \
    --n 20 --seed 0 --out /tmp/proposals.json

# LLM (or both) — offline context from a recorded trial + a saved EE pose
python scripts/propose_actions.py --backend both \
    --data_dir /path/to/data/0211_obstacle_0 --cam 1 \
    --objects "orange bottle. brown purple box." --task obstacle \
    --instruction "Push the orange bottle right, avoiding the box." \
    --ee-pose-file /path/to/ee_pose.txt --out /tmp/proposals.json
```

`--task` selects the context template **and** the allowed primitive set
(`TASK_PROFILES`, e.g. `obstacle → {PUSH,LIFT,DESCEND,ROTATE}`, from the original experiments's
per-task "Critical Rules"); `--allowed` overrides. The LLM backend needs
`--ee-pose-file` (offline) or `--host` (live) plus `GOOGLE_API_KEY`.

## Worked example (verified end-to-end)

**Random backend** — no GPU/LLM/robot, runs immediately:
```bash
python scripts/propose_actions.py --backend random --task obstacle --n 20 --seed 0 --out /tmp/p.json
# -> 20 proposals over {PUSH,LIFT,DESCEND,ROTATE}, all valid
```

**LLM backend** — verified live against Gemini (`gemini-2.5-pro`) on recorded trial
`0211_obstacle_0`. Inputs needed: `GOOGLE_API_KEY` in `.env`; a scene dir with
`camera1_rgb.png` + `{obj}_mujoco_cam1.txt`; and an EE pose file (here a synthesized
gripper-above-table 4×4):
```bash
python scripts/propose_actions.py --backend llm \
  --data_dir /path/to/data/0211_obstacle_0 --cam 1 \
  --objects "orange bottle. brown purple box." --task obstacle \
  --instruction "Push the orange bottle to the right while avoiding the brown purple box." \
  --ee-pose-file /tmp/ee_pose.txt --out /tmp/proposals_llm.json
```
The model returned **3 diverse, valid proposals** (abbreviated):
```
[0] orthogonal two-stage: clear the box forward, reposition, push right
      PUSH dx=0.0 dy=-0.106 | DESCEND dz=0.06 | PUSH dx=0.18 dy=0.0 | LIFT dz=0.06
      | PUSH dx=0.034 dy=-0.055 | ROTATE dyaw=-1.57 | DESCEND dz=0.06 | PUSH dx=0.0 dy=0.3 | LIFT dz=0.06
[1] angled curve-around in fewer combined moves
[2] conservative variant using a lower, more stable contact point
validation (obstacle profile {PUSH,LIFT,DESCEND,ROTATE} + ranges): all valid
```
This exercises the whole chain — context build → prompt → model → fence-strip →
parse → schema (incl. `grasp_width` normalization) → task validation.

> EE pose: the original read it live from the robot; rigid trials don't store it. For an
> offline run, synthesize one (4×4 with the gripper above the table) or pass the
> real gripper pose via `--ee-pose-file`; use `--host <FCI_IP>` for a live read.

## Tests

- `tests/test_actions.py` — schema round-trip against **recorded proposals**
  (`tests/fixtures/pag_proposal_{sand,rope}.json`), GRASP alias, seeded
  determinism, range/type validation.
- `tests/test_context.py` — `EEPose` loaders; rigid/rope/MPM context build
  (hermetic + a real `*_mujoco_cam` fixture).
- `tests/test_propose.py` — proposer parse/retry/validation with a stub
  `generate_fn` (no network); a real-Gemini test gated on `GOOGLE_API_KEY`.

All CPU-safe; no GPU/robot/LLM key required except the env-gated cases.

## Status & what's next

Done (and verified end-to-end — random offline + LLM live against Gemini, see
above): schema + random sampler + context builder + VLM proposer + driver +
ported, fixed prompt templates. Deferred to the optimization phase: proposal
**pooling** (the original `propose_combine.py` / `merge_proposals.py`) — it feeds the CEM
optimizer.

These proposals are consumed by the next component, **action evaluation** (sim
rollouts in the executor), which scores each proposal in MuJoCo / Warp-MPM / ARAP.
