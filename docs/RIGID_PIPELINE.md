# Rigid real2sim pipeline — RGB-D + language → MuJoCo scene

How the rigid perception pipeline turns a **single RGB-D snapshot of a scene plus
a natural-language list of objects** into a **MuJoCo XML scene** you can simulate:
one watertight, textured, metric-scaled mesh per object, each placed at its
estimated 6-DoF pose, on an auto-sized table.

It is the simpact port of the original system's `real2sim/test.sh`.
This document explains *what happens and why*; for installing the environment see
[RIGID_ENV_SETUP.md](RIGID_ENV_SETUP.md), and for correctness/metrics see
[VALIDATION_rigid.md](VALIDATION_rigid.md).

---

## At a glance

```
            language: "pringles. white coconut milk carton."
                       │
 RGB-D snapshot ───────┤
 (rgb + depth + K)     │
                       ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ 1  capture            rgb.npy, depth.npy (metres), cam_K.txt       │
   │ 2  SEGMENT  (model)   Grounded-SAM-2:  rgb + text → masks/boxes    │
   │ 3  mask-extract       per object → mask.npy + RGBA crop            │
   │ 4  RECONSTRUCT(model) SAM-3D:  RGBA crop → complete mesh (glb)     │
   │ 5  scale              depth + mask → metric  *_scaled.obj + tex    │
   │ 6  POSE     (model)   FoundationPose:  scaled mesh + RGB-D → 4×4   │
   │ 7  transform          camera→robot/world frame → *_mujoco.txt      │
   │ 8  assemble           write MuJoCo scene_*.xml                     │
   └──────────────────────────────────────────────────────────────────┘
                       │
                       ▼
              scene_<trial>.xml  (bodies + meshes + table)
```

Three of the eight stages run learned models (**2, 4, 6**); the rest is
deterministic geometry/IO glue. All three models run **in one uv env, in one
process** (see RIGID_ENV_SETUP.md).

---

## Inputs and outputs

**Inputs**
- `cameraX_rgb.npy` — RGB image, `HxWx3` uint8.
- `cameraX_depth.npy` — depth, `HxW` float **in metres** (already metric; do *not* divide by 1e3).
- `camX_K.txt` — 3×3 camera intrinsics.
- a **text prompt** listing the objects, lowercase and dot-separated, e.g.
  `"pringles. white coconut milk carton."`.
- *(stage 7 only)* a camera→robot/world extrinsic calibration.

**Output**
- `scene_<trial>.xml` — a MuJoCo scene: one free-floating textured body per
  object at its estimated pose, plus a ground table sized to the objects.
- The intermediate per-object assets (mesh, texture, pose) that the XML references.

---

## Why each model is needed (the key idea)

A single RGB-D view is **partial**: depth only sees the front faces of objects, so
you cannot simulate from it directly. The pipeline therefore splits the problem:

- **Geometry** comes from **SAM-3D** (image→3D), which *hallucinates a complete,
  watertight mesh* from one masked image — the back of the object included.
- **Metric scale** and **placement** come from the **depth**: stage 5 scales the
  (scale-free) generated mesh to the observed point cloud, and stage 6
  (FoundationPose) finds the 6-DoF pose that aligns that mesh to the RGB-D.
- **Which pixels belong to which object** comes from **Grounded-SAM-2**, driven by
  your language prompt.

So: segmentation says *what and where in 2D*, SAM-3D supplies *full 3D shape*,
depth supplies *real-world size and pose*.

---

## Stage by stage

### 1 — Capture *(glue)*
`get_stream.py`. Snapshot the RealSense camera(s) → `cameraX_rgb.npy`,
`cameraX_depth.npy` (metres), and the intrinsics `camX_K.txt`. In offline runs
these already exist in the trial directory.

### 2 — Segment *(model: Grounded-SAM-2)*
`run_gsam2.py` →
[`GroundedSAM2Segmenter`](../perception/grounded_sam2.py).
GroundingDINO (open-vocab detection from the **text prompt**) proposes boxes;
SAM2 turns each box into a precise mask.
- **in:** `rgb`, `text_prompt`
- **out:** `SegmentationResult(masks (N,H,W), labels, scores, boxes)`
- GroundingDINO loads from `transformers`; SAM2 from the `sam2` wheel — no repo
  build. Detection thresholds `threshold=0.4`, `text_threshold=0.3`.

### 3 — Mask extraction *(glue)*
`mask_extraction.py`. Splits the
segmentation into per-object artifacts:
- `cameraX_mask_{obj}.npy` — full-image binary mask → used by **stages 5 & 6**.
- `cameraX_mask_{obj}_cropped.png` — **RGBA** crop (bbox + 10 px padding, object
  pasted on transparent background, mask in the alpha channel) → used by **stage 4**.

### 4 — Reconstruct *(model: SAM-3D)*
`run_imgto3d.py` (the original used Hunyuan3D) →
[`SAM3DReconstructor`](../perception/sam3d.py).
Single masked image → a complete textured mesh. **Scale-free** (canonical units).
- **in:** the RGBA crop
- **out:** `{obj}_textured.glb` (mesh + vertex colors)

### 5 — Scale *(glue, uses depth)*
`estimate_scale.py`. Back-projects the
masked depth to a point cloud and fits the generated mesh's scale to it, making
the mesh **metric**.
- **in:** `{obj}_textured.glb` + `cameraX_depth.npy` + `cameraX_mask_{obj}.npy` + `K`
- **out:** `{obj}_scaled.obj` + `{obj}_scaled_0.png` (texture)

### 6 — Pose *(model: FoundationPose)*
`estimate_pose.py` →
[`FoundationPoseEstimator`](../perception/foundationpose.py).
Finds the 6-DoF pose that best aligns the metric mesh with the RGB-D observation.
- **in:** `{obj}_scaled.obj`, `rgb`, `depth`, `mask`, `K`
- **out:** `{obj}_6d_cam{id}.txt` — 4×4 object→**camera** transform

### 7 — Transform to world frame *(glue)*
`transform_6d.py`. Composes the
object→camera pose with the **camera→robot/world** calibration so objects land in
a common simulation frame.
- **in:** `{obj}_6d_cam{id}.txt` + camera→robot extrinsics
- **out:** `{obj}_mujoco_cam{id}.txt` — 4×4 object→world transform
- ⚠️ Needs your own calibration. Without it, keep the pose in the **camera frame**
  (a camera-frame scene is still valid, just not robot-aligned).

### 8 — Assemble the scene *(glue)*
`generate_xml.py`. Emits the MuJoCo XML.
- **in (per object):** `{obj}_scaled.obj`, `{obj}_scaled_0.png`, `{obj}_mujoco_cam{id}.txt`
- **out:** `scene_<trial>.xml`

---

## The per-object data contract (file naming)

Every stage communicates through files keyed by the object's label. For object
`name` and camera `id`:

| file | produced by | consumed by |
|---|---|---|
| `cameraX_mask_{name}.npy` | 3 mask-extract | 5 scale, 6 pose |
| `cameraX_mask_{name}_cropped.png` (RGBA) | 3 mask-extract | 4 reconstruct |
| `{name}_textured.glb` | 4 reconstruct | 5 scale |
| `{name}_scaled.obj` + `{name}_scaled_0.png` | 5 scale | 6 pose, 8 xml |
| `{name}_6d_cam{id}.txt` (4×4, obj→cam) | 6 pose | 7 transform |
| `{name}_mujoco_cam{id}.txt` (4×4, obj→world) | 7 transform | 8 xml |

The XML only needs the last three: **scaled mesh + texture + world pose.**

---

## What the generated XML contains

From `generate_xml.py`, each object becomes:

```xml
<asset>
  <mesh name="pringles_mesh" file="./data/<trial>/pringles_scaled.obj"/>
  <texture name="pringles_tex" type="2d" file=".../pringles_scaled_0.png"/>
  <material name="pringles_mat" texture="pringles_tex" .../>
</asset>
<worldbody>
  <body name="pringles" pos="x y z" quat="w x y z">
    <joint type="free" name="pringles_joint" damping="0.1"/>
    <geom type="mesh" mass="0.5" friction="0.3 0.005 0.0001"
          mesh="pringles_mesh" material="pringles_mat"/>
  </body>
</worldbody>
```

- **pose** → the body's `pos` + `quat` (MuJoCo wants `wxyz`; the script converts).
- each object gets a **free joint** (it's a movable rigid body).
- a **table** is auto-generated: centered on the objects' XY bounds and sized to
  span them + 15 cm padding, placed just under the lowest object.
- mass/friction are fixed defaults (0.5 kg; `0.3 0.005 0.0001`) — tune per object
  if dynamics matter.

---

## How it maps to simpact

The three models sit behind the stable ABCs in
[perception/base.py](../perception/base.py); the glue stages are plain functions.

| ABC | adapter | stage |
|---|---|---|
| `Segmenter.segment()` | `GroundedSAM2Segmenter` | 2 |
| `ImageTo3DReconstructor.reconstruct()` | `SAM3DReconstructor` (or `Hunyuan3DReconstructor`) | 4 |
| `PoseEstimator.estimate()` | `FoundationPoseEstimator` | 6 |

Each adapter locates its model repo/weights via an env var
([repos.py](../perception/repos.py)) and runs **in-process** in the shared uv env.
Because the seam is the ABC, swapping image→3D backends (SAM-3D ↔ Hunyuan3D) is a
one-line change. A thin `run_perception` driver (the simpact equivalent of
`test.sh`) chains segment → mask-extract → reconstruct → scale → pose → transform
→ xml.

> Adapter status: `GroundedSAM2Segmenter`, `SAM3DReconstructor`, and
> `FoundationPoseEstimator` are **implemented** and exercised end-to-end by
> `scripts/run_rigid_pipeline.py` (verified on `1030_push_8`). The package stays
> importable without a GPU/repos — heavy imports and model loads are lazy, and
> each adapter raises `PerceptionRepoNotFound` only when actually used without its
> repo/weights. `Hunyuan3DReconstructor` remains a pending port.

---

## Coordinate frames (read this before trusting poses)

- Stage 6 outputs **object→camera** (`_6d_cam`).
- Stage 7 outputs **object→world/robot** (`_mujoco_cam`) — *only* as good as your
  camera→robot calibration.
- The XML places bodies in the **world frame**; MuJoCo quaternions are `wxyz`.

If you have no robot calibration, run the pipeline through stage 6 and treat the
camera frame as the world frame — the relative placement of objects is still
correct, only the global orientation differs.

---

## Worked example (end to end, one command)

[`scripts/run_rigid_pipeline.py`](../scripts/run_rigid_pipeline.py) runs all eight
stages on one RGB-D snapshot and writes a MuJoCo-ready scene. It drives the
perception **adapters** (`GroundedSAM2Segmenter`, `SAM3DReconstructor`,
`FoundationPoseEstimator`) behind their [base.py](../perception/base.py)
interfaces; the deterministic stages (mask-extract, scale, world-align, XML) are
plain functions in the driver.

The committed **runnable** example (`examples/push_real2sim/`, via
the perception build; see examples/README.md) is trial **`0103_push_0`** — captured with the committed `0103`
camera→robot calibration, so its poses lift cleanly into the robot frame. The
inline numbers below are a reference perception run on `1030_push_8` (kept for the
scale-fix story); the command is identical bar `--data_dir`/`--objects`.

```bash
.venv/bin/python scripts/run_rigid_pipeline.py \
  --data_dir /path/to/data/1030_push_8 \
  --objects "pringles. white coconut milk carton." --cam 1 \
  --out_dir /tmp/rigid_demo --xml /tmp/rigid_demo/scene.xml
```

Observed run (RTX 5090, one process, **peak 18.7 GB / 32 GB**):

```
Stage 1+2  Grounded-SAM-2  detected: [('white coconut milk carton', 0.71), ('pringles', 0.50)]
Stages 3-5 SAM-3D + scale  pringles: 213k-vert mesh, x0.229 -> 26.2 cm
                           white coconut milk carton: 217k-vert mesh, x0.234 -> 26.5 cm
Stage 6    FoundationPose  pringles t=[0.114 -0.034 0.718] (camera frame, ~72 cm depth)
Validation 6-DoF overlay   pose_overlay.png (3-D box + axes on the RGB)
Stage 7    world-align     fit table plane from depth -> z-up; objects settled onto table
Stage 8    write XML       scene.xml
RIGID_PIPELINE_OK
```

The driver also writes **`pose_overlay.png`** — each object's estimated pose drawn
back on the input RGB as FoundationPose's posed 3-D box + XYZ axes. A tight box
== correct pose+scale; an oversized box flags a scale error. (This overlay is how
the early carton over-scale — 64 cm — was caught and fixed via depth-outlier
removal in stage 5.) The committed reference
`examples/push_real2sim/pose_overlay_example.png` is the bundled **`0103_push_0`**
overlay (tight boxes on both cartons).

Memory note: SAM-3D (~20 GB) and FoundationPose are run in **two phases**
(reconstruct all objects → free SAM-3D → pose all objects) so they are never
peak-resident together — loading both at once OOMs a 32 GB card.

**Outputs** in `--out_dir`, per object: `{obj}_mask.npy`, `{obj}_cropped.png`
(RGBA), `{obj}_scaled.obj` (metric, decimated for MuJoCo), and `scene.xml`.

**Validated in MuJoCo** — the scene loads and simulates:

```python
import mujoco
m = mujoco.MjModel.from_xml_path("/tmp/rigid_demo/scene.xml")
d = mujoco.MjData(m)
for _ in range(200): mujoco.mj_step(m, d)   # 4 bodies, 2 meshes — no crash
```

Two honest caveats this example surfaces:
- **Scale (stage 5) is the roughest stage.** The bbox-diagonal ratio is sensitive
  to mask-edge depth bleed; the driver removes statistical outliers from the
  masked depth cloud first (this fixed an early 64 cm → 26.5 cm carton over-scale
  — verify via `pose_overlay.png`). It's still approximate — single-view depth +
  a generated mesh make metric scale hard, which is why the original sometimes hard-coded
  it. The driver also adds a **"settle on the table"** step (shift each object so
  its lowest vertex rests at z=0) so the scene is physically plausible regardless;
  for exact sizes, refine stage 5 (e.g. ICP fit) or supply known dimensions.
- **World frame from a plane fit, not robot calibration.** With no camera→robot
  extrinsics, stage 7 fits the table plane from depth to get a z-up world. Supply
  a calibration for true robot-frame placement.

---

## Validating perception (pose + shape) — `scripts/visualize_poses.py`

Before trusting a reconstructed scene, check that the **estimated 6-DoF poses**
and the **completed meshes** actually agree with the sensor data. This tool
back-projects the RGB-D frame into a coloured 3-D point cloud and overlays, per
object, the completed mesh placed at its estimated pose plus the observed (masked)
points. If perception is correct, the observed points lie on the posed mesh's
visible surface and the mesh completes the occluded back.

Everything is shown in the **robot base frame** by default: the camera→robot
extrinsic transforms both the cloud and the poses (`mujoco_cam = camera_to_robot @
6d_cam`). The extrinsic is taken from `--extrinsic`, else the **aligned packaged
calibration** (`simpact.real2sim.transform_6d.get_camera_to_robot` — the committed
0103 `optimized_transform*`, the same the real2sim pipeline uses). The per-trial
`_mujoco_cam` files are deliberately **not** used to derive the frame: for trials
recorded before the 0103 calibration they are ~15–18 cm off, which would put the
cloud and meshes in the wrong place. Use `--frame camera` to stay in the camera
frame; only trials captured with the committed calibration (e.g. `0103_push_0`)
line up in the robot frame.

**Inputs** (a real2sim trial dir, e.g. a recorded trial dir):
`camera{cam}_rgb.npy`, `camera{cam}_depth.npy` (metres), per object
`{obj}_scaled.obj`, `{obj}_6d_cam{cam}.txt` and `camera{cam}_mask_{obj}.npy`,
intrinsics `cam_utils/cam{cam}_K.txt`. The camera→robot extrinsic comes from the
packaged calibration (no per-trial file needed); override with `--extrinsic`.

**Run** (robot frame, all objects auto-discovered):
```bash
python scripts/visualize_poses.py \
  --data_dir /path/to/data/0103_push_0 --cam 1 \
  --frame robot --out_dir /tmp/pose_vis
```

**Interactive 3D** (rotate/zoom the cloud + textured posed meshes in an Open3D
window; needs a local display / X forwarding):
```bash
python scripts/visualize_poses.py \
  --data_dir .../0103_push_0 --cam 1 --frame robot --interactive
```

Useful flags: `--frame {robot,camera}`; `--extrinsic <cam→robot 4x4.txt>`;
`--objects "white coconut milk carton. blue coconut milk carton."` (subset);
`--crop 0.12` (tighten the cloud for a clearer view); `--max_scene_pts`,
`--mesh_pts`, `--k_file`.

**Outputs** (in `--out_dir`, in the chosen frame):
- `overlay.png` — three views (robot 3/4 + top + side; or camera/top/side):
  scene cloud (RGB) + observed points + posed mesh, per object.
- `scene_cloud.ply` and `{obj}_posed.obj` — open them together in MeshLab/Open3D
  for full-resolution interactive inspection (`--interactive` opens this directly).
- printed **fit**: mean observed→mesh distance per object (lower = better).

**Reading it:** a ~cm-or-better fit and observed points hugging the mesh's front
face mean pose + shape are correct (the fit is extrinsic-invariant — both the
observed points and the posed mesh are transformed by the same camera→robot, so it
measures the estimate, not the frame). Example (`0103_push_0`): blue carton
**15.0 mm**, white carton **20.2 mm**; both meshes upright (<2°) and co-planar on
the table. (A large fit, an offset mesh, or a tilted/floating mesh flags a bad
pose or scale.)

> This validates both the estimate *and*, in the robot frame, the world placement.
> A trial recorded with a stale calibration (e.g. `1111_push_0`, pre-0103) shows a
> small fit but a tilted, table-penetrating placement that topples in sim — that is
> a calibration/placement issue, not an estimation one. Use a trial captured with
> the committed extrinsics (`0103_push_0`) for a placement that holds up in sim.

## Multi-camera note

The rig captured two cameras (`cam0`, `cam1`) and ran the model stages on **one**
(usually `cam1`). The second view is available for disambiguation/occlusion but is
not required; a single RGB-D view is sufficient for the full pipeline.
