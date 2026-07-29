"""Grounded-SAM-2 weighted smoke in simpact's shared .venv: GroundingDINO (HF
transformers) + SAM2 -> masks from a language prompt, on the committed example scene image.
Confirms segmentation runs in the SAME env as SAM-3D + FoundationPose.

No Grounded-SAM-2 repo clone needed: GroundingDINO is a transformers model and
SAM2 + its configs ship in the `sam2` wheel. Only the SAM2 checkpoint is external.

Env:
  SIMPACT_SAM2_CHECKPOINT  path to sam2.1_hiera_large.pt (default below)
Run:  .venv/bin/python scripts/smoke_gsam2.py
"""
import os
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

DEVICE = "cuda"
SAM2_CKPT = os.environ.get("SIMPACT_SAM2_CHECKPOINT",
                           os.path.expanduser("~/sam2/checkpoints/sam2.1_hiera_large.pt"))
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"     # bundled in the sam2 wheel
GDINO = "IDEA-Research/grounding-dino-tiny"
TEXT = os.environ.get("SIMPACT_SMOKE_TEXT",
                      "white coconut milk carton. blue milk carton.")  # lowercase, dot-sep
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.environ.get("SIMPACT_SMOKE_IMG",
                     f"{_REPO}/examples/push_real2sim/0103_push_0/capture/camera1_rgb.png")

img_np = (np.array(Image.open(IMG).convert("RGB")) if IMG.lower().endswith((".png", ".jpg", ".jpeg"))
          else np.load(IMG))
image = Image.fromarray(img_np)
print(f"[input] {os.path.basename(IMG)} {img_np.shape}  prompt={TEXT!r}")

predictor = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_CKPT, device=DEVICE))
processor = AutoProcessor.from_pretrained(GDINO)
gdino = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).to(DEVICE)
print("[load] SAM2 + GroundingDINO ready")

torch.cuda.reset_peak_memory_stats()
with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):   # scoped, not global
    inputs = processor(images=image, text=TEXT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = gdino(**inputs)
    # transformers 5.x: box_threshold->threshold; string labels under text_labels
    res = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.4, text_threshold=0.3,
        target_sizes=[image.size[::-1]])[0]
    boxes = res["boxes"].cpu().numpy()
    predictor.set_image(np.array(image.convert("RGB")))
    masks, scores, _ = predictor.predict(point_coords=None, point_labels=None,
                                         box=boxes, multimask_output=False)
if masks.ndim == 4:
    masks = masks.squeeze(1)
labels = res.get("text_labels", res.get("labels"))
print(f"[detect] labels={labels} scores={[f'{s:.2f}' for s in res['scores'].tolist()]}")
print(f"[masks ] shape={masks.shape} px/obj={[int(m.sum()) for m in masks.astype(bool)]}")
print(f"[gpu   ] peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | torch {torch.__version__}")
print("GSAM2_SMOKE_OK")
