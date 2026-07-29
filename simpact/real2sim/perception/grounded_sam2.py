"""GroundingDINO + SAM2 open-vocabulary segmentation backend.

Ported from the original ``real2sim/run_gsam2.py``; the call sequence is the one verified in
``scripts/run_rigid_pipeline.py`` / ``scripts/smoke_gsam2.py``.

No Grounded-SAM-2 repo clone is required: GroundingDINO loads from
``transformers`` (HF ``IDEA-Research/grounding-dino-tiny``) and SAM2 + its configs
ship in the ``sam2`` wheel. The only external artifact is the SAM2 checkpoint
(``sam2.1_hiera_large.pt``, ~900 MB), resolved from (in order):
    1. the ``sam2_checkpoint`` argument
    2. ``SIMPACT_SAM2_CHECKPOINT``
    3. ``<SIMPACT_GROUNDED_SAM2_DIR>/checkpoints/sam2.1_hiera_large.pt``

The models are loaded **once**, lazily, on first ``segment()``.
"""
import os

import numpy as np

from simpact.real2sim.perception.base import Segmenter, SegmentationResult
from simpact.real2sim.perception.repos import PerceptionRepoNotFound, get_repo_dir

DEFAULT_SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"  # bundled in the sam2 wheel
DEFAULT_GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"


class GroundedSAM2Segmenter(Segmenter):
    def __init__(
        self,
        sam2_checkpoint: str | None = None,
        sam2_config: str = DEFAULT_SAM2_CONFIG,
        grounding_model: str = DEFAULT_GROUNDING_MODEL,
        device: str = "cuda",
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ):
        self.sam2_config = sam2_config
        self.grounding_model = grounding_model
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.sam2_checkpoint = self._resolve_checkpoint(sam2_checkpoint)
        self._predictor = None
        self._processor = None
        self._gdino = None

    @staticmethod
    def _resolve_checkpoint(explicit: str | None) -> str:
        env = os.environ.get("SIMPACT_SAM2_CHECKPOINT")
        if env and not os.path.exists(env):
            env = None  # unresolvable (e.g. a /path/to/... placeholder) -> fall through
        ckpt = explicit or env
        if not ckpt:
            try:  # optional: fall back to a Grounded-SAM-2 repo's checkpoints/
                ckpt = str(get_repo_dir("grounded_sam2") / DEFAULT_SAM2_CHECKPOINT)
            except PerceptionRepoNotFound:
                raise PerceptionRepoNotFound(
                    "Grounded-SAM-2 needs the SAM2 checkpoint. Set "
                    "SIMPACT_SAM2_CHECKPOINT=/path/to/sam2.1_hiera_large.pt "
                    "(or SIMPACT_GROUNDED_SAM2_DIR with a checkpoints/ dir)."
                )
        return ckpt

    def _ensure_loaded(self):
        if self._predictor is not None:
            return
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self._predictor = SAM2ImagePredictor(
            build_sam2(self.sam2_config, self.sam2_checkpoint, device=self.device)
        )
        self._processor = AutoProcessor.from_pretrained(self.grounding_model)
        self._gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.grounding_model
        ).to(self.device)

    def segment(self, image: np.ndarray, text_prompt: str) -> SegmentationResult:
        """Segment objects named in ``text_prompt`` (e.g. ``"jar. red box."``)."""
        import torch
        from PIL import Image

        self._ensure_loaded()
        rgb = np.asarray(image)[..., :3].astype(np.uint8)
        pil = Image.fromarray(rgb)
        prompt = text_prompt.lower().strip()
        if not prompt.endswith("."):
            prompt += "."  # GroundingDINO wants lowercase, dot-separated queries

        # autocast scoped to this call (NOT process-wide) so it can't leak into
        # other models sharing the env (SAM-3D / FoundationPose).
        with torch.autocast(device_type=self.device.split(":")[0], dtype=torch.bfloat16):
            inputs = self._processor(images=pil, text=prompt, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs = self._gdino(**inputs)
            res = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,  # transformers 5.x: was box_threshold
                text_threshold=self.text_threshold,
                target_sizes=[pil.size[::-1]],
            )[0]
            boxes = res["boxes"].cpu().numpy()
            self._predictor.set_image(rgb)
            masks, _, _ = self._predictor.predict(
                point_coords=None, point_labels=None, box=boxes, multimask_output=False
            )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        labels = res.get("text_labels", res.get("labels"))  # 5.x: strings in text_labels
        scores = res["scores"].cpu().numpy()
        return SegmentationResult(
            masks=masks.astype(bool),
            labels=list(labels),
            scores=np.asarray(scores),
            boxes=np.asarray(boxes),
        )
