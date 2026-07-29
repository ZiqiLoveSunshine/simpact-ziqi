"""Stage 3 of the real2sim pipeline: extract per-object masks from the
Grounded-SAM-2 JSON output.

Decodes the RLE segmentation for each detected object into a full-image binary
mask, and (optionally) a tight RGBA crop suitable for single-image-to-3D
reconstruction (stage 4). Ported from the original ``real2sim/mask_extraction.py``; the
decode/crop logic is unchanged, but the CLI is wrapped in an importable
``extract_masks`` function and paths route through arguments rather than CWD.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util


def extract_masks(
    json_path,
    output_prefix,
    *,
    png=False,
    crop=False,
    padding=10,
    image_path=None,
):
    """Decode masks from a Grounded-SAM-2 JSON file.

    Args:
        json_path: Path to the ``*_gsam2.json`` produced by stage 2.
        output_prefix: Output path prefix; per-object files are written as
            ``{output_prefix}_{class_name}.npy`` (and ``.png`` / ``_cropped.png``).
        png: Also write each full-image mask as a visualisable PNG.
        crop: Also write a tight RGBA crop of the source image (padded bbox).
        padding: Pixels of padding around each bbox when cropping.
        image_path: Override for the source image. Defaults to the ``image_path``
            recorded in the JSON (which the original stored relative to its CWD).

    Returns:
        List of ``(class_name, mask_path)`` for each decoded annotation.
    """
    json_path = Path(json_path)
    output_prefix = Path(output_prefix)

    with open(json_path, "r") as f:
        data = json.load(f)

    img_width = data["img_width"]
    img_height = data["img_height"]
    src_image = image_path if image_path is not None else data["image_path"]

    written = []
    for annotation in data["annotations"]:
        rle_data = annotation["segmentation"]
        class_name = annotation["class_name"]
        print(f"Processing class: {class_name}")

        mask_full_image = mask_util.decode(rle_data)
        mask_path = output_prefix.with_name(f"{output_prefix.name}_{class_name}.npy")
        np.save(mask_path, mask_full_image)
        written.append((class_name, mask_path))

        if png:
            mask_pil = Image.fromarray(mask_full_image.astype(np.uint8) * 255)
            png_path = output_prefix.with_name(f"{output_prefix.name}_{class_name}.png")
            mask_pil.save(png_path)
            print(f"Saved the binary mask as {png_path}")

        if crop:
            xmin, ymin, xmax, ymax = map(int, annotation["bbox"])
            xmin_p = int(max(0, xmin - padding))
            ymin_p = int(max(0, ymin - padding))
            xmax_p = int(min(img_width, xmax + padding))
            ymax_p = int(min(img_height, ymax + padding))

            if str(src_image).endswith(".npy"):
                image_np = np.load(src_image)
                original_image = Image.fromarray(image_np).convert("RGB")
            else:
                original_image = Image.open(src_image).convert("RGB")

            cropped_image = original_image.crop((xmin_p, ymin_p, xmax_p, ymax_p))
            cropped_mask_np = mask_full_image[ymin_p:ymax_p, xmin_p:xmax_p]
            cropped_mask = Image.fromarray(cropped_mask_np.astype(np.uint8) * 255, "L")

            rgba_cropped_image = Image.new("RGBA", cropped_image.size, (0, 0, 0, 0))
            rgba_cropped_image.paste(cropped_image, (0, 0), mask=cropped_mask)

            crop_path = output_prefix.with_name(
                f"{output_prefix.name}_{class_name}_cropped.png"
            )
            rgba_cropped_image.save(crop_path, "PNG")
            print(f"Saved the final RGBA image to {crop_path}")

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, required=True, help="path to the JSON file")
    parser.add_argument("--output", type=str, default="mask", help="output prefix")
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--crop", action="store_true")
    parser.add_argument(
        "--image", type=str, default=None, help="override source image for --crop"
    )
    args = parser.parse_args()

    extract_masks(
        args.json,
        args.output,
        png=args.png,
        crop=args.crop,
        image_path=args.image,
    )
