#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from PIL import Image

from ImageProcessor.XRayImage import XRayImage
from predict import PatchcoreAnomalyRunner


def parse_coords_from_name(overlay_path: Path):
    """
    Expects filename like:
    '48001F003202511190033 czarno_65_512_994_overlay.png'
                                          ^   ^
                                          |   +-- y
                                          +------ x
    """
    stem = overlay_path.stem  # without extension
    parts = stem.split("_")
    # ['48001F003202511190033 czarno', '65', '512', '994', 'overlay']
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {stem}")

    x = int(parts[-3])
    y = int(parts[-2])
    return x, y


def apply_all_overlays(big_image_path: Path, overlays_dir: Path, output_path: Path):
    """Open big X-ray, paste all *_overlay.png files, save one BMP with all heatmaps."""
    # Collect overlays
    overlay_paths = sorted(
        p for p in overlays_dir.iterdir()
        if p.is_file() and p.name.endswith("_heatmap.png")
    )

    if not overlay_paths:
        print(f"No overlay PNGs found in {overlays_dir}")
        return

    # Open base image as RGB (for colored heatmaps)
    big_img = Image.open(big_image_path).convert("RGB")
    bw, bh = big_img.size

    for overlay_path in overlay_paths:
        x, y = parse_coords_from_name(overlay_path)
        overlay = Image.open(overlay_path).convert("RGBA")
        ow, oh = overlay.size

        # Sanity check
        if x + ow > bw or y + oh > bh:
            print(
                f"Warning: overlay {overlay_path.name} outside bounds "
                f"(big={bw}x{bh}, at ({x},{y}), size={ow}x{oh}). Skipping."
            )
            continue

        # Adjust opacity (0.0 = fully transparent, 1.0 = fully opaque)
        opacity = 0.5  # Change this value to adjust transparency
        alpha = overlay.split()[3]  # Get alpha channel
        alpha = alpha.point(lambda p: int(p * opacity))  # Scale alpha values
        overlay.putalpha(alpha)  # Apply modified alpha

        # Paste using alpha channel
        big_img.paste(overlay, (x, y), overlay)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    big_img.save(output_path, format="BMP")
    print(f"Saved final image with overlays: {output_path}")


def process_image(image_path: str, output_dir: str = "processed_image") -> None:
    """Load BMP, apply filters, generate and save tiles, run Patchcore."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    if not image_path.lower().endswith(".bmp"):
        raise ValueError(f"Expected a .bmp file, got: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    img = XRayImage(src=image_path)
    img.applyFilters()
    img.generateTiles()
    img.saveTiles(output_dir)

    try:
        PatchcoreAnomalyRunner(image_path=output_dir)()
    except Exception as exc:
        print(f"Anomaly detection error: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply filters to an X-ray BMP image and generate tiles."
    )
    parser.add_argument("image_path", help="Path to the input .bmp image")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="processed_image",
        help="Directory to save processed tiles (default: processed_image)",
    )

    args = parser.parse_args()
    image_path = Path(args.image_path)

    try:
        process_image(str(image_path), args.output_dir)
        print(f"Processed '{image_path}' -> '{args.output_dir}'")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Remove tiles after anomaly detection, if you want
    import shutil
    shutil.rmtree(args.output_dir, ignore_errors=True)

    # Now apply all overlays to the original image
    overlays_dir = Path("anomaly_output")
    final_output_dir = Path("final_output")
    final_output_dir.mkdir(parents=True, exist_ok=True)

    out_name = image_path.stem + "_with_overlays.bmp"
    output_path = final_output_dir / out_name

    apply_all_overlays(
        big_image_path=image_path,
        overlays_dir=overlays_dir,
        output_path=output_path,
    )
    #shutil.rmtree("anomaly_output", ignore_errors=True)


if __name__ == "__main__":
    main()
