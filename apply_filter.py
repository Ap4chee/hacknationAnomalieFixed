#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

from ImageProcessor.XRayImage import XRayImage
from predict import PatchcoreAnomalyRunner


def parse_coords_from_name(file_path: Path):
    """
    Expects filename like:
    '48001F003202511190033 czarno_65_512_994_anomaly.npy'
                                          ^   ^
                                          |   +-- y
                                          +------ x
    """
    stem = file_path.stem  # without extension
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {stem}")

    x = int(parts[-3])
    y = int(parts[-2])
    return x, y


def apply_all_heatmaps(big_image_path: Path, anomaly_dir: Path, output_path: Path, threshold: float = 0.45):
    """
    Agreguje wszystkie mapy anomalii (_anomaly.png) do jednej mapy,
    uśrednia nakładające się obszary, i nakłada heatmapę tylko gdzie anomaly > threshold.
    """
    anomaly_paths = sorted(
        p for p in anomaly_dir.iterdir()
        if p.is_file() and p.name.endswith("_anomaly.png")
    )

    if not anomaly_paths:
        print(f"No anomaly PNG files found in {anomaly_dir}")
        return

    # Wczytaj oryginalny obraz
    big_img = cv2.imread(str(big_image_path))
    if big_img is None:
        print(f"Failed to load image: {big_image_path}")
        return
    bh, bw = big_img.shape[:2]

    # Mapa do agregacji anomalii
    anomaly_sum = np.zeros((bh, bw), dtype=np.float32)
    count_map = np.zeros((bh, bw), dtype=np.float32)

    for anomaly_path in anomaly_paths:
        try:
            x, y = parse_coords_from_name(anomaly_path)
        except ValueError as e:
            print(f"Skipping {anomaly_path.name}: {e}")
            continue

        # Wczytaj mapę anomalii (grayscale PNG, 0-255 -> 0-1)
        anomaly_map = cv2.imread(str(anomaly_path), cv2.IMREAD_GRAYSCALE)
        if anomaly_map is None:
            print(f"Failed to load {anomaly_path}")
            continue
        anomaly_map = anomaly_map.astype(np.float32) / 255.0
        
        oh, ow = anomaly_map.shape[:2]

        # Oblicz granice
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(bw, x + ow)
        y2 = min(bh, y + oh)

        ox1 = max(0, -x)
        oy1 = max(0, -y)
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        if x2 <= x1 or y2 <= y1:
            continue

        # Dodaj do sumy
        anomaly_sum[y1:y2, x1:x2] += anomaly_map[oy1:oy2, ox1:ox2]
        count_map[y1:y2, x1:x2] += 1

    # Uśrednij
    count_map[count_map == 0] = 1
    aggregated = anomaly_sum / count_map
    
    # Debug - pokaż zakres wartości
    print(f"Aggregated anomaly range: {aggregated.min():.3f} - {aggregated.max():.3f}")

    # NIE normalizuj do max - używaj surowych wartości 0-1
    # Tylko powyżej threshold
    mask = aggregated > threshold
    
    if not np.any(mask):
        print(f"No anomalies above threshold {threshold} found.")
        cv2.imwrite(str(output_path), big_img)
        return
    
    print(f"Pixels above threshold: {np.sum(mask)} ({100*np.sum(mask)/(bh*bw):.2f}%)")

    # Heatmapa - cała mapa, nie tylko powyżej threshold
    heatmap_values = np.clip((aggregated - threshold) / (1 - threshold), 0, 1)
    heatmap_uint8 = (heatmap_values * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blending - podkręć opacity
    min_opacity = 0.25  # niebieskie
    max_opacity = 0.9   # czerwone
    alpha = (heatmap_values * (max_opacity - min_opacity) + min_opacity)[:, :, np.newaxis]

    result = big_img.copy().astype(np.float32)
    result = (1 - alpha) * big_img + alpha * heatmap_color
    result = result.astype(np.uint8)

    cv2.imwrite(str(output_path), result)
    print(f"Saved aggregated heatmap to {output_path}")


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
    
    print(f"Generated {len(img.tiles)} tiles")

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

    # Nie usuwaj kafelków - zostaw do analizy
    # import shutil
    # shutil.rmtree(args.output_dir, ignore_errors=True)

    # Agreguj heatmapy na oryginalny obraz
    anomaly_dir = Path("anomaly_output")
    final_output_dir = Path("final_output")
    final_output_dir.mkdir(parents=True, exist_ok=True)

    out_name = image_path.stem + "_result.bmp"
    output_path = final_output_dir / out_name

    apply_all_heatmaps(
        big_image_path=image_path,
        anomaly_dir=anomaly_dir,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
