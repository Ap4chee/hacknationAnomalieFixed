from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore
import torch

import cv2
import numpy as np
from pathlib import Path


def aggregate_tile_predictions(predictions, tiles, original_shape, tile_size=128):
    """Agreguj mapy anomalii z kafelków do jednej mapy dla całego obrazu."""
    h, w = original_shape[:2]

    # Mapa sumująca anomalie i mapa liczników (dla uśrednienia w overlapach)
    anomaly_sum = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for pred, tile in zip(predictions, tiles):
        x, y = tile.x, tile.y
        anomaly_map = pred.anomaly_map.squeeze().cpu().numpy()

        # Dodaj do sumy i zwiększ licznik
        anomaly_sum[y : y + tile_size, x : x + tile_size] += anomaly_map
        count_map[y : y + tile_size, x : x + tile_size] += 1

    # Uśrednij w miejscach overlapa
    count_map[count_map == 0] = 1  # Unikaj dzielenia przez zero
    aggregated = anomaly_sum / count_map

    return aggregated


def compute_global_anomaly_score(aggregated_map, method="percentile"):
    """Oblicz globalny wynik anomalii dla całego obrazu."""
    if method == "percentile":
        return np.percentile(aggregated_map, 99)  # Top 1% pikseli
    elif method == "mean_top_k":
        flat = aggregated_map.flatten()
        top_k = int(len(flat) * 0.01)
        return np.mean(np.sort(flat)[-top_k:])
    else:
        return np.max(aggregated_map)


class PatchcoreAnomalyRunner:
    def __init__(
        self,
        image_path: str,
        ckpt_path: str = "results/Patchcore/contraband_xray/latest/weights/lightning/model.ckpt",
        out_dir: str = "anomaly_output",
    ):
        """
        Callable class for running Patchcore anomaly detection on a single image
        (or a folder) and saving heatmaps and overlays.

        :param image_path: Path to a .bmp image or directory of images.
        :param ckpt_path: Path to Patchcore checkpoint.
        :param out_dir: Output directory for results.
        """
        self.image_path = image_path
        self.ckpt_path = ckpt_path
        self.out_dir = Path(out_dir)

        # Prepare output directory
        self.out_dir.mkdir(exist_ok=True)

        # Initialize model & engine
        self.model = Patchcore(
            backbone="resnet18",
            pre_trained=False,
            coreset_sampling_ratio=0.01,
        )
        self.engine = Engine()

    def __call__(self):
        """
        Run anomaly detection and save results.
        Returns list of result dicts for each prediction.
        """
        dataset = PredictDataset(path=self.image_path)

        predictions = self.engine.predict(
            model=self.model,
            dataset=dataset,
            ckpt_path=self.ckpt_path,
        )

        if predictions is None:
            print("No predictions returned.")
            return []

        results = []

        for i, prediction in enumerate(predictions, start=1):
            # Fix image_path formatting from anomalib (sometimes list)
            image_path = (
                prediction.image_path[0]
                if isinstance(prediction.image_path, list)
                else prediction.image_path
            )
            stem = Path(image_path).stem

            # ---- Load original image (grayscale) ----
            original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # ---- Convert anomaly map → NumPy 2D ----
            anom = prediction.anomaly_map
            if isinstance(anom, torch.Tensor):
                anom = anom.detach().cpu()
                if anom.ndim == 4:
                    anom = anom[0, 0]
                elif anom.ndim == 3:
                    anom = anom[0]
                anom = anom.numpy()
            # Filtruj po score
            if prediction.pred_score < 0.45:
                continue

            # ---- Zapisz heatmapę jako PNG ----
            # Normalizuj do 0-255
            anom_normalized = np.clip(anom, 0, 1)
            anom_uint8 = (anom_normalized * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(anom_uint8, cv2.COLORMAP_JET)
            
            heatmap_path = self.out_dir / f"{stem}_heatmap.png"
            cv2.imwrite(str(heatmap_path), heatmap_color)
            
            # Zapisz też surową mapę jako grayscale PNG (do agregacji)
            anomaly_path = self.out_dir / f"{stem}_anomaly.png"
            cv2.imwrite(str(anomaly_path), anom_uint8)

            # Print results
            print(f"[{i}] {image_path}")
            print(f"  label: {prediction.pred_label}")
            print(f"  score: {float(prediction.pred_score):.4f}")
            print(f"  saved heatmap: {heatmap_path}")
            print(f"  saved anomaly: {anomaly_path}")

            results.append(
                {
                    "index": i,
                    "image_path": image_path,
                    "label": prediction.pred_label,
                    "score": float(prediction.pred_score),
                    "anomaly_path": str(anomaly_path),
                }
            )

        return results


if __name__ == "__main__":
    # Example CLI usage:
    # python script.py cropped_images_bad/202511190100crop_0040_x1152_y256.bmp
    import argparse

    parser = argparse.ArgumentParser(description="Run Patchcore anomaly detection.")
    parser.add_argument("image_path", help="Path to image or directory.")
    parser.add_argument(
        "--ckpt",
        default="results/Patchcore/contraband_xray/latest/weights/lightning/model.ckpt",
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "--out-dir",
        default="anomaly_output",
        help="Output directory.",
    )

    args = parser.parse_args()

    runner = PatchcoreAnomalyRunner(
        image_path=args.image_path,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
    )
    runner()
