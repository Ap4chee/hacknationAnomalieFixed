from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.visualization import visualize_anomaly_map
import torch

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from pathlib import Path


def save_overlay(original_img, anomaly_map, out_path):
    """
    Creates a heatmap overlay with alpha channel based on anomaly intensity.
    Saves to out_path as PNG with transparency.
    """
    anomaly_clipped = np.clip(anomaly_map, 0.0, 1.0)
    anomaly_uint8 = (anomaly_clipped * 255).astype(np.uint8)

    # Apply "JET" heatmap for visualization
    heatmap_color = cv2.applyColorMap(anomaly_uint8, cv2.COLORMAP_JET)

    # Create alpha channel based on anomaly intensity
    # Areas with higher anomaly scores get higher opacity
    alpha = (anomaly_clipped * 255).astype(np.uint8)

    # Add alpha channel to heatmap (BGRA format)
    heatmap_rgba = np.dstack((heatmap_color, alpha))

    cv2.imwrite(str(out_path), heatmap_rgba)


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
            pre_trained=True,
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
            if prediction.pred_score < 0.76:
                continue

            # Create sample anomaly map

            # Visualize with default settings
            #vis = visualize_anomaly_map(anom)

            # Customize visualization
            vis = visualize_anomaly_map(
                prediction.anomaly_map.detach().cpu().numpy()[0, 0],
                colormap=True,      # Apply colormap
                normalize=True      # Normalize values to [0, 255]
            )

            # vis_img is a numpy array
            vis = self.out_dir / f"{stem}_vis.png"
            cv2.imwrite(str(vis), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            # ---- Save heatmap only ----
            heatmap_path = self.out_dir / f"{stem}_heatmap.png"
            plt.imsave(heatmap_path, anom, cmap="jet")

            # ---- Save overlay heatmap-on-original ----
            overlay_path = self.out_dir / f"{stem}_overlay.png"
            save_overlay(original, anom, overlay_path)

            # Print results
            print(f"[{i}] {image_path}")
            print(f"  label: {prediction.pred_label}")
            print(f"  score: {float(prediction.pred_score):.4f}")
            print(f"  saved heatmap: {heatmap_path}")
            print(f"  saved overlay: {overlay_path}")

            results.append(
                {
                    "index": i,
                    "image_path": image_path,
                    "label": prediction.pred_label,
                    "score": float(prediction.pred_score),
                    "heatmap_path": str(heatmap_path),
                    "overlay_path": str(overlay_path),
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
