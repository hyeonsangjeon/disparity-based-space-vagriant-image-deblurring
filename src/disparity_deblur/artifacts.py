from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .image_io import write_png, write_rgb24
from .models import DeblurResult, InputSpec, KernelEstimate


class ArtifactWriter:
    """Persist full-resolution pipeline outputs and compact diagnostics."""

    def __init__(self, output_dir: str | Path) -> None:
        """Target all generated artifacts at one output directory."""

        self.output_dir = Path(output_dir)

    def write(
        self,
        result: DeblurResult,
        inputs: InputSpec,
        config: PipelineConfig,
    ) -> Path:
        """Write RGB, label, kernel, mask, and metadata artifacts."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        registration = result.registration
        segmentation = result.segmentation

        write_rgb24(self.output_dir / "input_blurred.raw", result.blurred)
        write_png(self.output_dir / "input_blurred.png", result.blurred)
        write_rgb24(self.output_dir / "input_noisy.raw", result.noisy)
        write_png(self.output_dir / "input_noisy.png", result.noisy)
        write_rgb24(
            self.output_dir / "registered_noisy.raw", registration.registered_noisy
        )
        write_png(
            self.output_dir / "registered_noisy.png", registration.registered_noisy
        )
        write_rgb24(self.output_dir / "deconvolved.raw", result.deconvolved)
        write_png(self.output_dir / "deconvolved.png", result.deconvolved)
        write_rgb24(self.output_dir / "restored.raw", result.restored)
        write_png(self.output_dir / "restored.png", result.restored)
        write_png(
            self.output_dir / "initial_regions.png",
            _label_preview(segmentation.initial_labels),
        )
        write_png(
            self.output_dir / "merged_regions.png",
            _label_preview(segmentation.merged_labels),
        )
        write_png(
            self.output_dir / "features.png",
            _feature_preview(
                result.blurred,
                registration.blur_points,
                registration.disparities,
            ),
        )
        write_png(self.output_dir / "kernels.png", _kernel_preview(result.kernels))
        write_png(
            self.output_dir / "registration_valid.png",
            _mask_preview(registration.valid_mask),
        )

        np.save(self.output_dir / "initial_regions.npy", segmentation.initial_labels)
        np.save(self.output_dir / "merged_regions.npy", segmentation.merged_labels)
        np.save(self.output_dir / "registration_valid.npy", registration.valid_mask)
        np.save(
            self.output_dir / "kernels.npy",
            np.stack([estimate.kernel for estimate in result.kernels]),
        )

        metadata_path = self.output_dir / "run.json"
        metadata_path.write_text(
            json.dumps(
                _metadata(result, inputs, config),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return metadata_path


def _metadata(
    result: DeblurResult,
    inputs: InputSpec,
    config: PipelineConfig,
) -> dict[str, object]:
    """Build machine-readable run metadata with independent dimensions."""

    registration = result.registration
    segmentation = result.segmentation
    config_values = asdict(config)
    config_values["effective_boundary_padding"] = config.effective_boundary_padding
    return {
        "input": {
            "blurred": str(inputs.blurred_path.resolve()),
            "noisy": str(inputs.noisy_path.resolve()),
            "blurred_format": inputs.blurred_path.suffix.lower().lstrip("."),
            "noisy_format": inputs.noisy_path.suffix.lower().lstrip("."),
            "height": inputs.height,
            "width": inputs.width,
        },
        "config": config_values,
        "registration": {
            "detected_corners": registration.detected_count,
            "ransac_inliers": len(registration.blur_points),
            "homography": registration.homography.tolist(),
            "median_disparity": (
                np.median(registration.disparities, axis=0).tolist()
                if len(registration.disparities)
                else [0.0, 0.0]
            ),
        },
        "regions": [
            {
                "id": region,
                "pixels": int(np.sum(segmentation.merged_labels == region)),
                "disparity": segmentation.region_disparities[region].tolist(),
                "feature_count": int(segmentation.region_feature_counts[region]),
                "kernel_kurtosis": result.kernels[region].kurtosis,
                "kernel_replaced_from": result.kernels[region].replaced_from,
                "kernel_patch": result.kernels[region].patch_bounds,
            }
            for region in range(int(segmentation.merged_labels.max()) + 1)
        ],
    }


def _label_preview(labels: np.ndarray) -> np.ndarray:
    """Colorize labels and draw their boundaries in black."""

    count = int(labels.max()) + 1
    colors = np.empty((count, 3), dtype=np.float32)
    for label in range(count):
        hue = int(round(179 * label / max(count, 1)))
        hsv = np.array([[[hue, 180, 230]]], dtype=np.uint8)
        colors[label] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0] / 255.0
    preview = colors[labels]
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    preview[boundary] = 0.0
    return preview


def _feature_preview(
    image: np.ndarray,
    points: np.ndarray,
    disparities: np.ndarray,
) -> np.ndarray:
    """Draw disparity vectors over the blurred input."""

    preview = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    for point, disparity in zip(points, disparities, strict=True):
        start = tuple(np.rint(point).astype(int))
        end = tuple(np.rint(point + disparity * 8.0).astype(int))
        cv2.arrowedLine(
            preview,
            start,
            end,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
            tipLength=0.3,
        )
    return preview.astype(np.float32) / 255.0


def _kernel_preview(estimates: tuple[KernelEstimate, ...]) -> np.ndarray:
    """Tile normalized regional PSFs into a compact diagnostic image."""

    if not estimates:
        raise ValueError("no kernels to preview")
    tiles = []
    for estimate in estimates:
        normalized = estimate.kernel / max(estimate.kernel.max(), 1e-12)
        tile = cv2.resize(normalized, (192, 192), interpolation=cv2.INTER_NEAREST)
        tiles.append(np.repeat(tile[..., None], 3, axis=2))
    columns = min(3, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    canvas = np.zeros((rows * 192, columns * 192, 3), dtype=np.float64)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[row * 192 : (row + 1) * 192, column * 192 : (column + 1) * 192] = tile
    return canvas


def _mask_preview(mask: np.ndarray) -> np.ndarray:
    """Convert a 2D registration mask into a three-channel preview."""

    return np.repeat(mask[..., None], 3, axis=2).astype(np.float32)
