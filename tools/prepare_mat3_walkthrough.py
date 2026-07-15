"""Prepare web-sized, public-safe assets for the authorized Mat3 walkthrough."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

RIGHTS = (
    "Copyright © 2026 Hyeon Sang Jeon. All rights reserved. Repository "
    "demonstration/archive only; no reuse/redistribution/derivatives."
)
ASSETS = {
    "blurred": ("input", "Mat3_B1.png"),
    "noisy": ("input", "Mat3_N.png"),
    "registered_noisy": ("output", "registered_noisy.png"),
    "features": ("output", "features.png"),
    "initial_regions": ("output", "initial_regions.png"),
    "merged_regions": ("output", "merged_regions.png"),
    "kernels": ("output", "kernels.png"),
    "deconvolved": ("output", "deconvolved.png"),
    "restored": ("output", "restored.png"),
}


def prepare_walkthrough(
    *,
    input_dir: Path,
    output_dir: Path,
    hpo_path: Path,
    run_path: Path,
    docs_output_dir: Path,
    max_dimension: int = 512,
) -> Path:
    """Write resized public assets after verifying the recorded optimized preset."""
    if max_dimension < 64:
        raise ValueError("max_dimension must be at least 64")
    hpo = json.loads(hpo_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    _verify_hpo(run, hpo)

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    source_roots = {"input": input_dir, "output": run_path.parent}
    prepared: dict[str, np.ndarray] = {}
    source_sizes: dict[str, dict[str, int]] = {}
    for identifier, (kind, filename) in ASSETS.items():
        path = source_roots[kind] / filename
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read Mat3 walkthrough asset {identifier}")
        source_sizes[identifier] = _size(image)
        prepared[identifier] = _resize(image, max_dimension)
        _write_png(output_dir / f"{identifier}.png", prepared[identifier])
        _write_webp(docs_output_dir / f"{identifier}.webp", prepared[identifier])

    comparison = _comparison(
        prepared["blurred"], prepared["noisy"], prepared["restored"]
    )
    _write_webp(output_dir / "comparison.webp", comparison)
    _write_webp(docs_output_dir / "comparison.webp", comparison)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "id": "mat3-living-room-walkthrough",
        "rights": RIGHTS,
        "metric_type": "no-reference-proxy",
        "metric_note": (
            "No clean ground truth is available. The HPO reference is the "
            "RANSAC-registered, non-local-means-denoised noisy auxiliary; "
            "PSNR is not reported."
        ),
        "source_sizes": source_sizes,
        "web_max_dimension": max_dimension,
        "selected_hpo": hpo["selected"],
        "proxy_metrics": hpo["metrics"]["optimized_final"],
        "intermediate_assets": list(ASSETS),
        "asset_checksums": {
            path.relative_to(output_dir).as_posix(): _sha256(path)
            for path in sorted(output_dir.glob("*.png"))
        },
        "comparison_checksum": _sha256(output_dir / "comparison.webp"),
    }
    metadata_path = output_dir / "walkthrough.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata_path


def _verify_hpo(run: Mapping[str, object], hpo: Mapping[str, object]) -> None:
    config = run.get("config")
    selected = hpo.get("selected")
    if not isinstance(config, dict) or not isinstance(selected, dict):
        raise ValueError("Mat3 run and HPO records must contain configuration objects")
    expected = {
        "color_labels": selected["segmentation"]["color_labels"],
        "graph_cut_smoothness": selected["segmentation"]["graph_cut_smoothness"],
        "disparity_threshold": selected["segmentation"]["disparity_threshold"],
        "max_regions": selected["segmentation"]["max_regions"],
        "minimum_region_fraction": selected["segmentation"]["minimum_region_fraction"],
        "kernel_size": selected["kernel"]["kernel_size"],
        "patch_size": selected["kernel"]["patch_size"],
        "tikhonov": selected["kernel"]["tikhonov"],
        "kurtosis_min": selected["kernel"]["kurtosis_min"],
        "data_weight": selected["restoration"]["data_weight"],
        "beta_max": selected["restoration"]["beta_max"],
        "feather_sigma": selected["restoration"]["feather_sigma"],
        "unsharp_sigma": selected["unsharp"]["sigma"],
        "unsharp_amount": selected["unsharp"]["amount"],
        "unsharp_threshold": selected["unsharp"]["threshold"],
        "detail_denoise_strength": selected["guided_detail_fusion"]["denoise_strength"],
        "detail_fusion_sigma": selected["guided_detail_fusion"]["sigma"],
        "detail_fusion_amount": selected["guided_detail_fusion"]["amount"],
        "detail_fusion_tolerance": selected["guided_detail_fusion"]["tolerance"],
        "detail_fusion_threshold": selected["guided_detail_fusion"]["threshold"],
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Mat3 output does not use the selected HPO preset: {mismatches}")


def _resize(image: np.ndarray, maximum: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _comparison(blurred: np.ndarray, noisy: np.ndarray, restored: np.ndarray) -> np.ndarray:
    labeled = []
    for image, label in zip(
        (blurred, noisy, restored), ("Blurred", "Noisy", "Restored"), strict=True
    ):
        preview = image.copy()
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(
            preview,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        labeled.append(preview)
    return np.concatenate(labeled, axis=1)


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise OSError(f"failed to write {path}")


def _write_webp(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_WEBP_QUALITY, 88]):
        raise OSError(f"failed to write {path}")


def _size(image: np.ndarray) -> dict[str, int]:
    return {"width": int(image.shape[1]), "height": int(image.shape[0])}


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Prepare public-safe Mat3 walkthrough assets and checksummed metadata."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hpo", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--docs-output-dir", required=True, type=Path)
    parser.add_argument("--max-dimension", type=int, default=512)
    args = parser.parse_args()
    print(
        prepare_walkthrough(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            hpo_path=args.hpo,
            run_path=args.run,
            docs_output_dir=args.docs_output_dir,
            max_dimension=args.max_dimension,
        )
    )


if __name__ == "__main__":
    main()
