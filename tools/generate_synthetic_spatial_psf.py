"""Generate the project-created synthetic spatially varying PSF benchmark."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

from disparity_deblur.image_io import write_png

RIGHTS = (
    "Copyright © 2026 Hyeon Sang Jeon. All rights reserved. Included in this "
    "repository for demonstration and archival presentation only. No reuse, "
    "redistribution, or derivative use is granted."
)
MASTER_SIZE = (1024, 768)
NOISE_SEED = 20260715


def generate_dataset(max_dimension: int = 512) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Return deterministic web-sized inputs, reference, and three soft depth masks."""
    if max_dimension < 64:
        raise ValueError("max_dimension must be at least 64")
    width, height = MASTER_SIZE
    reference = _scene(width, height)
    masks = _soft_masks(width, height)
    kernels = _kernels()
    blurred = _blur_by_region(reference, masks, kernels)
    noisy = _noisy_auxiliary(reference, masks)

    images = {"blurred": blurred, "noisy": noisy, "reference": reference, **masks}
    if max_dimension < max(width, height):
        scale = max_dimension / max(width, height)
        size = (round(width * scale), round(height * scale))
        images = {
            name: cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            for name, image in images.items()
        }
        stacked_masks = np.stack([images[name] for name in masks], axis=-1)
        stacked_masks /= np.maximum(stacked_masks.sum(axis=-1, keepdims=True), 1e-8)
        images.update(
            {
                name: stacked_masks[..., index].astype(np.float32)
                for index, name in enumerate(masks)
            }
        )

    metadata: dict[str, object] = {
        "schema_version": 1,
        "generator": "tools/generate_synthetic_spatial_psf.py",
        "master_size": {"width": width, "height": height},
        "output_size": {
            "width": int(images["reference"].shape[1]),
            "height": int(images["reference"].shape[0]),
        },
        "random_seed": NOISE_SEED,
        "rights": RIGHTS,
        "regions": [
            {
                "id": "far",
                "mask": "masks/far.png",
                "translation": {"x": 1.0, "y": 0.0},
                "psf": _kernel_metadata("short-horizontal-defocus", kernels[0]),
            },
            {
                "id": "mid",
                "mask": "masks/mid.png",
                "translation": {"x": 3.0, "y": -2.0},
                "psf": _kernel_metadata("medium-diagonal", kernels[1]),
            },
            {
                "id": "near",
                "mask": "masks/near.png",
                "translation": {"x": 6.0, "y": -3.0},
                "psf": _kernel_metadata("long-curved", kernels[2]),
            },
        ],
        "noisy_auxiliary": {
            "exposure_scale": 0.72,
            "exposure_offset": 0.018,
            "poisson_photon_scale": 255.0,
            "gaussian_read_noise_sigma": 0.012,
            "seed": NOISE_SEED,
        },
    }
    return images, metadata


def write_dataset(output_dir: str | Path, max_dimension: int = 512) -> Path:
    """Write only reproducible web assets and machine-readable generation metadata."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    images, metadata = generate_dataset(max_dimension)
    for name in ("blurred", "noisy", "reference"):
        write_png(output / f"{name}.png", images[name])
    mask_dir = output / "masks"
    mask_dir.mkdir(exist_ok=True)
    for name in ("far", "mid", "near"):
        mask = np.rint(np.clip(images[name], 0.0, 1.0) * 255.0).astype(np.uint8)
        if not cv2.imwrite(str(mask_dir / f"{name}.png"), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
            raise OSError(f"failed to write {name} mask")
    metadata["asset_checksums"] = {
        path.relative_to(output).as_posix(): _file_hash(path)
        for path in sorted(output.rglob("*.png"))
    }
    metadata_path = output / "generator-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def _scene(width: int, height: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    horizon = np.clip(yy / height, 0.0, 1.0)
    scene = np.stack(
        (
            0.18 + 0.35 * (1.0 - horizon),
            0.28 + 0.36 * (1.0 - horizon),
            0.34 + 0.38 * (1.0 - horizon),
        ),
        axis=-1,
    )
    glow = np.exp(-(((xx - width * 0.72) / 210.0) ** 2 + ((yy - height * 0.22) / 150.0) ** 2))
    scene += glow[..., None] * np.array([0.22, 0.15, 0.05], dtype=np.float32)

    mountains = np.array(
        [
            [0, 330],
            *[
                [x, int(302 - 42 * np.sin(x / 105.0) - 18 * np.sin(x / 37.0))]
                for x in range(0, width + 1, 24)
            ],
            [width, 430],
            [0, 430],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(scene, [mountains], (0.18, 0.32, 0.28), lineType=cv2.LINE_AA)
    cv2.polylines(scene, [mountains[:-2]], False, (0.62, 0.76, 0.67), 2, cv2.LINE_AA)

    building = np.array([[116, 240], [888, 206], [930, 610], [82, 632]], dtype=np.int32)
    cv2.fillConvexPoly(scene, building, (0.25, 0.29, 0.34), lineType=cv2.LINE_AA)
    cv2.polylines(scene, [building], True, (0.82, 0.72, 0.49), 5, cv2.LINE_AA)
    for row in range(5):
        for column in range(10):
            x0 = 145 + column * 70 + (row % 2) * 7
            y0 = 270 + row * 58
            cv2.rectangle(scene, (x0, y0), (x0 + 42, y0 + 28), (0.62, 0.53, 0.30), -1)
            cv2.rectangle(scene, (x0 + 4, y0 + 4), (x0 + 38, y0 + 24), (0.17, 0.25, 0.33), -1)
    cv2.putText(scene, "MID / MARKET", (332, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0.85, 0.82, 0.70), 3, cv2.LINE_AA)

    ground = np.array([[0, 500], [width, 472], [width, height], [0, height]], dtype=np.int32)
    cv2.fillConvexPoly(scene, ground, (0.20, 0.18, 0.16), lineType=cv2.LINE_AA)
    vanishing = (width // 2, 492)
    for x in range(-300, width + 350, 130):
        cv2.line(scene, vanishing, (x, height), (0.52, 0.48, 0.39), 2, cv2.LINE_AA)
    for y in range(560, height, 60):
        cv2.line(scene, (0, y), (width, y - 18), (0.34, 0.31, 0.28), 2, cv2.LINE_AA)

    sign = np.array([[660, 490], [930, 500], [900, 690], [625, 675]], dtype=np.int32)
    cv2.fillConvexPoly(scene, sign, (0.10, 0.24, 0.31), lineType=cv2.LINE_AA)
    cv2.polylines(scene, [sign], True, (0.91, 0.62, 0.25), 6, cv2.LINE_AA)
    cv2.putText(scene, "NEAR", (694, 588), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0.94, 0.90, 0.77), 3, cv2.LINE_AA)
    cv2.putText(scene, "PSF", (731, 640), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0.64, 0.88, 0.88), 2, cv2.LINE_AA)

    rng = np.random.default_rng(NOISE_SEED)
    for x, y, radius in rng.integers([30, 80, 3], [width - 30, height - 40, 14], size=(85, 3)):
        color = tuple(float(value) for value in rng.uniform(0.25, 0.75, size=3))
        cv2.circle(scene, (int(x), int(y)), int(radius), color, 1, cv2.LINE_AA)

    facade_weight = np.exp(-((yy - height * 0.535) / (height * 0.22)) ** 2)
    facade_texture = 0.10 * (
        np.sin(xx * 0.38)
        + np.sin(yy * 0.29)
        + 0.5 * np.sin((xx + yy) * 0.21)
    )
    scene += facade_texture[..., None] * facade_weight[..., None]
    return np.clip(scene, 0.0, 1.0).astype(np.float32)


def _soft_masks(width: int, height: int) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    far = 1.0 / (1.0 + np.exp((yy - 320.0) / 34.0))
    mid = np.exp(-((yy - 410.0) / 175.0) ** 2)
    near = 1.0 / (1.0 + np.exp((475.0 - yy) / 32.0))
    near += 0.8 * (((xx - 785.0) / 210.0) ** 2 + ((yy - 590.0) / 145.0) ** 2 < 1.0)
    weights = np.stack((far, mid, near), axis=-1)
    weights = cv2.GaussianBlur(weights, (0, 0), 5.0)
    weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1e-8)
    return {
        "far": weights[..., 0].astype(np.float32),
        "mid": weights[..., 1].astype(np.float32),
        "near": weights[..., 2].astype(np.float32),
    }


def _kernels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    far = _motion_kernel(9, 0.0, 1.2)
    mid = _motion_kernel(17, 38.0, 0.7)
    near = _curved_kernel()
    return far, mid, near


def _motion_kernel(length: int, angle: float, defocus: float) -> np.ndarray:
    size = 25
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    radius = (length - 1) / 2.0
    radians = np.deg2rad(angle)
    start = (round(center - radius * np.cos(radians)), round(center - radius * np.sin(radians)))
    end = (round(center + radius * np.cos(radians)), round(center + radius * np.sin(radians)))
    cv2.line(kernel, start, end, 1.0, 1, cv2.LINE_AA)
    return _normalize_kernel(cv2.GaussianBlur(kernel, (0, 0), defocus))


def _curved_kernel() -> np.ndarray:
    size = 29
    kernel = np.zeros((size, size), dtype=np.float32)
    points = np.array(
        [[4, 22], [8, 19], [12, 16], [16, 12], [20, 9], [24, 7]], dtype=np.int32
    )
    cv2.polylines(kernel, [points], False, 1.0, 2, cv2.LINE_AA)
    return _normalize_kernel(cv2.GaussianBlur(kernel, (0, 0), 0.8))


def _normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    total = float(kernel.sum())
    if total <= 0.0:
        raise ValueError("PSF kernel must have non-zero mass")
    return (kernel / total).astype(np.float32)


def _blur_by_region(
    reference: np.ndarray, masks: dict[str, np.ndarray], kernels: tuple[np.ndarray, ...]
) -> np.ndarray:
    result = np.zeros_like(reference)
    for mask, kernel in zip(masks.values(), kernels, strict=True):
        convolved = cv2.filter2D(reference, -1, kernel, borderType=cv2.BORDER_REFLECT101)
        result += convolved * mask[..., None]
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _noisy_auxiliary(reference: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    translations = ((1.0, 0.0), (3.0, -2.0), (6.0, -3.0))
    shifted = np.zeros_like(reference)
    for mask, (dx, dy) in zip(masks.values(), translations, strict=True):
        transform = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        translated = cv2.warpAffine(
            reference,
            transform,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        shifted += translated * mask[..., None]
    exposed = np.clip(shifted * 0.72 + 0.018, 0.0, 1.0)
    rng = np.random.default_rng(NOISE_SEED)
    shot = rng.poisson(exposed * 255.0).astype(np.float32) / 255.0
    read = rng.normal(0.0, 0.012, exposed.shape).astype(np.float32)
    return np.clip(shot + read, 0.0, 1.0).astype(np.float32)


def _kernel_metadata(name: str, kernel: np.ndarray) -> dict[str, object]:
    return {
        "name": name,
        "shape": list(kernel.shape),
        "normalized_sum": float(kernel.sum()),
        "sha256": sha256(kernel.tobytes()).hexdigest(),
    }


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-dimension", type=int, default=512)
    args = parser.parse_args()
    print(write_dataset(args.output_dir, args.max_dimension))


if __name__ == "__main__":
    main()
