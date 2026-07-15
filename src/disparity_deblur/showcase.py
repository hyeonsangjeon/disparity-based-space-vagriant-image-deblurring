"""Project-owned, reproducible public showcase image generator."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .benchmark import deterministic_noise
from .image_io import write_png


def generate_procedural_showcase(output_dir: str | Path, *, size: int = 256) -> None:
    """Generate a depth-layered reference, blur, and noisy auxiliary with fixed seeds."""
    if size < 128:
        raise ValueError("showcase size must be at least 128")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference, blurred = _render_scene(size)
    noisy_base = _translate(reference, 3, -2)
    noisy = deterministic_noise(noisy_base, seed=20260715, sigma=0.025)
    write_png(output / "reference.png", reference)
    write_png(output / "blurred.png", blurred)
    write_png(output / "noisy.png", noisy)


def _render_scene(size: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    reference = np.empty((size, size, 3), dtype=np.float32)
    reference[..., 0] = 0.08 + 0.32 * xx / (size - 1)
    reference[..., 1] = 0.14 + 0.40 * yy / (size - 1)
    reference[..., 2] = 0.24 + 0.36 * (1.0 - yy / (size - 1))
    for offset in range(0, size, max(12, size // 18)):
        cv2.line(reference, (offset, 0), (offset, size - 1), (0.72, 0.82, 0.9), 1)
        cv2.line(reference, (0, offset), (size - 1, offset), (0.72, 0.82, 0.9), 1)
    cv2.putText(
        reference,
        "DEPTH",
        (size // 4, size // 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        size / 300,
        (0.95, 0.95, 0.95),
        2,
        cv2.LINE_AA,
    )

    near_mask = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(
        near_mask,
        (size * 29 // 100, size * 63 // 100),
        (size * 17 // 100, size * 24 // 100),
        -12,
        0,
        360,
        1.0,
        -1,
    )
    far_mask = np.zeros((size, size), dtype=np.float32)
    cv2.rectangle(
        far_mask,
        (size * 60 // 100, size * 31 // 100),
        (size * 88 // 100, size * 62 // 100),
        1.0,
        -1,
    )
    near = np.zeros_like(reference)
    near[near_mask > 0] = (0.92, 0.16, 0.08)
    for offset in range(-size // 8, size // 8, max(6, size // 28)):
        cv2.line(
            near,
            (size * 12 // 100, size * 63 // 100 + offset),
            (size * 45 // 100, size * 63 // 100 + offset),
            (1.0, 0.78, 0.16),
            2,
        )
    far = np.zeros_like(reference)
    far[far_mask > 0] = (0.08, 0.67, 0.9)
    for offset in range(size * 63 // 100, size * 88 // 100, max(8, size // 22)):
        cv2.line(
            far,
            (offset, size * 34 // 100),
            (offset, size * 59 // 100),
            (0.82, 0.96, 1.0),
            2,
        )
    cv2.circle(far, (size * 74 // 100, size * 47 // 100), size * 7 // 100, (0.06, 0.22, 0.56), -1)
    reference = _composite(reference, near, near_mask)
    reference = _composite(reference, far, far_mask)

    background_blur = _filter(reference, _motion_kernel(11, 2))
    near_blur = _filter(near, _motion_kernel(21, 26))
    far_blur = _filter(far, _motion_kernel(15, 108))
    blurred = _composite(background_blur, near_blur, _filter_mask(near_mask, _motion_kernel(21, 26)))
    blurred = _composite(blurred, far_blur, _filter_mask(far_mask, _motion_kernel(15, 108)))
    return np.clip(reference, 0.0, 1.0), np.clip(blurred, 0.0, 1.0)


def _composite(background: np.ndarray, layer: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return background * (1.0 - mask[..., None]) + layer


def _motion_kernel(size: int, angle: float) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    start = tuple(np.rint(np.array([center, center]) - direction * (center - 1)).astype(int))
    end = tuple(np.rint(np.array([center, center]) + direction * (center - 1)).astype(int))
    cv2.line(kernel, start, end, 1.0, 1, cv2.LINE_AA)
    kernel = cv2.GaussianBlur(kernel, (3, 3), 0.45)
    return kernel / kernel.sum()


def _filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(image, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)


def _filter_mask(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.clip(_filter(mask, kernel), 0.0, 1.0)


def _translate(image: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
