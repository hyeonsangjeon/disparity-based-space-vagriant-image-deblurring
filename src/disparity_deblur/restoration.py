from typing import Sequence

import cv2
import numpy as np

from .backends import deconvolve_channels


def restore_regions(
    blurred: np.ndarray,
    labels: np.ndarray,
    kernels: Sequence[np.ndarray],
    *,
    data_weight: float = 200.0,
    beta_max: float = 32.0,
    backend: str = "numpy",
    feather_sigma: float = 5.0,
    boundary_padding: int | None = None,
) -> np.ndarray:
    """Restore each unique regional PSF once, then feather regions together."""

    region_count = int(labels.max()) + 1
    if len(kernels) != region_count:
        raise ValueError(f"received {len(kernels)} kernels for {region_count} regions")
    restored_by_kernel: dict[tuple[tuple[int, ...], str, bytes], np.ndarray] = {}
    restored = []
    for kernel in kernels:
        contiguous = np.ascontiguousarray(kernel)
        key = (contiguous.shape, contiguous.dtype.str, contiguous.tobytes())
        if key not in restored_by_kernel:
            restored_by_kernel[key] = tv_l1_deconvolve(
                blurred,
                contiguous,
                data_weight=data_weight,
                beta_max=beta_max,
                backend=backend,
                boundary_padding=boundary_padding,
            )
        restored.append(restored_by_kernel[key])
    return normalized_region_merge(restored, labels, feather_sigma=feather_sigma)


def tv_l1_deconvolve(
    blurred: np.ndarray,
    kernel: np.ndarray,
    *,
    data_weight: float,
    beta_max: float,
    backend: str = "numpy",
    boundary_padding: int | None = None,
) -> np.ndarray:
    """Restore a 2D or HxWxC image with half-quadratic TV/L1 FFT updates."""

    if blurred.ndim == 2:
        channels = blurred[..., None]
        squeeze = True
    elif blurred.ndim == 3:
        channels = blurred
        squeeze = False
    else:
        raise ValueError(f"unsupported image shape {blurred.shape}")
    if min(channels.shape[:2]) < 2:
        raise ValueError("deconvolution requires image height and width of at least 2")

    padding = (
        max(kernel.shape) - 1 if boundary_padding is None else boundary_padding
    )
    if padding < 0:
        raise ValueError("boundary padding cannot be negative")
    if padding:
        channels = np.pad(
            channels,
            ((padding, padding), (padding, padding), (0, 0)),
            mode="reflect",
        )

    if backend == "numpy":
        output = _tv_l1_deconvolve_numpy(
            channels,
            kernel,
            data_weight=data_weight,
            beta_max=beta_max,
        )
    else:
        output = deconvolve_channels(
            channels,
            kernel,
            data_weight=data_weight,
            beta_max=beta_max,
            backend=backend,
        )
    if output.shape != channels.shape or not np.isfinite(output).all():
        raise RuntimeError(
            f"{backend} backend returned invalid output shape or non-finite values"
        )
    if padding:
        output = output[padding:-padding, padding:-padding]
    return output[..., 0] if squeeze else output


def _tv_l1_deconvolve_numpy(
    channels: np.ndarray,
    kernel: np.ndarray,
    *,
    data_weight: float,
    beta_max: float,
) -> np.ndarray:
    """Execute the canonical NumPy FFT implementation on an HxWxC array."""

    height, width = channels.shape[:2]
    kernel_otf = psf_to_otf(kernel, (height, width))
    difference_x = np.zeros((height, width), dtype=np.float64)
    difference_y = np.zeros((height, width), dtype=np.float64)
    difference_x[0, 0], difference_x[0, 1] = -1.0, 1.0
    difference_y[0, 0], difference_y[1, 0] = -1.0, 1.0
    dx_otf = np.fft.fft2(difference_x)
    dy_otf = np.fft.fft2(difference_y)
    derivative_power = np.abs(dx_otf) ** 2 + np.abs(dy_otf) ** 2
    kernel_power = np.abs(kernel_otf) ** 2

    output = np.empty_like(channels, dtype=np.float64)
    for channel in range(channels.shape[2]):
        observation = channels[..., channel].astype(np.float64)
        observation_fft = np.fft.fft2(observation)
        image = observation.copy()
        beta = 1.0
        while beta <= beta_max + 1e-12:
            gradient_x = np.roll(image, -1, axis=1) - image
            gradient_y = np.roll(image, -1, axis=0) - image
            auxiliary_x = _soft_threshold(gradient_x, 1.0 / beta)
            auxiliary_y = _soft_threshold(gradient_y, 1.0 / beta)
            numerator = (
                data_weight * np.conj(kernel_otf) * observation_fft
                + beta
                * (
                    np.conj(dx_otf) * np.fft.fft2(auxiliary_x)
                    + np.conj(dy_otf) * np.fft.fft2(auxiliary_y)
                )
            )
            denominator = (
                data_weight * kernel_power + beta * derivative_power + 1e-8
            )
            image = np.fft.ifft2(numerator / denominator).real
            beta *= 2.0
        output[..., channel] = np.clip(image, 0.0, 1.0)
    return output


def normalized_region_merge(
    restored: list[np.ndarray],
    labels: np.ndarray,
    *,
    feather_sigma: float,
) -> np.ndarray:
    """Blend full-frame regional restorations with normalized feather weights."""

    if not restored:
        raise ValueError("at least one restored image is required")
    shape = restored[0].shape
    if any(image.shape != shape for image in restored):
        raise ValueError("restored image shapes differ")
    if labels.shape != shape[:2]:
        raise ValueError("region labels must match restored image height and width")
    region_count = int(labels.max()) + 1
    if len(restored) != region_count:
        raise ValueError(
            f"received {len(restored)} restored images for {region_count} regions"
        )

    numerator = np.zeros(shape, dtype=np.float64)
    denominator = np.zeros(labels.shape, dtype=np.float64)
    for region, image in enumerate(restored):
        mask = (labels == region).astype(np.float64)
        if feather_sigma > 0:
            mask = cv2.GaussianBlur(
                mask,
                (0, 0),
                feather_sigma,
                borderType=cv2.BORDER_REFLECT,
            )
        denominator += mask
        if image.ndim == 3:
            numerator += image * mask[..., None]
        else:
            numerator += image * mask
    denominator = np.maximum(denominator, 1e-12)
    return (
        numerator / denominator[..., None]
        if numerator.ndim == 3
        else numerator / denominator
    )


def psf_to_otf(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Embed a centered PSF in a rectangular FFT domain."""

    if kernel.shape[0] > shape[0] or kernel.shape[1] > shape[1]:
        raise ValueError(f"kernel {kernel.shape} is larger than target {shape}")
    padded = np.zeros(shape, dtype=np.float64)
    padded[: kernel.shape[0], : kernel.shape[1]] = kernel
    padded = np.roll(padded, -(kernel.shape[0] // 2), axis=0)
    padded = np.roll(padded, -(kernel.shape[1] // 2), axis=1)
    return np.fft.fft2(padded)


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Apply element-wise soft thresholding."""

    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)
