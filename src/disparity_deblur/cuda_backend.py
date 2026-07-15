"""Lazy CuPy/cuFFT implementation of the canonical TV/L1 solver."""

from __future__ import annotations

import numpy as np


class CudaUnavailableError(RuntimeError):
    """Raised when CuPy is installed but no usable CUDA device is available."""


def available() -> bool:
    """Return whether CuPy can access at least one CUDA device."""

    try:
        import cupy as cp
    except ImportError:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def deconvolve(
    channels: np.ndarray,
    kernel: np.ndarray,
    *,
    data_weight: float,
    beta_max: float,
) -> np.ndarray:
    """Run all TV/L1 iterations on a CUDA device with CuPy FFT operations."""

    import cupy as cp

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise CudaUnavailableError("CuPy found no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError as error:
        raise CudaUnavailableError("CUDA runtime initialization failed") from error

    output = _deconvolve_array_module(
        channels,
        kernel,
        data_weight=data_weight,
        beta_max=beta_max,
        array_module=cp,
    )
    return cp.asnumpy(output)


def _deconvolve_array_module(
    channels: np.ndarray,
    kernel: np.ndarray,
    *,
    data_weight: float,
    beta_max: float,
    array_module: object,
) -> object:
    """Execute canonical solver operations with NumPy or the CuPy array API."""

    xp = array_module
    source = xp.asarray(channels, dtype=xp.float64)
    kernel_array = xp.asarray(kernel, dtype=xp.float64)
    height, width = source.shape[:2]
    kernel_otf = _psf_to_otf(kernel_array, (height, width), xp)
    difference_x = xp.zeros((height, width), dtype=xp.float64)
    difference_y = xp.zeros((height, width), dtype=xp.float64)
    difference_x[0, 0], difference_x[0, 1] = -1.0, 1.0
    difference_y[0, 0], difference_y[1, 0] = -1.0, 1.0
    dx_otf = xp.fft.fft2(difference_x)
    dy_otf = xp.fft.fft2(difference_y)
    derivative_power = xp.abs(dx_otf) ** 2 + xp.abs(dy_otf) ** 2
    kernel_power = xp.abs(kernel_otf) ** 2

    output = xp.empty_like(source, dtype=xp.float64)
    for channel in range(source.shape[2]):
        observation = source[..., channel]
        observation_fft = xp.fft.fft2(observation)
        current = observation.copy()
        beta = 1.0
        while beta <= beta_max + 1e-12:
            gradient_x = xp.roll(current, -1, axis=1) - current
            gradient_y = xp.roll(current, -1, axis=0) - current
            auxiliary_x = _soft_threshold(gradient_x, 1.0 / beta, xp)
            auxiliary_y = _soft_threshold(gradient_y, 1.0 / beta, xp)
            numerator = (
                data_weight * xp.conj(kernel_otf) * observation_fft
                + beta
                * (
                    xp.conj(dx_otf) * xp.fft.fft2(auxiliary_x)
                    + xp.conj(dy_otf) * xp.fft.fft2(auxiliary_y)
                )
            )
            denominator = (
                data_weight * kernel_power + beta * derivative_power + 1e-8
            )
            current = xp.fft.ifft2(numerator / denominator).real
            beta *= 2.0
        output[..., channel] = xp.clip(current, 0.0, 1.0)
    return output


def _psf_to_otf(kernel: object, shape: tuple[int, int], cp: object) -> object:
    """Embed a centered PSF into a CuPy FFT domain."""

    if kernel.shape[0] > shape[0] or kernel.shape[1] > shape[1]:
        raise ValueError(f"kernel {kernel.shape} is larger than target {shape}")
    padded = cp.zeros(shape, dtype=cp.float64)
    padded[: kernel.shape[0], : kernel.shape[1]] = kernel
    padded = cp.roll(padded, -(kernel.shape[0] // 2), axis=0)
    padded = cp.roll(padded, -(kernel.shape[1] // 2), axis=1)
    return cp.fft.fft2(padded)


def _soft_threshold(values: object, threshold: float, cp: object) -> object:
    """Apply soft thresholding without transferring GPU arrays to the host."""

    return cp.sign(values) * cp.maximum(cp.abs(values) - threshold, 0.0)
