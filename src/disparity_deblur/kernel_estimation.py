import cv2
import numpy as np
from scipy import ndimage

from .models import KernelEstimate


def estimate_region_kernels(
    blurred: np.ndarray,
    registered_noisy: np.ndarray,
    labels: np.ndarray,
    region_disparities: np.ndarray,
    *,
    kernel_size: int = 33,
    patch_size: int = 256,
    tikhonov: float = 1e-3,
    kurtosis_min: float = 20.0,
    kurtosis_max: float = 300.0,
) -> list[KernelEstimate]:
    blur_gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    noisy_gray = cv2.cvtColor(registered_noisy, cv2.COLOR_RGB2GRAY)
    noisy_gray = cv2.GaussianBlur(noisy_gray, (0, 0), 0.8)
    region_count = int(labels.max()) + 1

    raw: list[KernelEstimate] = []
    for region in range(region_count):
        kernel, bounds = estimate_tikhonov_kernel(
            blur_gray,
            noisy_gray,
            labels == region,
            kernel_size=kernel_size,
            patch_size=patch_size,
            regularization=tikhonov,
        )
        raw.append(
            KernelEstimate(
                kernel=kernel,
                kurtosis=kernel_kurtosis(kernel),
                patch_bounds=bounds,
            )
        )

    reliable = [
        index
        for index, estimate in enumerate(raw)
        if kurtosis_min <= estimate.kurtosis <= kurtosis_max
    ]
    if not reliable:
        global_kernel, bounds = estimate_tikhonov_kernel(
            blur_gray,
            noisy_gray,
            np.ones(labels.shape, dtype=bool),
            kernel_size=kernel_size,
            patch_size=patch_size,
            regularization=tikhonov,
        )
        global_estimate = KernelEstimate(
            kernel=global_kernel,
            kurtosis=kernel_kurtosis(global_kernel),
            patch_bounds=bounds,
        )
        return [
            KernelEstimate(
                kernel=global_estimate.kernel.copy(),
                kurtosis=global_estimate.kurtosis,
                patch_bounds=estimate.patch_bounds,
                replaced_from=-1,
            )
            for estimate in raw
        ]

    result = list(raw)
    for index, estimate in enumerate(raw):
        if index in reliable:
            continue
        replacement = min(
            reliable,
            key=lambda candidate: np.linalg.norm(
                region_disparities[index] - region_disparities[candidate]
            ),
        )
        source = raw[replacement]
        result[index] = KernelEstimate(
            kernel=source.kernel.copy(),
            kurtosis=source.kurtosis,
            patch_bounds=estimate.patch_bounds,
            replaced_from=replacement,
        )
    return result


def estimate_tikhonov_kernel(
    blurred_gray: np.ndarray,
    noisy_gray: np.ndarray,
    mask: np.ndarray,
    *,
    kernel_size: int,
    patch_size: int,
    regularization: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if kernel_size % 2 != 1:
        raise ValueError("kernel size must be odd")
    if blurred_gray.shape != noisy_gray.shape or blurred_gray.shape != mask.shape:
        raise ValueError("blurred, noisy, and mask shapes must match")
    blurred_gray = np.asarray(blurred_gray, dtype=np.float32)
    noisy_gray = np.asarray(noisy_gray, dtype=np.float32)

    edge_x = cv2.Sobel(noisy_gray, cv2.CV_32F, 1, 0, ksize=3)
    edge_y = cv2.Sobel(noisy_gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = np.hypot(edge_x, edge_y)
    bounds = _edge_rich_patch(mask, edge_energy, patch_size)
    y0, y1, x0, x1 = bounds
    blur_patch = blurred_gray[y0:y1, x0:x1].astype(np.float64)
    noisy_patch = noisy_gray[y0:y1, x0:x1].astype(np.float64)
    mask_patch = mask[y0:y1, x0:x1]

    distance = ndimage.distance_transform_edt(mask_patch)
    weights = np.clip(distance / max(kernel_size / 4.0, 1.0), 0.0, 1.0)
    if weights.max() == 0:
        weights = mask_patch.astype(np.float64)
    if weights.sum() < kernel_size * kernel_size:
        weights = np.ones_like(weights)

    denominator = float(np.sum(weights * noisy_patch * noisy_patch))
    exposure_scale = (
        float(np.sum(weights * noisy_patch * blur_patch)) / max(denominator, 1e-12)
    )
    noisy_patch *= exposure_scale

    noisy_x = cv2.Sobel(noisy_patch, cv2.CV_64F, 1, 0, ksize=3) * weights
    noisy_y = cv2.Sobel(noisy_patch, cv2.CV_64F, 0, 1, ksize=3) * weights
    blur_x = cv2.Sobel(blur_patch, cv2.CV_64F, 1, 0, ksize=3) * weights
    blur_y = cv2.Sobel(blur_patch, cv2.CV_64F, 0, 1, ksize=3) * weights

    noisy_x_fft = np.fft.fft2(noisy_x)
    noisy_y_fft = np.fft.fft2(noisy_y)
    blur_x_fft = np.fft.fft2(blur_x)
    blur_y_fft = np.fft.fft2(blur_y)
    numerator = np.conj(noisy_x_fft) * blur_x_fft + np.conj(
        noisy_y_fft
    ) * blur_y_fft
    spectral_power = np.abs(noisy_x_fft) ** 2 + np.abs(noisy_y_fft) ** 2
    solution = np.fft.ifft2(
        numerator / (spectral_power + regularization * spectral_power.max() + 1e-12)
    ).real
    centered = np.fft.fftshift(solution)
    center_y, center_x = np.array(centered.shape) // 2
    radius = kernel_size // 2
    kernel = centered[
        center_y - radius : center_y + radius + 1,
        center_x - radius : center_x + radius + 1,
    ]
    kernel = np.maximum(kernel, 0.0)
    if kernel.max() > 0:
        kernel[kernel < kernel.max() * 0.01] = 0.0
    if kernel.sum() <= 1e-12:
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
        kernel[radius, radius] = 1.0
    else:
        kernel /= kernel.sum()
        kernel = _center_kernel(kernel)
    return kernel, bounds


def kernel_kurtosis(kernel: np.ndarray) -> float:
    values = kernel.ravel().astype(np.float64)
    centered = values - values.mean()
    variance = np.mean(centered * centered)
    if variance <= 1e-20:
        return float("inf")
    return float(np.mean(centered**4) / (variance * variance))


def _edge_rich_patch(
    mask: np.ndarray, edge_energy: np.ndarray, patch_size: int
) -> tuple[int, int, int, int]:
    height, width = mask.shape
    size = min(patch_size, height, width)
    weighted_edge = edge_energy * mask.astype(np.float32)
    edge_sum = cv2.boxFilter(
        weighted_edge,
        cv2.CV_32F,
        (size, size),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    coverage = cv2.boxFilter(
        mask.astype(np.float32),
        cv2.CV_32F,
        (size, size),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    score = edge_sum * np.sqrt(np.clip(coverage / (size * size), 0.0, 1.0))
    score[coverage < size * size * 0.05] = -1.0
    center_y, center_x = np.unravel_index(np.argmax(score), score.shape)
    y0 = int(np.clip(center_y - size // 2, 0, height - size))
    x0 = int(np.clip(center_x - size // 2, 0, width - size))
    return y0, y0 + size, x0, x0 + size


def _center_kernel(kernel: np.ndarray) -> np.ndarray:
    yy, xx = np.indices(kernel.shape)
    center_of_mass = np.array(
        [
            np.sum(yy * kernel) / kernel.sum(),
            np.sum(xx * kernel) / kernel.sum(),
        ]
    )
    target = (np.array(kernel.shape) - 1) / 2.0
    shifted = ndimage.shift(
        kernel,
        target - center_of_mass,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    shifted = np.maximum(shifted, 0.0)
    return shifted / max(shifted.sum(), 1e-12)
