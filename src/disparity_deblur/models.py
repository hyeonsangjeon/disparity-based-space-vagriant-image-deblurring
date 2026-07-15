from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RegistrationResult:
    """Noisy-to-blurred registration, valid support, and inlier disparities."""

    registered_noisy: np.ndarray
    valid_mask: np.ndarray
    blur_points: np.ndarray
    disparities: np.ndarray
    homography: np.ndarray
    detected_count: int


@dataclass(frozen=True)
class SegmentationResult:
    """Initial color regions and their disparity-guided merged partition."""

    initial_labels: np.ndarray
    merged_labels: np.ndarray
    region_disparities: np.ndarray
    region_feature_counts: np.ndarray


@dataclass(frozen=True)
class KernelEstimate:
    """One regional PSF estimate and the source patch used to estimate it."""

    kernel: np.ndarray
    kurtosis: float
    patch_bounds: tuple[int, int, int, int]
    replaced_from: int | None = None


@dataclass(frozen=True)
class DeblurResult:
    """Complete in-memory output of the disparity deblurring pipeline."""

    blurred: np.ndarray
    noisy: np.ndarray
    registration: RegistrationResult
    segmentation: SegmentationResult
    kernels: tuple[KernelEstimate, ...]
    deconvolved: np.ndarray
    restored: np.ndarray


@dataclass(frozen=True)
class InputSpec:
    """Source paths and independently recorded image height and width."""

    blurred_path: Path
    noisy_path: Path
    height: int
    width: int
