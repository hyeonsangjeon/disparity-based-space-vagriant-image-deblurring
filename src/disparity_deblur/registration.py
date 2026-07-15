import cv2
import numpy as np

from .models import RegistrationResult


def register_noisy_to_blur(
    blurred: np.ndarray,
    noisy: np.ndarray,
    *,
    max_corners: int = 2500,
    ransac_threshold: float = 2.5,
) -> RegistrationResult:
    """Register the noisy exposure to the blurred frame with Harris, LK, and RANSAC."""

    if blurred.shape != noisy.shape:
        raise ValueError(f"image shapes differ: {blurred.shape} vs {noisy.shape}")

    blur_gray = _feature_gray(blurred)
    noisy_gray = _feature_gray(noisy)
    points = cv2.goodFeaturesToTrack(
        blur_gray,
        maxCorners=max_corners,
        qualityLevel=0.005,
        minDistance=7,
        blockSize=5,
        useHarrisDetector=True,
        k=0.04,
    )
    if points is None or len(points) < 8:
        raise RuntimeError("Harris detector produced fewer than eight corners")

    lk = {
        "winSize": (31, 31),
        "maxLevel": 4,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            40,
            0.001,
        ),
    }
    tracked, status_forward, errors = cv2.calcOpticalFlowPyrLK(
        blur_gray, noisy_gray, points, None, **lk
    )
    if tracked is None or status_forward is None or errors is None:
        raise RuntimeError("forward LK optical flow failed")
    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        noisy_gray, blur_gray, tracked, None, **lk
    )
    if backward is None or status_backward is None:
        raise RuntimeError("backward LK optical flow failed")

    forward_backward = np.linalg.norm(points - backward, axis=2).ravel()
    valid_status = status_forward.ravel().astype(bool) & status_backward.ravel().astype(
        bool
    )
    if not valid_status.any():
        raise RuntimeError("LK optical flow produced no bidirectional matches")
    error_limit = np.percentile(errors.ravel()[valid_status], 90)
    valid = (
        valid_status
        & np.isfinite(forward_backward)
        & (forward_backward < 1.5)
        & (errors.ravel() <= error_limit)
    )
    blur_points = points.reshape(-1, 2)[valid]
    noisy_points = tracked.reshape(-1, 2)[valid]
    if len(blur_points) < 8:
        raise RuntimeError("forward-backward LK filtering left fewer than eight matches")

    homography, inlier_mask = cv2.findHomography(
        noisy_points,
        blur_points,
        cv2.RANSAC,
        ransac_threshold,
    )
    if homography is None or inlier_mask is None:
        raise RuntimeError("RANSAC homography estimation failed")
    inliers = inlier_mask.ravel().astype(bool)
    if inliers.sum() < 8:
        raise RuntimeError("RANSAC retained fewer than eight inlier matches")

    height, width = blurred.shape[:2]
    noisy_u8 = np.rint(np.clip(noisy, 0.0, 1.0) * 255.0).astype(np.uint8)
    registered = cv2.warpPerspective(
        noisy_u8,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    ).astype(np.float32) / 255.0
    valid_mask = cv2.warpPerspective(
        np.ones((height, width), dtype=np.uint8),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)

    blur_inliers = blur_points[inliers]
    noisy_inliers = noisy_points[inliers]
    return RegistrationResult(
        registered_noisy=registered,
        valid_mask=valid_mask,
        blur_points=blur_inliers,
        disparities=noisy_inliers - blur_inliers,
        homography=homography,
        detected_count=len(points),
    )


def identity_registration(blurred: np.ndarray, noisy: np.ndarray) -> RegistrationResult:
    """Return a shape-preserving identity registration for pre-aligned inputs."""

    if blurred.shape != noisy.shape:
        raise ValueError(f"image shapes differ: {blurred.shape} vs {noisy.shape}")
    height, width = blurred.shape[:2]
    return RegistrationResult(
        registered_noisy=noisy.copy(),
        valid_mask=np.ones((height, width), dtype=bool),
        blur_points=np.empty((0, 2), dtype=np.float32),
        disparities=np.empty((0, 2), dtype=np.float32),
        homography=np.eye(3, dtype=np.float64),
        detected_count=0,
    )


def _feature_gray(image: np.ndarray) -> np.ndarray:
    """Build a CLAHE-enhanced grayscale image for feature tracking."""

    rgb = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
