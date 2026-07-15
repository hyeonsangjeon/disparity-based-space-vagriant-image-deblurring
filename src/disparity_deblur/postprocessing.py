import cv2
import numpy as np


def luminance_unsharp(
    image: np.ndarray,
    *,
    sigma: float,
    amount: float,
    threshold: float,
) -> np.ndarray:
    if amount == 0:
        return image.copy()
    ycrcb = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    luminance = ycrcb[..., 0]
    detail = luminance - cv2.GaussianBlur(luminance, (0, 0), sigma)
    if threshold > 0:
        gain = np.clip((np.abs(detail) - threshold) / threshold, 0.0, 1.0)
        detail *= gain
    ycrcb[..., 0] = np.clip(luminance + amount * detail, 0.0, 1.0)
    return np.clip(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB), 0.0, 1.0)


def guided_noisy_detail_fusion(
    restored: np.ndarray,
    registered_noisy: np.ndarray,
    valid_mask: np.ndarray,
    *,
    denoise_strength: float,
    sigma: float,
    amount: float,
    tolerance: float,
    threshold: float,
) -> np.ndarray:
    if amount == 0:
        return restored.copy()
    _validate_guided_inputs(restored, registered_noisy, valid_mask)
    denoised = _denoise_registered_noisy(registered_noisy, denoise_strength)

    restored_ycrcb = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    restored_y = restored_ycrcb[..., 0]
    reference_y = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
    detail = reference_y - cv2.GaussianBlur(reference_y, (0, 0), sigma)

    low_restored = cv2.GaussianBlur(restored_y, (0, 0), 3.0)
    low_reference = cv2.GaussianBlur(reference_y, (0, 0), 3.0)
    confidence = np.exp(-((low_restored - low_reference) / tolerance) ** 2)
    confidence *= valid_mask
    if threshold > 0:
        edge_gain = np.clip((np.abs(detail) - threshold) / threshold, 0.0, 1.0)
        confidence *= edge_gain

    restored_ycrcb[..., 0] = np.clip(
        restored_y + amount * detail * confidence,
        0.0,
        1.0,
    )
    return np.clip(
        cv2.cvtColor(restored_ycrcb, cv2.COLOR_YCrCb2RGB),
        0.0,
        1.0,
    )


def guided_noisy_structure_fusion(
    restored: np.ndarray,
    registered_noisy: np.ndarray,
    valid_mask: np.ndarray,
    *,
    denoise_strength: float,
    sigma: float,
    amount: float,
    tolerance: float,
) -> np.ndarray:
    """Replace ringing-prone structure with tone-matched noisy-image structure."""
    if amount == 0:
        return restored.copy()
    if not 0 <= amount <= 1:
        raise ValueError("noisy structure fusion amount must be in [0, 1]")
    if sigma <= 0 or tolerance <= 0:
        raise ValueError("noisy structure fusion parameters must be positive")
    _validate_guided_inputs(restored, registered_noisy, valid_mask)

    denoised = _denoise_registered_noisy(registered_noisy, denoise_strength)
    restored_base = cv2.GaussianBlur(restored, (0, 0), sigma)
    noisy_base = cv2.GaussianBlur(denoised, (0, 0), sigma)
    tone_matched_noisy = np.clip(
        denoised + restored_base - noisy_base,
        0.0,
        1.0,
    )

    restored_luma = cv2.cvtColor(restored_base.astype(np.float32), cv2.COLOR_RGB2GRAY)
    noisy_luma = cv2.cvtColor(noisy_base.astype(np.float32), cv2.COLOR_RGB2GRAY)
    confidence = np.exp(-((restored_luma - noisy_luma) / tolerance) ** 2)
    valid = np.clip(valid_mask.astype(np.float32), 0.0, 1.0)
    confidence = cv2.GaussianBlur(confidence * valid, (0, 0), 1.0) * valid
    weight = (amount * np.clip(confidence, 0.0, 1.0))[..., None]
    return np.clip(
        restored * (1.0 - weight) + tone_matched_noisy * weight,
        0.0,
        1.0,
    )


def _denoise_registered_noisy(
    registered_noisy: np.ndarray,
    denoise_strength: float,
) -> np.ndarray:
    if denoise_strength < 0:
        raise ValueError("denoise strength cannot be negative")
    if denoise_strength == 0:
        return registered_noisy.astype(np.float32).copy()
    noisy_u8 = np.rint(np.clip(registered_noisy, 0.0, 1.0) * 255.0).astype(
        np.uint8
    )
    return (
        cv2.fastNlMeansDenoisingColored(
            noisy_u8,
            None,
            denoise_strength,
            denoise_strength,
            7,
            21,
        ).astype(np.float32)
        / 255.0
    )


def _validate_guided_inputs(
    restored: np.ndarray,
    registered_noisy: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    if restored.shape != registered_noisy.shape:
        raise ValueError("restored and registered noisy image shapes differ")
    if valid_mask.shape != restored.shape[:2]:
        raise ValueError("registration mask shape differs from image shape")
