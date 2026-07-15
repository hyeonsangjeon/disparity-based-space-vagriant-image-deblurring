from pathlib import Path

import cv2
import numpy as np

from .artifacts import ArtifactWriter
from .config import PipelineConfig
from .image_io import read_input_image
from .kernel_estimation import estimate_region_kernels
from .models import DeblurResult, InputSpec
from .postprocessing import guided_noisy_detail_fusion, luminance_unsharp
from .registration import identity_registration, register_noisy_to_blur
from .restoration import restore_regions
from .segmentation import disparity_segmentation


class DisparityDeblurPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def process(self, blurred: np.ndarray, noisy: np.ndarray) -> DeblurResult:
        _validate_images(blurred, noisy)
        config = self.config
        cv2.setRNGSeed(config.random_seed)

        registration = (
            identity_registration(blurred, noisy)
            if config.skip_registration
            else register_noisy_to_blur(
                blurred,
                noisy,
                max_corners=config.max_corners,
                ransac_threshold=config.ransac_threshold,
            )
        )
        segmentation = disparity_segmentation(
            blurred,
            registration.blur_points,
            registration.disparities,
            color_labels=config.color_labels,
            graph_cut_size=config.graph_cut_size,
            graph_cut_smoothness=config.graph_cut_smoothness,
            disparity_threshold=config.disparity_threshold,
            minimum_disparity_features=config.minimum_disparity_features,
            max_regions=config.max_regions,
            minimum_region_fraction=config.minimum_region_fraction,
        )
        kernels = tuple(
            estimate_region_kernels(
                blurred,
                registration.registered_noisy,
                segmentation.merged_labels,
                segmentation.region_disparities,
                kernel_size=config.kernel_size,
                patch_size=config.patch_size,
                tikhonov=config.tikhonov,
                kurtosis_min=config.kurtosis_min,
                kurtosis_max=config.kurtosis_max,
            )
        )
        deconvolved = restore_regions(
            blurred,
            segmentation.merged_labels,
            [estimate.kernel for estimate in kernels],
            data_weight=config.data_weight,
            beta_max=config.beta_max,
            feather_sigma=config.feather_sigma,
            boundary_padding=config.boundary_padding,
        )
        restored = luminance_unsharp(
            deconvolved,
            sigma=config.unsharp_sigma,
            amount=config.unsharp_amount,
            threshold=config.unsharp_threshold,
        )
        restored = guided_noisy_detail_fusion(
            restored,
            registration.registered_noisy,
            registration.valid_mask,
            denoise_strength=config.detail_denoise_strength,
            sigma=config.detail_fusion_sigma,
            amount=config.detail_fusion_amount,
            tolerance=config.detail_fusion_tolerance,
            threshold=config.detail_fusion_threshold,
        )
        if config.input_detail_blend:
            input_detail = luminance_unsharp(
                blurred,
                sigma=config.input_detail_sigma,
                amount=config.input_detail_amount,
                threshold=0.0,
            )
            restored = (
                (1.0 - config.input_detail_blend) * restored
                + config.input_detail_blend * input_detail
            )
        return DeblurResult(
            blurred=blurred,
            noisy=noisy,
            registration=registration,
            segmentation=segmentation,
            kernels=kernels,
            deconvolved=deconvolved,
            restored=restored,
        )


def run_pipeline(
    blurred_path: str | Path,
    noisy_path: str | Path,
    *,
    height: int | None = None,
    width: int | None = None,
    output_dir: str | Path,
    config: PipelineConfig | None = None,
) -> Path:
    blurred_source = Path(blurred_path)
    noisy_source = Path(noisy_path)
    blurred = read_input_image(blurred_source, height, width)
    noisy = read_input_image(noisy_source, height, width)
    actual_height, actual_width = blurred.shape[:2]
    inputs = InputSpec(
        blurred_path=blurred_source,
        noisy_path=noisy_source,
        height=actual_height,
        width=actual_width,
    )
    pipeline = DisparityDeblurPipeline(config)
    result = pipeline.process(blurred, noisy)
    return ArtifactWriter(output_dir).write(result, inputs, pipeline.config)


def _validate_images(blurred: np.ndarray, noisy: np.ndarray) -> None:
    if blurred.shape != noisy.shape:
        raise ValueError(f"image shapes differ: {blurred.shape} vs {noisy.shape}")
    if blurred.ndim != 3 or blurred.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB images, got {blurred.shape}")
    if not np.issubdtype(blurred.dtype, np.floating) or not np.issubdtype(
        noisy.dtype, np.floating
    ):
        raise TypeError("pipeline images must use floating-point RGB values")
    if not np.isfinite(blurred).all() or not np.isfinite(noisy).all():
        raise ValueError("pipeline images contain non-finite values")
