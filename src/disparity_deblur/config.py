from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    random_seed: int = 0
    max_corners: int = 2500
    ransac_threshold: float = 2.5
    kernel_size: int = 33
    color_labels: int = 12
    graph_cut_size: int = 384
    graph_cut_smoothness: float = 0.35
    disparity_threshold: float = 1.5
    minimum_disparity_features: int = 1
    max_regions: int = 6
    minimum_region_fraction: float = 0.01
    patch_size: int = 256
    tikhonov: float = 1e-3
    kurtosis_min: float = 20.0
    kurtosis_max: float = 300.0
    data_weight: float = 200.0
    beta_max: float = 32.0
    feather_sigma: float = 5.0
    boundary_padding: int | None = None
    unsharp_sigma: float = 1.0
    unsharp_amount: float = 0.0
    unsharp_threshold: float = 0.0
    detail_denoise_strength: float = 5.0
    detail_fusion_sigma: float = 1.5
    detail_fusion_amount: float = 0.0
    detail_fusion_tolerance: float = 0.08
    detail_fusion_threshold: float = 0.015
    skip_registration: bool = False

    def __post_init__(self) -> None:
        if self.max_corners < 8:
            raise ValueError("max_corners must be at least 8")
        if self.ransac_threshold <= 0:
            raise ValueError("ransac_threshold must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if self.color_labels < 2:
            raise ValueError("color_labels must be at least 2")
        if self.graph_cut_size < 16:
            raise ValueError("graph_cut_size must be at least 16")
        if self.graph_cut_smoothness < 0:
            raise ValueError("graph_cut_smoothness cannot be negative")
        if self.disparity_threshold < 0:
            raise ValueError("disparity_threshold cannot be negative")
        if self.minimum_disparity_features < 1:
            raise ValueError("minimum_disparity_features must be at least 1")
        if self.max_regions < 1:
            raise ValueError("max_regions must be at least 1")
        if not 0 < self.minimum_region_fraction <= 1:
            raise ValueError("minimum_region_fraction must be in (0, 1]")
        if self.patch_size < self.kernel_size:
            raise ValueError("patch_size cannot be smaller than kernel_size")
        if self.tikhonov <= 0:
            raise ValueError("tikhonov must be positive")
        if self.kurtosis_min < 0 or self.kurtosis_max <= self.kurtosis_min:
            raise ValueError("kurtosis bounds are invalid")
        if self.data_weight <= 0:
            raise ValueError("data_weight must be positive")
        if self.beta_max < 1:
            raise ValueError("beta_max must be at least 1")
        if self.feather_sigma < 0:
            raise ValueError("feather_sigma cannot be negative")
        if self.boundary_padding is not None and self.boundary_padding < 0:
            raise ValueError("boundary_padding cannot be negative")
        if self.unsharp_sigma <= 0 or self.unsharp_amount < 0:
            raise ValueError("unsharp parameters are invalid")
        if self.unsharp_threshold < 0:
            raise ValueError("unsharp_threshold cannot be negative")
        if self.detail_denoise_strength < 0:
            raise ValueError("detail_denoise_strength cannot be negative")
        if self.detail_fusion_sigma <= 0 or self.detail_fusion_amount < 0:
            raise ValueError("detail fusion parameters are invalid")
        if self.detail_fusion_tolerance <= 0 or self.detail_fusion_threshold < 0:
            raise ValueError("detail fusion confidence parameters are invalid")

    @property
    def effective_boundary_padding(self) -> int:
        return (
            self.kernel_size - 1
            if self.boundary_padding is None
            else self.boundary_padding
        )
