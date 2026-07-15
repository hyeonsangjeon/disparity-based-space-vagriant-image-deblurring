import argparse

from .config import PipelineConfig
from .pipeline import run_pipeline


def main() -> None:
    """Parse CLI arguments and run the aspect-ratio-independent pipeline."""

    parser = argparse.ArgumentParser(
        description="Restore RAW or standard RGB images using disparity-based regional deblurring.",
        epilog=(
            "Use standard RGB inputs, or headerless RGB24 RAW inputs with "
            "--height and --width."
        ),
    )
    parser.add_argument("--blurred", required=True)
    parser.add_argument("--noisy", required=True)
    parser.add_argument("--height", type=int, help="required for headerless RAW input")
    parser.add_argument("--width", type=int, help="required for headerless RAW input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-corners", type=int, default=2500)
    parser.add_argument("--ransac-threshold", type=float, default=2.5)
    parser.add_argument("--kernel-size", type=int, default=33)
    parser.add_argument("--color-labels", type=int, default=12)
    parser.add_argument("--graph-cut-size", type=int, default=384)
    parser.add_argument("--graph-cut-smoothness", type=float, default=0.35)
    parser.add_argument("--disparity-threshold", type=float, default=1.5)
    parser.add_argument("--minimum-disparity-features", type=int, default=1)
    parser.add_argument("--max-regions", type=int, default=6)
    parser.add_argument("--minimum-region-fraction", type=float, default=0.01)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--tikhonov", type=float, default=1e-3)
    parser.add_argument("--kurtosis-min", type=float, default=20.0)
    parser.add_argument("--kurtosis-max", type=float, default=300.0)
    parser.add_argument("--data-weight", type=float, default=200.0)
    parser.add_argument("--beta-max", type=float, default=32.0)
    parser.add_argument(
        "--backend",
        choices=("numpy", "native-cpp", "cuda"),
        default="numpy",
        help="explicit deconvolution backend; optional backends never silently fall back",
    )
    parser.add_argument("--feather-sigma", type=float, default=5.0)
    parser.add_argument("--boundary-padding", type=int)
    parser.add_argument("--unsharp-sigma", type=float, default=1.0)
    parser.add_argument("--unsharp-amount", type=float, default=0.0)
    parser.add_argument("--unsharp-threshold", type=float, default=0.0)
    parser.add_argument("--detail-denoise-strength", type=float, default=5.0)
    parser.add_argument("--detail-fusion-sigma", type=float, default=1.5)
    parser.add_argument("--detail-fusion-amount", type=float, default=0.0)
    parser.add_argument("--detail-fusion-tolerance", type=float, default=0.08)
    parser.add_argument("--detail-fusion-threshold", type=float, default=0.015)
    parser.add_argument("--noisy-structure-blend", type=float, default=0.0)
    parser.add_argument("--noisy-structure-sigma", type=float, default=3.0)
    parser.add_argument("--noisy-structure-tolerance", type=float, default=0.12)
    parser.add_argument("--skip-registration", action="store_true")
    args = parser.parse_args()

    config = PipelineConfig(
        random_seed=args.random_seed,
        max_corners=args.max_corners,
        ransac_threshold=args.ransac_threshold,
        kernel_size=args.kernel_size,
        color_labels=args.color_labels,
        graph_cut_size=args.graph_cut_size,
        graph_cut_smoothness=args.graph_cut_smoothness,
        disparity_threshold=args.disparity_threshold,
        minimum_disparity_features=args.minimum_disparity_features,
        max_regions=args.max_regions,
        minimum_region_fraction=args.minimum_region_fraction,
        patch_size=args.patch_size,
        tikhonov=args.tikhonov,
        kurtosis_min=args.kurtosis_min,
        kurtosis_max=args.kurtosis_max,
        data_weight=args.data_weight,
        beta_max=args.beta_max,
        backend=args.backend,
        feather_sigma=args.feather_sigma,
        boundary_padding=args.boundary_padding,
        unsharp_sigma=args.unsharp_sigma,
        unsharp_amount=args.unsharp_amount,
        unsharp_threshold=args.unsharp_threshold,
        detail_denoise_strength=args.detail_denoise_strength,
        detail_fusion_sigma=args.detail_fusion_sigma,
        detail_fusion_amount=args.detail_fusion_amount,
        detail_fusion_tolerance=args.detail_fusion_tolerance,
        detail_fusion_threshold=args.detail_fusion_threshold,
        noisy_structure_blend=args.noisy_structure_blend,
        noisy_structure_sigma=args.noisy_structure_sigma,
        noisy_structure_tolerance=args.noisy_structure_tolerance,
        skip_registration=args.skip_registration,
    )
    metadata = run_pipeline(
        args.blurred,
        args.noisy,
        height=args.height,
        width=args.width,
        output_dir=args.output_dir,
        config=config,
    )
    print(metadata)


if __name__ == "__main__":
    main()
