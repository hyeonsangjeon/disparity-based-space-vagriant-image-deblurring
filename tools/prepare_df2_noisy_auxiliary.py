"""Create the explicitly labeled, deterministic DF2 noisy showcase auxiliary."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

from disparity_deblur.benchmark import deterministic_noise
from disparity_deblur.image_io import read_input_image, write_png


EXPECTED_SOURCE_SHA256 = (
    "67e9e68e41b679dcd120a5ceb47983098e955cbf44ab1912e78c5867807ef576"
)
NOISE_SEED = 20260715
NOISE_SIGMA = 0.02


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add reproducible Gaussian sensor noise to the preserved DF2 auxiliary."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "benchmarks/public-assets/01_df2_object_motion/noisy-original.png"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/public-assets/01_df2_object_motion/noisy.png"),
    )
    args = parser.parse_args()

    source_digest = _sha256(args.source)
    if source_digest != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "DF2 source checksum mismatch: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {source_digest}"
        )
    augmented = deterministic_noise(
        read_input_image(args.source),
        seed=NOISE_SEED,
        sigma=NOISE_SIGMA,
    )
    write_png(args.output, augmented)
    print(f"{args.output}: {_sha256(args.output)}")


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
