"""Create the minimal, derived CC BY 4.0 GoPro Flower showcase subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from disparity_deblur.benchmark import deterministic_noise
from disparity_deblur.image_io import read_input_image, write_png


SOURCE_COMMIT = "2d5a698560e658718f0520e48dfb15bd52c80118"
BLURRED_SOURCE = (
    "https://raw.githubusercontent.com/SeungjunNah/DeepDeblur_release/"
    f"{SOURCE_COMMIT}/images/Flower_blur1.png"
)
REFERENCE_SOURCE = (
    "https://raw.githubusercontent.com/SeungjunNah/DeepDeblur_release/"
    f"{SOURCE_COMMIT}/images/Flower_sharp1.png"
)
NOISE_SEED = 20260715
NOISE_SIGMA = 0.018
MAX_DIMENSION = 384


def main() -> None:
    """Prepare the licensed GoPro Flower benchmark assets and provenance."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing Flower_blur1.png and Flower_sharp1.png.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = Path(args.source_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    blurred = _resize(read_input_image(source / "Flower_blur1.png"))
    reference = _resize(read_input_image(source / "Flower_sharp1.png"))
    if blurred.shape != reference.shape:
        raise ValueError(
            "the official Flower pair has incompatible dimensions after resizing: "
            f"{blurred.shape} vs {reference.shape}"
        )
    noisy = deterministic_noise(reference, seed=NOISE_SEED, sigma=NOISE_SIGMA)
    write_png(output / "blurred.png", blurred)
    write_png(output / "reference.png", reference)
    write_png(output / "noisy.png", noisy)


def _resize(image):
    if max(image.shape[:2]) <= MAX_DIMENSION:
        return image
    scale = MAX_DIMENSION / max(image.shape[:2])
    size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    main()
