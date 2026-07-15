"""Measure installed deconvolution backends without making speed claims."""

from __future__ import annotations

import argparse
import json
from statistics import median
from time import perf_counter

import numpy as np

from disparity_deblur.backends import BACKENDS, backend_status
from disparity_deblur.restoration import tv_l1_deconvolve


def main() -> None:
    """Benchmark available backends and print a reproducible JSON report."""

    parser = argparse.ArgumentParser(
        description="Benchmark available TV/L1 backends on deterministic RGB data."
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--kernel-size", type=int, default=15)
    parser.add_argument("--beta-max", type=float, default=8.0)
    parser.add_argument("--data-weight", type=float, default=160.0)
    parser.add_argument("--boundary-padding", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.height < 2 or args.width < 2 or args.repeats < 1:
        raise ValueError("height, width, and repeats must be positive")
    if args.kernel_size < 1 or args.kernel_size % 2 == 0:
        raise ValueError("kernel size must be a positive odd number")

    rng = np.random.default_rng(20260715)
    image = rng.random((args.height, args.width, 3))
    kernel = np.zeros((args.kernel_size, args.kernel_size), dtype=np.float64)
    center = args.kernel_size // 2
    radius = max(1, args.kernel_size // 4)
    kernel[center, center - radius : center + radius + 1] = 1.0
    kernel /= kernel.sum()
    availability = backend_status()
    records: list[dict[str, object]] = []
    outputs: dict[str, np.ndarray] = {}

    for backend in BACKENDS:
        if not availability[backend]:
            records.append({"backend": backend, "available": False})
            continue
        durations = []
        for _ in range(args.repeats + 1):
            started = perf_counter()
            output = tv_l1_deconvolve(
                image,
                kernel,
                data_weight=args.data_weight,
                beta_max=args.beta_max,
                boundary_padding=args.boundary_padding,
                backend=backend,
            )
            durations.append(perf_counter() - started)
        outputs[backend] = output
        records.append(
            {
                "backend": backend,
                "available": True,
                "median_seconds": median(durations[1:]),
            }
        )

    reference = outputs["numpy"]
    for record in records:
        backend = str(record["backend"])
        if backend in outputs:
            record["max_abs_error_vs_numpy"] = float(
                np.max(np.abs(outputs[backend] - reference))
            )
    print(
        json.dumps(
            {
                "shape": [args.height, args.width, 3],
                "kernel_size": args.kernel_size,
                "beta_max": args.beta_max,
                "boundary_padding": args.boundary_padding,
                "results": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
