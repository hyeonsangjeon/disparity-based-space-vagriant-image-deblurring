"""Explicit optional backend dispatch for TV/L1 deconvolution."""

from __future__ import annotations

import numpy as np


BACKENDS = ("numpy", "native-cpp", "cuda")


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly selected optional backend cannot run."""


def deconvolve_channels(
    channels: np.ndarray,
    kernel: np.ndarray,
    *,
    data_weight: float,
    beta_max: float,
    backend: str,
) -> np.ndarray:
    """Run one optional backend without silently changing implementations."""

    if backend == "native-cpp":
        from . import native_cpp_backend

        try:
            return native_cpp_backend.deconvolve(
                channels,
                kernel,
                data_weight=data_weight,
                beta_max=beta_max,
            )
        except ImportError as error:
            raise BackendUnavailableError(
                "native-cpp backend is not built; run "
                "`uv sync --group native` and "
                "`uv run --group native python tools/build_native.py`"
            ) from error
    if backend == "cuda":
        from . import cuda_backend

        try:
            return cuda_backend.deconvolve(
                channels,
                kernel,
                data_weight=data_weight,
                beta_max=beta_max,
            )
        except (ImportError, cuda_backend.CudaUnavailableError) as error:
            raise BackendUnavailableError(
                "cuda backend requires a CUDA-capable NVIDIA GPU and a matching "
                "CuPy package such as `cupy-cuda12x`"
            ) from error
    raise ValueError(f"unsupported optional backend: {backend}")


def backend_status() -> dict[str, bool]:
    """Report availability without selecting or falling back to a backend."""

    from . import cuda_backend, native_cpp_backend

    return {
        "numpy": True,
        "native-cpp": native_cpp_backend.available(),
        "cuda": cuda_backend.available(),
    }
