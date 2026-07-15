"""Lazy adapter for the optional pybind11 C++ extension."""

from __future__ import annotations

import numpy as np


def available() -> bool:
    """Return whether the in-place native extension can be imported."""

    try:
        from . import _native  # noqa: F401
    except ImportError:
        return False
    return True


def deconvolve(
    channels: np.ndarray,
    kernel: np.ndarray,
    *,
    data_weight: float,
    beta_max: float,
) -> np.ndarray:
    """Execute the C++17 TV/L1 implementation and return float64 RGB data."""

    from . import _native

    return np.asarray(
        _native.tv_l1_deconvolve(
            np.ascontiguousarray(channels, dtype=np.float64),
            np.ascontiguousarray(kernel, dtype=np.float64),
            data_weight,
            beta_max,
        ),
        dtype=np.float64,
    )


def implementation_info() -> dict[str, object]:
    """Return build metadata from the loaded extension."""

    from . import _native

    return dict(_native.implementation_info())
