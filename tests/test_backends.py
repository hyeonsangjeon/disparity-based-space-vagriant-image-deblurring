import unittest
from unittest import mock

import numpy as np

from disparity_deblur.backends import (
    BackendUnavailableError,
    backend_status,
    deconvolve_channels,
)
from disparity_deblur.cuda_backend import (
    CudaUnavailableError,
    _deconvolve_array_module,
)
from disparity_deblur.restoration import tv_l1_deconvolve


class BackendTest(unittest.TestCase):
    def test_numpy_is_always_available(self) -> None:
        self.assertTrue(backend_status()["numpy"])

    def test_native_selection_never_silently_falls_back(self) -> None:
        with mock.patch(
            "disparity_deblur.native_cpp_backend.deconvolve",
            side_effect=ImportError("missing extension"),
        ):
            with self.assertRaisesRegex(BackendUnavailableError, "not built"):
                deconvolve_channels(
                    np.zeros((8, 10, 3), dtype=np.float64),
                    np.ones((3, 3), dtype=np.float64) / 9.0,
                    data_weight=100.0,
                    beta_max=2.0,
                    backend="native-cpp",
                )

    def test_unknown_optional_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            deconvolve_channels(
                np.zeros((8, 10, 3), dtype=np.float64),
                np.ones((3, 3), dtype=np.float64) / 9.0,
                data_weight=100.0,
                beta_max=2.0,
                backend="automatic",
            )

    def test_cuda_selection_never_silently_falls_back(self) -> None:
        with mock.patch(
            "disparity_deblur.cuda_backend.deconvolve",
            side_effect=CudaUnavailableError("no device"),
        ):
            with self.assertRaisesRegex(BackendUnavailableError, "NVIDIA GPU"):
                deconvolve_channels(
                    np.zeros((8, 10, 3), dtype=np.float64),
                    np.ones((3, 3), dtype=np.float64) / 9.0,
                    data_weight=100.0,
                    beta_max=2.0,
                    backend="cuda",
                )

    def test_cuda_array_contract_matches_numpy_solver(self) -> None:
        rng = np.random.default_rng(17)
        image = rng.random((15, 21, 3))
        kernel = np.zeros((5, 5), dtype=np.float64)
        kernel[2, 1:4] = 1.0 / 3.0
        expected = tv_l1_deconvolve(
            image,
            kernel,
            data_weight=120.0,
            beta_max=4.0,
            boundary_padding=0,
            backend="numpy",
        )
        actual = _deconvolve_array_module(
            image,
            kernel,
            data_weight=120.0,
            beta_max=4.0,
            array_module=np,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
