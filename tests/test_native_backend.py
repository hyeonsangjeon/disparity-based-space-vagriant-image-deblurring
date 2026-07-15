import unittest

import numpy as np

from disparity_deblur.native_cpp_backend import (
    available,
    implementation_info,
)
from disparity_deblur.restoration import tv_l1_deconvolve


@unittest.skipUnless(available(), "optional C++ backend is not built")
class NativeBackendTest(unittest.TestCase):
    def test_matches_numpy_for_square_and_odd_rectangular_inputs(self) -> None:
        rng = np.random.default_rng(42)
        kernel = np.zeros((5, 5), dtype=np.float64)
        kernel[2, 1:4] = 1.0 / 3.0
        for shape in ((24, 24, 3), (17, 23, 3), (23, 17, 1)):
            with self.subTest(shape=shape):
                image = rng.random(shape)
                expected = tv_l1_deconvolve(
                    image,
                    kernel,
                    data_weight=160.0,
                    beta_max=4.0,
                    boundary_padding=4,
                    backend="numpy",
                )
                actual = tv_l1_deconvolve(
                    image,
                    kernel,
                    data_weight=160.0,
                    beta_max=4.0,
                    boundary_padding=4,
                    backend="native-cpp",
                )
                np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)

    def test_reports_portable_rectangular_implementation(self) -> None:
        info = implementation_info()
        self.assertEqual(info["name"], "portable-cpp17")
        self.assertEqual(info["fft"], "radix-2/Bluestein")
        self.assertTrue(info["supports_rectangular"])


if __name__ == "__main__":
    unittest.main()
