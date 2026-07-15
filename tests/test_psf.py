from unittest import mock
import unittest

import numpy as np

from disparity_deblur.kernel_estimation import (
    _edge_rich_patch,
    estimate_region_kernels,
    estimate_tikhonov_kernel,
)
from disparity_deblur.restoration import psf_to_otf


class PsfTest(unittest.TestCase):
    def test_estimated_kernel_preserves_horizontal_direction(self) -> None:
        rng = np.random.default_rng(7)
        sharp = rng.random((128, 128))
        sharp = (sharp + np.roll(sharp, 1, 0) + np.roll(sharp, 1, 1)) / 3.0
        true_kernel = np.zeros((15, 15), dtype=np.float64)
        true_kernel[7, 4:11] = 1.0 / 7.0
        blurred = np.fft.ifft2(
            np.fft.fft2(sharp) * psf_to_otf(true_kernel, sharp.shape)
        ).real
        estimated, _ = estimate_tikhonov_kernel(
            blurred,
            sharp,
            np.ones(sharp.shape, dtype=bool),
            kernel_size=15,
            patch_size=128,
            regularization=1e-3,
        )
        yy, xx = np.indices(estimated.shape)
        center = (np.array(estimated.shape) - 1) / 2.0
        horizontal_variance = np.sum(estimated * (xx - center[1]) ** 2)
        vertical_variance = np.sum(estimated * (yy - center[0]) ** 2)
        self.assertAlmostEqual(float(estimated.sum()), 1.0, places=8)
        self.assertGreater(horizontal_variance, vertical_variance)

    def test_patch_selection_uses_each_rectangular_axis(self) -> None:
        for shape, expected in (
            ((40, 120), (40, 64)),
            ((120, 40), (64, 40)),
        ):
            with self.subTest(shape=shape):
                mask = np.ones(shape, dtype=bool)
                edge_energy = np.ones(shape, dtype=np.float32)
                y0, y1, x0, x1 = _edge_rich_patch(mask, edge_energy, 64)
                self.assertEqual((y1 - y0, x1 - x0), expected)

    def test_bad_kernel_is_replaced_from_nearest_disparity(self) -> None:
        reliable = np.arange(1, 50, dtype=np.float64).reshape(7, 7)
        reliable /= reliable.sum()
        unreliable = np.zeros((7, 7), dtype=np.float64)
        unreliable[3, 3] = 1.0
        labels = np.zeros((16, 16), dtype=np.int32)
        labels[:, 8:] = 1
        disparities = np.array([[0.0, 0.0], [0.2, 0.0]])
        bounds = (0, 16, 0, 16)

        with mock.patch(
            "disparity_deblur.kernel_estimation.estimate_tikhonov_kernel",
            side_effect=[(reliable, bounds), (unreliable, bounds)],
        ):
            estimates = estimate_region_kernels(
                np.zeros((16, 16, 3), dtype=np.float32),
                np.zeros((16, 16, 3), dtype=np.float32),
                labels,
                disparities,
                kernel_size=7,
                patch_size=16,
                kurtosis_min=0.0,
                kurtosis_max=10.0,
            )

        self.assertIsNone(estimates[0].replaced_from)
        self.assertEqual(estimates[1].replaced_from, 0)
        np.testing.assert_allclose(estimates[1].kernel, reliable)


if __name__ == "__main__":
    unittest.main()
