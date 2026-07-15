from unittest import mock
import unittest

import numpy as np

from disparity_deblur.restoration import (
    normalized_region_merge,
    psf_to_otf,
    restore_regions,
    tv_l1_deconvolve,
)


class RestorationTest(unittest.TestCase):
    def test_known_kernel_reduces_blur_error(self) -> None:
        size = 96
        sharp = np.zeros((size, size), dtype=np.float64)
        sharp[20:76, 24:72] = 0.8
        sharp[35:60, 10:86] = 1.0
        kernel = np.zeros((9, 9), dtype=np.float64)
        kernel[4, 2:7] = 0.2
        blurred = np.fft.ifft2(
            np.fft.fft2(sharp) * psf_to_otf(kernel, sharp.shape)
        ).real
        restored = tv_l1_deconvolve(
            blurred,
            kernel,
            data_weight=600.0,
            beta_max=32.0,
        )
        blurred_error = np.mean((blurred - sharp) ** 2)
        restored_error = np.mean((restored - sharp) ** 2)
        self.assertLess(restored_error, blurred_error)

    def test_normalized_merge_preserves_constant_image(self) -> None:
        labels = np.zeros((32, 48), dtype=np.int32)
        labels[:, 24:] = 1
        image = np.full((32, 48, 3), 0.4, dtype=np.float64)
        merged = normalized_region_merge(
            [image, image],
            labels,
            feather_sigma=3.0,
        )
        np.testing.assert_allclose(merged, image, atol=1e-12)

    def test_duplicate_regional_kernels_are_deconvolved_once(self) -> None:
        labels = np.zeros((24, 40), dtype=np.int32)
        labels[:, 14:28] = 1
        labels[:, 28:] = 2
        image = np.full((24, 40, 3), 0.4, dtype=np.float64)
        first = np.zeros((5, 5), dtype=np.float64)
        first[2, 2] = 1.0
        second = np.zeros((5, 5), dtype=np.float64)
        second[2, 1:4] = 1.0 / 3.0

        with mock.patch(
            "disparity_deblur.restoration.tv_l1_deconvolve",
            side_effect=lambda source, _kernel, **_kwargs: source.copy(),
        ) as deconvolve:
            restored = restore_regions(
                image,
                labels,
                [first, first.copy(), second],
                feather_sigma=2.0,
            )

        self.assertEqual(deconvolve.call_count, 2)
        np.testing.assert_allclose(restored, image, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
