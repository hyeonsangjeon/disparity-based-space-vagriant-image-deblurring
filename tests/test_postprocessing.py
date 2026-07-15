import unittest

import numpy as np

from disparity_deblur.postprocessing import (
    guided_noisy_detail_fusion,
    luminance_unsharp,
)


class PostprocessingTest(unittest.TestCase):
    def test_zero_amount_preserves_input(self) -> None:
        image = np.random.default_rng(4).random((16, 16, 3), dtype=np.float32)
        np.testing.assert_array_equal(
            luminance_unsharp(image, sigma=1.0, amount=0.0, threshold=0.0),
            image,
        )
        np.testing.assert_array_equal(
            guided_noisy_detail_fusion(
                image,
                image,
                np.ones((16, 16), dtype=bool),
                denoise_strength=5.0,
                sigma=1.5,
                amount=0.0,
                tolerance=0.08,
                threshold=0.015,
            ),
            image,
        )

    def test_invalid_registration_mask_is_rejected(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "mask shape"):
            guided_noisy_detail_fusion(
                image,
                image,
                np.ones((8, 8), dtype=bool),
                denoise_strength=5.0,
                sigma=1.5,
                amount=0.5,
                tolerance=0.08,
                threshold=0.015,
            )


if __name__ == "__main__":
    unittest.main()
