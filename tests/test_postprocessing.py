import unittest

import numpy as np

from disparity_deblur.postprocessing import (
    guided_noisy_detail_fusion,
    guided_noisy_structure_fusion,
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

    def test_noisy_structure_fusion_reduces_high_frequency_ringing(self) -> None:
        yy, xx = np.mgrid[:48, :48]
        base = np.stack(
            (
                0.3 + 0.4 * xx / 47.0,
                0.25 + 0.35 * yy / 47.0,
                0.35 + 0.2 * (xx + yy) / 94.0,
            ),
            axis=-1,
        ).astype(np.float32)
        oscillation = np.where(xx % 2 == 0, 1.0, -1.0)[..., None].astype(
            np.float32
        )
        restored = np.clip(base + 0.16 * oscillation, 0.0, 1.0)
        noisy = np.clip(base + 0.03 * oscillation, 0.0, 1.0)

        fused = guided_noisy_structure_fusion(
            restored,
            noisy,
            np.ones((48, 48), dtype=np.float32),
            denoise_strength=0.0,
            sigma=2.0,
            amount=0.75,
            tolerance=1.0,
        )

        self.assertLess(
            np.mean(np.abs(fused - base)),
            np.mean(np.abs(restored - base)),
        )
        self.assertTrue(np.isfinite(fused).all())
        self.assertGreaterEqual(float(fused.min()), 0.0)
        self.assertLessEqual(float(fused.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
