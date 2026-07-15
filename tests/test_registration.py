import unittest

import cv2
import numpy as np

from disparity_deblur.registration import register_noisy_to_blur


class RegistrationTest(unittest.TestCase):
    def test_registers_wide_and_tall_frames_without_shape_changes(self) -> None:
        for height, width in ((72, 120), (120, 72)):
            with self.subTest(height=height, width=width):
                rng = np.random.default_rng(height + width)
                blurred = cv2.GaussianBlur(
                    rng.random((height, width, 3), dtype=np.float32),
                    (0, 0),
                    0.8,
                )
                noisy = cv2.warpAffine(
                    blurred,
                    np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]], dtype=np.float32),
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )

                result = register_noisy_to_blur(
                    blurred,
                    noisy,
                    max_corners=600,
                    ransac_threshold=1.5,
                )

                self.assertEqual(result.registered_noisy.shape, (height, width, 3))
                self.assertEqual(result.valid_mask.shape, (height, width))
                self.assertGreaterEqual(len(result.blur_points), 8)
                error = np.mean(
                    np.abs(
                        result.registered_noisy[result.valid_mask]
                        - blurred[result.valid_mask]
                    )
                )
                self.assertLess(float(error), 0.08)


if __name__ == "__main__":
    unittest.main()
