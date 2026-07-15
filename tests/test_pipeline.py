import unittest

import numpy as np

from disparity_deblur.config import PipelineConfig
from disparity_deblur.pipeline import DisparityDeblurPipeline


class PipelineTest(unittest.TestCase):
    def test_processes_in_memory_pair(self) -> None:
        yy, xx = np.mgrid[:64, :64]
        image = np.stack(
            (
                ((xx // 8 + yy // 8) % 2).astype(np.float32),
                xx.astype(np.float32) / 63.0,
                yy.astype(np.float32) / 63.0,
            ),
            axis=-1,
        )
        config = PipelineConfig(
            kernel_size=7,
            color_labels=4,
            graph_cut_size=64,
            max_regions=2,
            patch_size=32,
            skip_registration=True,
        )
        result = DisparityDeblurPipeline(config).process(image, image)
        self.assertEqual(result.restored.shape, image.shape)
        self.assertEqual(len(result.kernels), int(result.segmentation.merged_labels.max()) + 1)
        self.assertTrue(np.isfinite(result.restored).all())

    def test_processes_wide_tall_and_odd_sized_pairs(self) -> None:
        config = PipelineConfig(
            kernel_size=7,
            color_labels=4,
            graph_cut_size=64,
            max_regions=2,
            patch_size=32,
            skip_registration=True,
        )
        for height, width in ((48, 80), (80, 48), (53, 87)):
            with self.subTest(height=height, width=width):
                yy, xx = np.mgrid[:height, :width]
                image = np.stack(
                    (
                        ((xx // 7 + yy // 9) % 2).astype(np.float32),
                        xx.astype(np.float32) / max(width - 1, 1),
                        yy.astype(np.float32) / max(height - 1, 1),
                    ),
                    axis=-1,
                )
                result = DisparityDeblurPipeline(config).process(image, image)
                self.assertEqual(result.restored.shape, (height, width, 3))
                self.assertEqual(result.registration.valid_mask.shape, (height, width))
                self.assertEqual(result.segmentation.merged_labels.shape, (height, width))
                self.assertTrue(np.isfinite(result.restored).all())


if __name__ == "__main__":
    unittest.main()
