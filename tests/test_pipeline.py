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


if __name__ == "__main__":
    unittest.main()
