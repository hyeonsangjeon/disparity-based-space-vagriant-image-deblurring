import unittest

import numpy as np

from disparity_deblur.segmentation import disparity_segmentation


class SegmentationTest(unittest.TestCase):
    def test_disparity_merges_color_oversegmentation(self) -> None:
        image = np.zeros((96, 128, 3), dtype=np.float32)
        image[:, :32] = (0.8, 0.1, 0.1)
        image[:, 32:64] = (0.7, 0.2, 0.1)
        image[:, 64:96] = (0.1, 0.7, 0.2)
        image[:, 96:] = (0.1, 0.6, 0.3)
        points = np.array(
            [
                [12, 24],
                [16, 48],
                [20, 72],
                [44, 24],
                [48, 48],
                [52, 72],
                [76, 24],
                [80, 48],
                [84, 72],
                [108, 24],
                [112, 48],
                [116, 72],
            ],
            dtype=np.float32,
        )
        disparities = np.array(
            [
                [1.0, 0.0],
                [1.1, 0.0],
                [0.9, 0.0],
                [1.1, 0.0],
                [1.0, 0.0],
                [1.2, 0.0],
                [5.0, 0.0],
                [5.1, 0.0],
                [4.9, 0.0],
                [5.1, 0.0],
                [5.0, 0.0],
                [5.2, 0.0],
            ],
            dtype=np.float32,
        )
        result = disparity_segmentation(
            image,
            points,
            disparities,
            color_labels=4,
            graph_cut_size=128,
            graph_cut_smoothness=0.2,
            disparity_threshold=0.5,
            minimum_disparity_features=2,
            max_regions=2,
            minimum_region_fraction=0.01,
        )
        self.assertEqual(int(result.merged_labels.max()) + 1, 2)


if __name__ == "__main__":
    unittest.main()
