import importlib.util
from pathlib import Path
import unittest

import numpy as np


def _generator_module():
    path = Path("tools/generate_synthetic_spatial_psf.py")
    specification = importlib.util.spec_from_file_location("synthetic_generator", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load synthetic benchmark generator")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class SyntheticGeneratorTest(unittest.TestCase):
    def test_generator_is_deterministic_and_masks_cover_the_scene(self) -> None:
        generator = _generator_module()
        first, metadata = generator.generate_dataset(max_dimension=128)
        second, _ = generator.generate_dataset(max_dimension=128)

        self.assertEqual(metadata["master_size"], {"width": 1024, "height": 768})
        self.assertEqual(metadata["output_size"], {"width": 128, "height": 96})
        for name in ("blurred", "noisy", "reference", "far", "mid", "near"):
            self.assertEqual(first[name].shape[:2], (96, 128))
            np.testing.assert_array_equal(first[name], second[name])
        coverage = first["far"] + first["mid"] + first["near"]
        np.testing.assert_allclose(coverage, np.ones((96, 128)), atol=1e-5)
        self.assertTrue(all(region["psf"]["normalized_sum"] == 1.0 for region in metadata["regions"]))


if __name__ == "__main__":
    unittest.main()
