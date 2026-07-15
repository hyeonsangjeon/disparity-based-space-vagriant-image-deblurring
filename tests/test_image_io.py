from pathlib import Path
import tempfile
import unittest

import numpy as np

from disparity_deblur.image_io import (
    read_input_image,
    read_rgb24,
    write_png,
    write_rgb24,
)


class RawIoTest(unittest.TestCase):
    def test_missing_file_points_to_sample(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "actual image file path"):
            read_input_image("/path/that/does/not/exist.raw")

    def test_raw_requires_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.raw"
            path.write_bytes(b"\x00" * 12)
            with self.assertRaisesRegex(ValueError, "requires both"):
                read_input_image(path)

    def test_png_detects_dimensions(self) -> None:
        image = np.linspace(0.0, 1.0, 5 * 7 * 3, dtype=np.float32).reshape(5, 7, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            write_png(path, image)
            restored = read_input_image(path)
        np.testing.assert_allclose(restored, image, atol=1.0 / 255.0)

    def test_rgb24_round_trip(self) -> None:
        image = np.linspace(0.0, 1.0, 5 * 7 * 3, dtype=np.float32).reshape(5, 7, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.raw"
            write_rgb24(path, image)
            restored = read_rgb24(path, 5, 7)
        np.testing.assert_allclose(restored, image, atol=1.0 / 255.0)

    def test_rejects_wrong_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.raw"
            path.write_bytes(b"\x00" * 10)
            with self.assertRaisesRegex(ValueError, "expected"):
                read_rgb24(path, 2, 2)


if __name__ == "__main__":
    unittest.main()
