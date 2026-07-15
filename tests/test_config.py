import unittest

from disparity_deblur.config import PipelineConfig


class PipelineConfigTest(unittest.TestCase):
    def test_rejects_even_kernel(self) -> None:
        with self.assertRaisesRegex(ValueError, "odd"):
            PipelineConfig(kernel_size=32)

    def test_resolves_automatic_boundary_padding(self) -> None:
        self.assertEqual(PipelineConfig(kernel_size=33).effective_boundary_padding, 32)
        self.assertEqual(
            PipelineConfig(kernel_size=33, boundary_padding=8).effective_boundary_padding,
            8,
        )

    def test_rejects_unknown_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "backend"):
            PipelineConfig(backend="automatic")


if __name__ == "__main__":
    unittest.main()
