import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from disparity_deblur.benchmark import (
    BenchmarkRunner,
    CandidateEvaluation,
    Dataset,
    InputFile,
    _validate_public_reference_config,
    _thumbnail,
    deterministic_noise,
    load_manifest,
    select_hpo_candidate,
)
from disparity_deblur.config import PipelineConfig
from disparity_deblur.image_io import write_png


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BenchmarkTest(unittest.TestCase):
    def test_manifest_rejects_absolute_input_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "datasets": [
                            {
                                "id": "unsafe",
                                "description": "unsafe fixture",
                                "visibility": "private",
                                "blurred": {"path": "/private.png", "sha256": "0" * 64},
                                "noisy": {"path": "noisy.png", "sha256": "0" * 64},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "safe relative path"):
                load_manifest(path)

    def test_public_manifest_lists_only_the_four_authorized_datasets(self) -> None:
        manifest = load_manifest(Path("benchmarks/manifests/public.json"))
        self.assertEqual(
            [dataset.identifier for dataset in manifest.datasets],
            [
                "gopro-flower",
                "01_df2_object_motion",
                "02_building_low_light",
                "05_new1_parking",
            ],
        )
        historical = manifest.datasets[1:]
        self.assertTrue(all(dataset.reference is None for dataset in historical))
        self.assertTrue(
            all(
                dataset.rights
                == (
                    "Copyright © 2026 Hyeon Sang Jeon. All rights reserved. "
                    "Included in this repository for demonstration and archival "
                    "presentation only. No reuse, redistribution, or derivative use "
                    "is granted."
                )
                for dataset in historical
            )
        )
        self.assertEqual(
            manifest.datasets[1].noisy_label,
            "Noisy (deterministic noise boost)",
        )
        self.assertEqual(
            manifest.datasets[1].input_processing["noisy"]["sigma"],
            0.02,
        )
        self.assertEqual(
            manifest.datasets[3].config_overrides["noisy_structure_blend"],
            0.65,
        )
        building = json.loads(
            Path("showcase/benchmark/02_building_low_light/run.json").read_text()
        )
        self.assertNotEqual(
            building["output_size"]["width"],
            building["output_size"]["height"],
        )

    def test_noise_is_repeatable_and_seeded(self) -> None:
        image = np.full((12, 12, 3), 0.5, dtype=np.float32)
        first = deterministic_noise(image, seed=12, sigma=0.05)
        second = deterministic_noise(image, seed=12, sigma=0.05)
        different_seed = deterministic_noise(image, seed=13, sigma=0.05)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different_seed))

    def test_thumbnail_bounds_portrait_and_landscape_longest_axes(self) -> None:
        portrait = np.zeros((1200, 400, 3), dtype=np.float32)
        landscape = np.zeros((400, 1200, 3), dtype=np.float32)
        self.assertEqual(_thumbnail(portrait).shape, (640, 213, 3))
        self.assertEqual(_thumbnail(landscape).shape, (213, 640, 3))

    def test_public_reference_blend_is_capped(self) -> None:
        dataset = Dataset(
            identifier="reference-fixture",
            description="reference-backed fixture",
            visibility="public",
            rights=None,
            blurred=InputFile("blurred.png", "0" * 64),
            noisy=InputFile("noisy.png", "0" * 64),
            reference=InputFile("reference.png", "0" * 64),
        )
        with self.assertRaisesRegex(ValueError, "above 0.25"):
            _validate_public_reference_config(
                dataset,
                PipelineConfig(
                    kernel_size=7,
                    patch_size=32,
                    input_detail_blend=0.26,
                ),
            )
        with self.assertRaisesRegex(ValueError, "above 0.25"):
            _validate_public_reference_config(
                dataset,
                PipelineConfig(
                    kernel_size=7,
                    patch_size=32,
                    noisy_structure_blend=0.26,
                ),
            )

    def test_manifest_rejects_unknown_config_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "datasets": [
                            {
                                "id": "invalid-override",
                                "description": "invalid override fixture",
                                "visibility": "private",
                                "blurred": {
                                    "path": "blurred.png",
                                    "sha256": "0" * 64,
                                },
                                "noisy": {
                                    "path": "noisy.png",
                                    "sha256": "0" * 64,
                                },
                                "config_overrides": {
                                    "not_a_pipeline_field": 0.5,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown PipelineConfig"):
                load_manifest(path)

    def test_hpo_selection_uses_objective(self) -> None:
        base = PipelineConfig(kernel_size=7, patch_size=32)
        low = CandidateEvaluation(base, {"objective": 0.2}, 0.01)
        high = CandidateEvaluation(replace(base, tikhonov=0.002), {"objective": 0.8}, 0.02)
        self.assertIs(select_hpo_candidate((low, high)), high)

    def test_runner_writes_relative_web_assets_without_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "source-assets"
            assets.mkdir()
            yy, xx = np.mgrid[:64, :64]
            reference = np.stack(
                (xx / 63.0, yy / 63.0, ((xx + yy) % 16) / 15.0), axis=-1
            ).astype(np.float32)
            blurred = np.clip(reference * 0.85 + 0.04, 0.0, 1.0)
            noisy = deterministic_noise(reference, seed=10, sigma=0.02)
            for name, image in (
                ("blurred.png", blurred),
                ("noisy.png", noisy),
                ("reference.png", reference),
            ):
                write_png(assets / name, image)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "datasets": [
                            {
                                "id": "fixture",
                                "description": "benchmark fixture",
                                "visibility": "public",
                                "blurred": {
                                    "path": "blurred.png",
                                    "sha256": _digest(assets / "blurred.png"),
                                },
                                "noisy": {
                                    "path": "noisy.png",
                                    "sha256": _digest(assets / "noisy.png"),
                                },
                                "reference": {
                                    "path": "reference.png",
                                    "sha256": _digest(assets / "reference.png"),
                                },
                                "config_overrides": {
                                    "noisy_structure_blend": 0.2,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def processor(
                image: np.ndarray, _noisy: np.ndarray, config: PipelineConfig
            ) -> np.ndarray:
                return reference if config.data_weight == 220.0 else image

            output = root / "output"
            index = BenchmarkRunner(
                load_manifest(manifest_path),
                dataset_root=assets,
                output_dir=output,
                hpo_max_dimension=64,
                output_max_dimension=64,
                processor=processor,
            ).run()
            record = json.loads(index.read_text(encoding="utf-8"))["datasets"][0]
            self.assertEqual(record["objective_type"], "reference")
            self.assertEqual(record["hpo"]["max_dimension"], 64)
            self.assertIn("baseline_metrics", record)
            self.assertEqual(record["selected_config"]["noisy_structure_blend"], 0.2)
            self.assertEqual(record["config_overrides"]["noisy_structure_blend"], 0.2)
            self.assertIn("asset_checksums", record)
            for role, checksum in record["asset_checksums"].items():
                self.assertEqual(checksum, _digest(output / record["assets"][role]))
            self.assertTrue((output / "fixture" / "comparison.webp").is_file())
            self.assertTrue((output / "fixture" / "thumbnail.webp").is_file())
            self.assertFalse((output / "fixture" / "comparison.png").exists())
            self.assertTrue((output / "SUMMARY.md").is_file())
            self.assertNotIn(str(assets), (output / "fixture" / "run.json").read_text())


if __name__ == "__main__":
    unittest.main()
