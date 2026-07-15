import hashlib
import json
from pathlib import Path
import unittest


class Mat3WalkthroughTest(unittest.TestCase):
    def test_walkthrough_metadata_is_public_safe_and_complete(self) -> None:
        root = Path("benchmarks/public-assets/mat3-living-room")
        metadata_path = root / "walkthrough.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(metadata["id"], "mat3-living-room-walkthrough")
        self.assertEqual(metadata["metric_type"], "no-reference-proxy")
        self.assertNotIn("psnr", metadata["proxy_metrics"])
        self.assertEqual(metadata["selected_hpo"]["kernel"]["kernel_size"], 17)
        self.assertEqual(metadata["selected_hpo"]["guided_detail_fusion"]["amount"], 0.5)
        self.assertNotIn("/Users/", json.dumps(metadata))
        for relative_path, digest in metadata["asset_checksums"].items():
            self.assertEqual(
                hashlib.sha256((root / relative_path).read_bytes()).hexdigest(), digest
            )

    def test_public_manifest_records_the_featured_walkthrough(self) -> None:
        manifest = json.loads(
            Path("benchmarks/manifests/public.json").read_text(encoding="utf-8")
        )
        walkthrough = manifest["walkthroughs"][0]
        root = Path("benchmarks/public-assets")
        self.assertEqual(walkthrough["id"], "mat3-living-room-walkthrough")
        self.assertEqual(
            hashlib.sha256((root / walkthrough["metadata"]["path"]).read_bytes()).hexdigest(),
            walkthrough["metadata"]["sha256"],
        )
        self.assertEqual(len(walkthrough["assets"]), 10)


if __name__ == "__main__":
    unittest.main()
