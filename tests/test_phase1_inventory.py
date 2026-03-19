import json
import tempfile
import unittest
from pathlib import Path

from coral_thesis.phases.chart_detection import (
    build_chart_dataset_inventory,
    prepare_chart_detection_dataset,
)


class Phase1InventoryTests(unittest.TestCase):
    def _write(self, path: Path, contents: str) -> None:
        path.write_text(contents, encoding="utf-8")

    def _touch_image(self, path: Path) -> None:
        path.write_bytes(b"fake-image")

    def test_inventory_reports_missing_and_invalid_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir)
            self._touch_image(dataset_dir / "img_a.JPG")
            self._touch_image(dataset_dir / "img_b.JPG")
            self._write(dataset_dir / "img_a.txt", "0 0.5 0.5 0.2 0.2\n")
            self._write(dataset_dir / "img_b.txt", "0 1.2 0.5 0.2 0.2\n")
            self._write(dataset_dir / "orphan.txt", "0 0.5 0.5 0.2 0.2\n")
            self._write(dataset_dir / "img_a.json", "{}\n")

            inventory = build_chart_dataset_inventory(dataset_dir=dataset_dir, class_name="chart")
            summary = inventory.summary()

            self.assertEqual(summary["image_count"], 2)
            self.assertEqual(summary["labeled_image_count"], 2)
            self.assertEqual(summary["valid_labeled_image_count"], 1)
            self.assertEqual(summary["orphan_label_count"], 1)
            self.assertEqual(summary["json_annotation_count"], 1)
            self.assertGreaterEqual(summary["issue_count"], 2)

    def test_inventory_tolerates_rounding_error_on_image_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir)
            self._touch_image(dataset_dir / "img_a.JPG")
            self._write(dataset_dir / "img_a.txt", "0 0.727927 0.773994 0.334836 0.452013\n")

            inventory = build_chart_dataset_inventory(dataset_dir=dataset_dir, class_name="chart")
            summary = inventory.summary()

            self.assertEqual(summary["valid_labeled_image_count"], 1)
            self.assertEqual(summary["issue_count"], 0)

    def test_preparation_creates_data_yaml_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_dir = root / "dataset"
            output_dir = root / "prepared"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            for image_id in ("a", "b", "c"):
                self._touch_image(dataset_dir / f"{image_id}.JPG")
                self._write(dataset_dir / f"{image_id}.txt", "0 0.5 0.5 0.2 0.2\n")

            inventory = build_chart_dataset_inventory(dataset_dir=dataset_dir, class_name="chart")
            prepared = prepare_chart_detection_dataset(
                inventory=inventory,
                output_dir=output_dir,
                class_name="chart",
                val_split=0.34,
                seed=7,
                use_symlinks=False,
                skip_unlabeled_images=True,
            )

            self.assertTrue(prepared.data_yaml_path.exists())
            self.assertTrue(prepared.manifest_path.exists())
            self.assertEqual(prepared.train_count + prepared.val_count, 3)

            manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["train_image_ids"]) + len(manifest["val_image_ids"]), 3)
            self.assertTrue((output_dir / "images" / "train").exists())
            self.assertTrue((output_dir / "labels" / "val").exists())


if __name__ == "__main__":
    unittest.main()
