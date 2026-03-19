import json
import tempfile
import unittest
from pathlib import Path

from coral_thesis.phases.coral_segmentation import (
    build_segmentation_dataset_inventory,
    load_legacy_segmentation_split_reference,
    parse_yolo_segmentation_label,
    prepare_segmentation_dataset,
)


class Phase3InventoryTests(unittest.TestCase):
    def test_parse_yolo_segmentation_label_parses_polygon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            label_path = Path(tmp_dir) / "sample.txt"
            label_path.write_text(
                "0 0.100000 0.100000 0.900000 0.100000 0.500000 0.900000\n",
                encoding="utf-8",
            )

            polygons, issues = parse_yolo_segmentation_label(label_path)

            self.assertEqual(len(polygons), 1)
            self.assertEqual(polygons[0].class_id, 0)
            self.assertEqual(len(polygons[0].points), 3)
            self.assertEqual(polygons[0].source_format, "polygon")
            self.assertEqual(len(issues), 0)

    def test_parse_yolo_segmentation_label_converts_bbox_to_polygon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            label_path = Path(tmp_dir) / "sample.txt"
            label_path.write_text(
                "0 0.500000 0.500000 0.400000 0.200000\n",
                encoding="utf-8",
            )

            polygons, issues = parse_yolo_segmentation_label(label_path)

            self.assertEqual(len(polygons), 1)
            self.assertEqual(
                polygons[0].points,
                (
                    (0.3, 0.4),
                    (0.7, 0.4),
                    (0.7, 0.6),
                    (0.3, 0.6),
                ),
            )
            self.assertEqual(polygons[0].source_format, "bbox")
            self.assertEqual(len(issues), 0)

    def test_build_segmentation_inventory_reports_unlabeled_and_orphans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir)
            for image_name in ("a.JPG", "b.JPG", "d.JPG"):
                (dataset_dir / image_name).write_bytes(b"image")
            (dataset_dir / "a.txt").write_text(
                "0 0.1 0.1 0.9 0.1 0.5 0.9\n",
                encoding="utf-8",
            )
            (dataset_dir / "d.txt").write_text(
                "0 0.1 0.1 0.9 0.1 0.5\n",
                encoding="utf-8",
            )
            (dataset_dir / "orphan.txt").write_text(
                "0 0.1 0.1 0.9 0.1 0.5 0.9\n",
                encoding="utf-8",
            )

            inventory = build_segmentation_dataset_inventory(dataset_dir, class_name="coral")
            summary = inventory.summary()

            self.assertEqual(summary["image_count"], 3)
            self.assertEqual(summary["labeled_image_count"], 2)
            self.assertEqual(summary["unlabeled_image_count"], 1)
            self.assertEqual(summary["valid_labeled_image_count"], 1)
            self.assertEqual(summary["orphan_label_count"], 1)
            self.assertGreater(summary["issue_count"], 0)
            self.assertEqual(summary["annotation_format_counts"], {"polygon": 1})
            self.assertEqual(summary["unlabeled_image_ids"], ["b"])

    def test_build_segmentation_inventory_supports_external_label_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_dir = root / "dataset"
            labels_dir = root / "labels" / "train"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "a.JPG").write_bytes(b"image")
            (dataset_dir / "b.JPG").write_bytes(b"image")
            (labels_dir / "a.txt").write_text(
                "0 0.5 0.5 0.4 0.2\n",
                encoding="utf-8",
            )

            inventory = build_segmentation_dataset_inventory(
                dataset_dir=dataset_dir,
                label_dir=root / "labels",
                class_name="coral",
            )
            summary = inventory.summary()

            self.assertEqual(Path(summary["label_dir"]).resolve(), (root / "labels").resolve())
            self.assertEqual(summary["labeled_image_count"], 1)
            self.assertEqual(summary["valid_labeled_image_count"], 1)
            self.assertEqual(summary["annotation_format_counts"], {"bbox": 1})
            self.assertEqual(summary["unlabeled_image_ids"], ["b"])

    def test_load_legacy_segmentation_split_deduplicates_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_dir = root / "dataset"
            train_dir = root / "legacy" / "images" / "train"
            val_dir = root / "legacy" / "images" / "val"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)

            source_image = dataset_dir / "sample.JPG"
            val_image = dataset_dir / "val_only.JPG"
            source_image.write_bytes(b"image")
            val_image.write_bytes(b"image")

            (train_dir / "sample.JPG").symlink_to(source_image)
            (train_dir / "sample 2.JPG").symlink_to(source_image)
            (val_dir / "val_only.JPG").symlink_to(val_image)

            reference = load_legacy_segmentation_split_reference(root / "legacy")

            self.assertEqual(reference.train_image_ids, ("sample",))
            self.assertEqual(reference.val_image_ids, ("val_only",))
            self.assertIn("sample 2.JPG", reference.duplicate_alias_names)
            self.assertEqual(reference.conflicting_image_ids, ())

    def test_prepare_segmentation_dataset_uses_legacy_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_dir = root / "dataset"
            legacy_root = root / "legacy"
            train_dir = legacy_root / "images" / "train"
            val_dir = legacy_root / "images" / "val"
            output_dir = root / "prepared"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)

            for image_id in ("a", "b", "c"):
                (dataset_dir / f"{image_id}.JPG").write_bytes(b"image")
                (dataset_dir / f"{image_id}.txt").write_text(
                    "0 0.5 0.5 0.4 0.2\n",
                    encoding="utf-8",
                )

            (train_dir / "a.JPG").symlink_to(dataset_dir / "a.JPG")
            (train_dir / "b.JPG").symlink_to(dataset_dir / "b.JPG")
            (train_dir / "b 2.JPG").symlink_to(dataset_dir / "b.JPG")
            (val_dir / "c.JPG").symlink_to(dataset_dir / "c.JPG")

            inventory = build_segmentation_dataset_inventory(dataset_dir, class_name="coral")
            reference = load_legacy_segmentation_split_reference(legacy_root)
            prepared = prepare_segmentation_dataset(
                inventory=inventory,
                output_dir=output_dir,
                class_name="coral",
                split_strategy="legacy",
                use_symlinks=False,
                val_split=0.2,
                seed=42,
                legacy_split_reference=reference,
            )

            self.assertEqual(prepared.train_count, 2)
            self.assertEqual(prepared.val_count, 1)
            self.assertEqual(prepared.unassigned_count, 0)
            self.assertTrue((output_dir / "data.yaml").exists())
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["train_image_ids"], ["a", "b"])
            self.assertEqual(manifest["val_image_ids"], ["c"])
            self.assertEqual(manifest["legacy_split_summary"]["duplicate_alias_count"], 1)
            prepared_label = (output_dir / "labels" / "train" / "a.txt").read_text(encoding="utf-8").strip()
            self.assertEqual(prepared_label, "0 0.300000 0.400000 0.700000 0.400000 0.700000 0.600000 0.300000 0.600000")


if __name__ == "__main__":
    unittest.main()
