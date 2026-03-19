import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from coral_thesis.phases.feature_extraction import (
    FEATURE_COLUMNS,
    build_feature_extraction_inventory,
    extract_feature_dataset,
)


class Phase4FeatureExtractionTests(unittest.TestCase):
    def _write_image(self, path: Path, color: tuple[int, int, int]) -> None:
        image = np.full((20, 20, 3), color, dtype=np.uint8)
        cv2.imwrite(str(path), image)

    def _write_mask(self, path: Path, foreground: bool) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        if foreground:
            mask[5:15, 5:15] = 255
        cv2.imwrite(str(path), mask)

    def test_inventory_reports_missing_masks_and_orphans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = root / "sources"
            mask_dir = root / "masks"
            source_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            self._write_image(source_dir / "a.png", (10, 20, 30))
            self._write_image(source_dir / "b.png", (40, 50, 60))
            self._write_mask(mask_dir / "a.png", foreground=True)
            self._write_mask(mask_dir / "orphan.png", foreground=True)

            inventory = build_feature_extraction_inventory(source_dir=source_dir, mask_dir=mask_dir)
            summary = inventory.summary()

            self.assertEqual(summary["source_image_count"], 2)
            self.assertEqual(summary["mask_count"], 2)
            self.assertEqual(summary["matched_pair_count"], 1)
            self.assertEqual(summary["missing_mask_count"], 1)
            self.assertEqual(summary["orphan_mask_count"], 1)
            self.assertEqual(summary["missing_mask_image_ids"], ["b"])

    def test_extract_feature_dataset_writes_csv_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = root / "sources"
            mask_dir = root / "masks"
            output_dir = root / "outputs"
            reports_dir = root / "reports"
            source_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)

            self._write_image(source_dir / "a.png", (10, 20, 30))
            self._write_image(source_dir / "b.png", (40, 50, 60))
            self._write_mask(mask_dir / "a.png", foreground=True)
            self._write_mask(mask_dir / "b.png", foreground=False)

            inventory = build_feature_extraction_inventory(source_dir=source_dir, mask_dir=mask_dir)
            extracted = extract_feature_dataset(
                inventory=inventory,
                csv_path=output_dir / "features.csv",
                report_path=reports_dir / "features.json",
            )

            self.assertTrue(extracted.csv_path.exists())
            self.assertTrue(extracted.report_path.exists())
            self.assertEqual(extracted.extracted_count, 1)
            self.assertEqual(extracted.skipped_count, 1)
            self.assertEqual(extracted.feature_columns, FEATURE_COLUMNS)

            frame = pd.read_csv(extracted.csv_path)
            self.assertEqual(list(frame.columns), ["image_id", *FEATURE_COLUMNS])
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["image_id"], "a")

            report = json.loads(extracted.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["extracted_count"], 1)
            self.assertEqual(report["skipped_count"], 1)
            self.assertEqual(report["inventory"]["matched_pair_count"], 2)


if __name__ == "__main__":
    unittest.main()
