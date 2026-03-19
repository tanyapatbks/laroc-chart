import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from coral_thesis.phases.feature_extraction import FEATURE_COLUMNS
from coral_thesis.phases.health_estimation import (
    HealthEstimationPhase,
    build_coralwatch_reference,
    export_phase5_label_template,
    health_level_to_score,
    normalize_phase5_labels,
    train_health_models,
)


class Phase5HealthEstimationTests(unittest.TestCase):
    def _write_synthetic_chart(self, path: Path) -> Path:
        cell_size = 10
        chart = np.full((cell_size * 7, cell_size * 7, 3), 255, dtype=np.uint8)

        def color_for(group: str, level: int) -> tuple[int, int, int]:
            group_index = {"B": 0, "C": 1, "D": 2, "E": 3}[group]
            return (
                30 + (group_index * 40) + level,
                60 + (group_index * 30) + (level * 2),
                90 + (group_index * 20) + (level * 3),
            )

        def paint_cell(row: int, col: int, color: tuple[int, int, int]) -> None:
            y1, y2 = row * cell_size, (row + 1) * cell_size
            x1, x2 = col * cell_size, (col + 1) * cell_size
            chart[y1:y2, x1:x2] = color
            chart[y1 + 3:y1 + 7, x1 + 3:x1 + 7] = 0

        for level in range(1, 7):
            paint_cell(0, level - 1, color_for("B", level))
            paint_cell(level - 1, 6, color_for("C", level))
            paint_cell(6, 7 - level, color_for("D", level))
            paint_cell(7 - level, 0, color_for("E", level))

        cv2.imwrite(str(path), chart)
        return path

    def _feature_row(self, image_id: str, base: float, hue_index: int, health_score: float) -> dict[str, float | str]:
        return {
            "image_id": image_id,
            "R_mean": base + 1.0,
            "G_mean": base + 2.0,
            "B_mean": base + 3.0,
            "R_std": 1.0 + health_score,
            "G_std": 2.0 + health_score,
            "B_std": 3.0 + health_score,
            "H_mean": float(hue_index * 40),
            "S_mean": 100.0 + base,
            "V_mean": 120.0 + base,
            "L_mean": 40.0 + (health_score * 100.0),
            "a_mean": float(hue_index * 20),
            "b_mean": 80.0 - (health_score * 10.0),
            "R_p10": base,
            "R_p90": base + 10.0,
        }

    def test_normalize_phase5_labels_from_category(self) -> None:
        labels = pd.DataFrame(
            {
                "image_id": ["img_a", "img_b"],
                "category": ["B2", "E6"],
            }
        )

        normalized, source_format = normalize_phase5_labels(labels)

        self.assertEqual(source_format, "category")
        self.assertEqual(list(normalized["hue_group"]), ["B", "E"])
        self.assertEqual(list(normalized["health_level"]), [2, 6])
        self.assertAlmostEqual(normalized.iloc[0]["health_score"], health_level_to_score(2))
        self.assertAlmostEqual(normalized.iloc[1]["health_score"], health_level_to_score(6))

    def test_export_phase5_label_template_writes_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            features_csv = root / "features.csv"
            template_csv = root / "labels_template.csv"
            pd.DataFrame(
                [
                    self._feature_row("a", base=10.0, hue_index=0, health_score=health_level_to_score(1)),
                    self._feature_row("b", base=20.0, hue_index=1, health_score=health_level_to_score(2)),
                ]
            ).to_csv(features_csv, index=False)

            written_path = export_phase5_label_template(features_csv, template_csv)

            self.assertEqual(written_path, template_csv.resolve())
            template = pd.read_csv(template_csv)
            self.assertEqual(
                list(template.columns),
                ["image_id", "category", "hue_group", "health_level", "health_score", "notes"],
            )
            self.assertEqual(list(template["image_id"]), ["a", "b"])

    def test_build_coralwatch_reference_samples_expected_categories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chart_path = self._write_synthetic_chart(Path(tmp_dir) / "baseline_chart.png")

            reference = build_coralwatch_reference(chart_path)

            self.assertEqual(len(reference.patches), 24)
            categories = {patch.category for patch in reference.patches}
            self.assertIn("B1", categories)
            self.assertIn("C6", categories)
            self.assertIn("D4", categories)
            self.assertIn("E2", categories)

    def test_health_estimation_phase_heuristic_returns_nearest_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chart_path = self._write_synthetic_chart(Path(tmp_dir) / "baseline_chart.png")
            reference = build_coralwatch_reference(chart_path)
            target_patch = next(patch for patch in reference.patches if patch.category == "C4")

            features = {
                column: 0.0
                for column in FEATURE_COLUMNS
            }
            features["L_mean"] = target_patch.mean_lab[0]
            features["a_mean"] = target_patch.mean_lab[1]
            features["b_mean"] = target_patch.mean_lab[2]
            phase = HealthEstimationPhase(
                hue_model_path=None,
                health_model_path=None,
                baseline_chart_path=chart_path,
                strategy="heuristic",
                reference=reference,
            )

            result = phase.run(type("Feature", (), {"image_id": "sample", "values": features})())

            self.assertEqual(result.hue_group, "C")
            self.assertEqual(result.category, "C4")
            self.assertAlmostEqual(result.health_score, health_level_to_score(4))

    def test_train_health_models_writes_models_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            chart_path = self._write_synthetic_chart(root / "baseline_chart.png")
            features_csv = root / "features.csv"
            labels_csv = root / "labels.csv"
            hue_model_path = root / "models" / "hue_model.joblib"
            health_model_path = root / "models" / "health_model.joblib"
            report_path = root / "reports" / "phase5_training.json"

            feature_rows: list[dict[str, float | str]] = []
            label_rows: list[dict[str, str]] = []
            for repeat_index in range(2):
                for hue_index, hue_group in enumerate(("B", "C", "D", "E")):
                    for level in range(1, 7):
                        image_id = f"{hue_group}{level}_{repeat_index}"
                        health_score = health_level_to_score(level)
                        feature_rows.append(
                            self._feature_row(
                                image_id=image_id,
                                base=float((hue_index * 100) + (level * 10) + repeat_index),
                                hue_index=hue_index,
                                health_score=health_score,
                            )
                        )
                        label_rows.append({"image_id": image_id, "category": f"{hue_group}{level}"})

            pd.DataFrame(feature_rows).to_csv(features_csv, index=False)
            pd.DataFrame(label_rows).to_csv(labels_csv, index=False)

            trained = train_health_models(
                features_csv_path=features_csv,
                labels_csv_path=labels_csv,
                baseline_chart_path=chart_path,
                hue_model_path=hue_model_path,
                health_model_path=health_model_path,
                report_path=report_path,
                seed=42,
                validation_split=0.25,
                classifier_estimators=20,
                regressor_estimators=20,
                max_depth=None,
                min_samples_leaf=1,
            )

            self.assertTrue(trained.hue_model_path.exists())
            self.assertTrue(trained.health_model_path.exists())
            self.assertTrue(trained.report_path.exists())
            report = json.loads(trained.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["train_row_count"], trained.train_row_count)
            self.assertEqual(report["validation_row_count"], trained.validation_row_count)
            self.assertIn("model", report["validation_metrics"])
            self.assertIn("heuristic_baseline", report["validation_metrics"])


if __name__ == "__main__":
    unittest.main()
