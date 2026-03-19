import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from coral_thesis.phases.color_calibration import (
    ChartCropNormalizer,
    ChartGridSampler,
    ColorCalibrationPhase,
    LinearColorCorrector,
)


class Phase2CalibrationTests(unittest.TestCase):
    def _make_chart(self) -> np.ndarray:
        rows, cols = 6, 4
        cell_h, cell_w = 40, 60
        chart = np.full((rows * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)
        for row in range(rows):
            for col in range(cols):
                color = np.array([20 + row * 20, 40 + col * 30, 80 + row * 10 + col * 10], dtype=np.uint8)
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                chart[y1:y2, x1:x2] = color
        cv2.rectangle(chart, (2, 2), (chart.shape[1] - 3, chart.shape[0] - 3), (0, 0, 0), 4)
        return chart

    def _apply_cast(self, image: np.ndarray) -> np.ndarray:
        cast = image.astype(np.float32)
        cast[:, :, 0] *= 1.1
        cast[:, :, 1] *= 0.8
        cast[:, :, 2] *= 0.7
        return np.clip(cast, 0, 255).astype(np.uint8)

    def _perspective_crop(self, chart: np.ndarray) -> np.ndarray:
        src = np.array(
            [
                [0, 0],
                [chart.shape[1] - 1, 0],
                [chart.shape[1] - 1, chart.shape[0] - 1],
                [0, chart.shape[0] - 1],
            ],
            dtype=np.float32,
        )
        dst = np.array(
            [
                [50, 30],
                [290, 20],
                [270, 310],
                [30, 300],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(chart, transform, (320, 340))

    def test_grid_sampler_returns_expected_patch_count(self) -> None:
        chart = self._make_chart()
        sampler = ChartGridSampler(rows=6, cols=4, cell_sample_ratio=0.5)
        patches = sampler.sample(chart)
        self.assertEqual(patches.shape, (24, 3))

    def test_chart_crop_normalizer_reduces_patch_error(self) -> None:
        reference = self._make_chart()
        warped = self._perspective_crop(reference)
        normalizer = ChartCropNormalizer(
            target_width=reference.shape[1],
            target_height=reference.shape[0],
        )
        normalized, metadata = normalizer.normalize(warped)

        sampler = ChartGridSampler(rows=6, cols=4, cell_sample_ratio=0.5)
        reference_patches = sampler.sample(reference)
        normalized_patches = sampler.sample(normalized)
        naive_patches = sampler.sample(cv2.resize(warped, (reference.shape[1], reference.shape[0])))

        normalized_error = np.mean(np.abs(reference_patches - normalized_patches))
        naive_error = np.mean(np.abs(reference_patches - naive_patches))

        self.assertEqual(normalized.shape[:2], reference.shape[:2])
        self.assertLess(normalized_error, naive_error)
        self.assertIn(metadata["strategy"], {"quadrilateral", "min_area_rect", "full_image"})

    def test_linear_color_corrector_reduces_error(self) -> None:
        reference = self._make_chart()
        source = self._apply_cast(reference)

        sampler = ChartGridSampler(rows=6, cols=4, cell_sample_ratio=0.5)
        reference_patches = sampler.sample(reference)
        source_patches = sampler.sample(source)

        corrector = LinearColorCorrector()
        corrector.fit(source_patches, reference_patches)
        corrected = corrector.transform_image(source)

        original_error = np.mean((reference.astype(np.float32) - source.astype(np.float32)) ** 2)
        corrected_error = np.mean((reference.astype(np.float32) - corrected.astype(np.float32)) ** 2)
        self.assertLess(corrected_error, original_error)

    def test_phase2_builds_baseline_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline_chart_path = root / "baseline.png"
            output_dir = root / "phase2"
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(baseline_chart_path), self._make_chart())

            phase = ColorCalibrationPhase(
                baseline_chart_path=baseline_chart_path,
                baseline_profile_path=output_dir / "baseline_profile.json",
                output_dir=output_dir,
                patch_rows=6,
                patch_cols=4,
                cell_sample_ratio=0.5,
                min_patch_count=8,
                method="linear",
            )
            profile = phase.build_and_save_baseline_profile()

            self.assertEqual(profile.patch_rows, 6)
            self.assertEqual(profile.patch_cols, 4)
            self.assertEqual(profile.normalized_chart_width, self._make_chart().shape[1])
            self.assertEqual(profile.normalized_chart_height, self._make_chart().shape[0])
            self.assertTrue((output_dir / "baseline_profile.json").exists())

    def test_phase2_calibrate_single_reduces_error_with_perspective_crop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline = self._make_chart()
            source = self._apply_cast(baseline)
            crop = self._perspective_crop(source)

            baseline_chart_path = root / "baseline.png"
            source_image_path = root / "source.png"
            crop_path = root / "crop.png"
            output_dir = root / "phase2"
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(baseline_chart_path), baseline)
            cv2.imwrite(str(source_image_path), source)
            cv2.imwrite(str(crop_path), crop)

            phase = ColorCalibrationPhase(
                baseline_chart_path=baseline_chart_path,
                baseline_profile_path=output_dir / "baseline_profile.json",
                output_dir=output_dir,
                patch_rows=6,
                patch_cols=4,
                cell_sample_ratio=0.5,
                min_patch_count=8,
                method="linear",
            )
            phase.build_and_save_baseline_profile()
            result = phase.calibrate_single(
                source_image_path=source_image_path,
                chart_crop_path=crop_path,
                output_stem="sample",
            )

            corrected = cv2.imread(result["output_path"])
            self.assertIsNotNone(corrected)
            self.assertTrue(Path(result["normalized_chart_path"]).exists())
            self.assertTrue(result["quality_metrics"]["improved"])
            self.assertLess(
                result["quality_metrics"]["after"]["mae"],
                result["quality_metrics"]["before"]["mae"],
            )
            self.assertLess(
                np.mean((baseline.astype(np.float32) - corrected.astype(np.float32)) ** 2),
                np.mean((baseline.astype(np.float32) - source.astype(np.float32)) ** 2),
            )

    def test_phase2_batch_calibration_generates_report_with_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = root / "sources"
            crops_dir = root / "crops"
            output_dir = root / "phase2"
            source_dir.mkdir(parents=True, exist_ok=True)
            crops_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            baseline = self._make_chart()
            baseline_chart_path = root / "baseline.png"
            cv2.imwrite(str(baseline_chart_path), baseline)

            for image_id in ("a", "b"):
                source = self._apply_cast(baseline)
                crop = self._perspective_crop(source)
                cv2.imwrite(str(source_dir / f"{image_id}.png"), source)
                cv2.imwrite(str(crops_dir / f"{image_id}_chart_0.jpg"), crop)

            cv2.imwrite(str(source_dir / "bad.png"), self._apply_cast(baseline))
            (crops_dir / "bad_chart_0.jpg").write_text("not-an-image", encoding="utf-8")

            phase = ColorCalibrationPhase(
                baseline_chart_path=baseline_chart_path,
                baseline_profile_path=output_dir / "baseline_profile.json",
                output_dir=output_dir,
                patch_rows=6,
                patch_cols=4,
                cell_sample_ratio=0.5,
                min_patch_count=8,
                method="linear",
            )
            phase.build_and_save_baseline_profile()
            report = phase.calibrate_batch(
                source_dir=source_dir,
                crops_dir=crops_dir,
                report_name="batch_report",
            )

            self.assertEqual(report["success_count"], 2)
            self.assertEqual(report["failure_count"], 1)
            self.assertEqual(report["metrics"]["quality"]["sample_count"], 2)
            self.assertEqual(report["metrics"]["quality"]["improved_count"], 2)
            self.assertLess(
                report["metrics"]["quality"]["mean_after_mae"],
                report["metrics"]["quality"]["mean_before_mae"],
            )
            self.assertEqual(report["metrics"]["failure_type_counts"]["unreadable_chart_crop"], 1)
            self.assertEqual(report["metrics"]["unstable_input_count"], 1)
            self.assertEqual(len(report["unstable_inputs"]), 1)
            self.assertTrue(Path(report["report_path"]).exists())
            saved_report = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(saved_report["attempted_count"], 3)
            self.assertEqual(saved_report["failures"][0]["failure_type"], "unreadable_chart_crop")

    def test_phase2_batch_classifies_timeout_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = root / "sources"
            crops_dir = root / "crops"
            output_dir = root / "phase2"
            source_dir.mkdir(parents=True, exist_ok=True)
            crops_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            baseline = self._make_chart()
            baseline_chart_path = root / "baseline.png"
            cv2.imwrite(str(baseline_chart_path), baseline)
            cv2.imwrite(str(source_dir / "a.png"), self._apply_cast(baseline))
            cv2.imwrite(str(crops_dir / "a_chart_0.jpg"), self._perspective_crop(baseline))

            phase = ColorCalibrationPhase(
                baseline_chart_path=baseline_chart_path,
                baseline_profile_path=output_dir / "baseline_profile.json",
                output_dir=output_dir,
                patch_rows=6,
                patch_cols=4,
                cell_sample_ratio=0.5,
                min_patch_count=8,
                method="linear",
                sample_timeout_seconds=1,
            )
            phase.build_and_save_baseline_profile()

            with mock.patch.object(
                phase,
                "_calibrate_batch_sample",
                side_effect=TimeoutError("Phase 2 calibration for a_chart_0.jpg timed out after 1 seconds"),
            ):
                report = phase.calibrate_batch(
                    source_dir=source_dir,
                    crops_dir=crops_dir,
                    report_name="timeout_report",
                )

            self.assertEqual(report["success_count"], 0)
            self.assertEqual(report["failure_count"], 1)
            self.assertEqual(report["metrics"]["failure_type_counts"]["sample_timeout"], 1)
            self.assertEqual(report["metrics"]["unstable_input_count"], 1)
            self.assertEqual(report["unstable_inputs"][0]["failure_type"], "sample_timeout")


if __name__ == "__main__":
    unittest.main()
