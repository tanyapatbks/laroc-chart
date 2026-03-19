import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from coral_thesis.phases.color_calibration import (
    ChartGridSampler,
    ColorCalibrationPhase,
    LinearColorCorrector,
)


class Phase2CalibrationTests(unittest.TestCase):
    def _make_chart(self) -> np.ndarray:
        rows, cols = 6, 4
        cell_h, cell_w = 40, 40
        chart = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        for row in range(rows):
            for col in range(cols):
                color = np.array([20 + row * 20, 40 + col * 30, 80 + row * 10 + col * 10], dtype=np.uint8)
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                chart[y1:y2, x1:x2] = color
        return chart

    def test_grid_sampler_returns_expected_patch_count(self) -> None:
        chart = self._make_chart()
        sampler = ChartGridSampler(rows=6, cols=4, cell_sample_ratio=0.5)
        patches = sampler.sample(chart)
        self.assertEqual(patches.shape, (24, 3))

    def test_linear_color_corrector_reduces_error(self) -> None:
        reference = self._make_chart()
        source = reference.astype(np.float32)
        source[:, :, 0] *= 1.1
        source[:, :, 1] *= 0.8
        source[:, :, 2] *= 0.7
        source = np.clip(source, 0, 255).astype(np.uint8)

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
            self.assertTrue((output_dir / "baseline_profile.json").exists())


if __name__ == "__main__":
    unittest.main()

