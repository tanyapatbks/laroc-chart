from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class BaselineProfile:
    baseline_chart_path: str
    patch_rows: int
    patch_cols: int
    cell_sample_ratio: float
    patch_colors_bgr: list[list[float]]

    def to_dict(self) -> dict:
        return asdict(self)


class ChartGridSampler:
    def __init__(self, rows: int, cols: int, cell_sample_ratio: float) -> None:
        self.rows = rows
        self.cols = cols
        self.cell_sample_ratio = cell_sample_ratio

    def sample(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("image is required")

        height, width = image.shape[:2]
        row_edges = np.linspace(0, height, self.rows + 1, dtype=int)
        col_edges = np.linspace(0, width, self.cols + 1, dtype=int)
        patch_colors: list[np.ndarray] = []

        for row in range(self.rows):
            for col in range(self.cols):
                y1, y2 = row_edges[row], row_edges[row + 1]
                x1, x2 = col_edges[col], col_edges[col + 1]
                patch = image[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                patch_height, patch_width = patch.shape[:2]
                sample_h = max(1, int(patch_height * self.cell_sample_ratio))
                sample_w = max(1, int(patch_width * self.cell_sample_ratio))
                y_start = (patch_height - sample_h) // 2
                x_start = (patch_width - sample_w) // 2
                center_patch = patch[y_start:y_start + sample_h, x_start:x_start + sample_w]
                patch_colors.append(center_patch.reshape(-1, 3).mean(axis=0))

        if not patch_colors:
            raise ValueError("no chart patches were sampled")

        return np.vstack(patch_colors).astype(np.float32)


class LinearColorCorrector:
    def __init__(self) -> None:
        self.coefficients: np.ndarray | None = None

    def fit(self, source_colors: np.ndarray, target_colors: np.ndarray) -> None:
        if source_colors.shape != target_colors.shape:
            raise ValueError("source_colors and target_colors must have the same shape")
        if source_colors.ndim != 2 or source_colors.shape[1] != 3:
            raise ValueError("expected color arrays shaped as (n, 3)")

        design = np.concatenate(
            [source_colors.astype(np.float32), np.ones((source_colors.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, target_colors.astype(np.float32), rcond=None)
        self.coefficients = coefficients

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("fit must be called before transform_image")

        flat = image.reshape(-1, 3).astype(np.float32)
        design = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float32)], axis=1)
        corrected = design @ self.coefficients
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        return corrected.reshape(image.shape)


class ColorCalibrationPhase:
    phase_name = "color-calibration"

    def __init__(
        self,
        baseline_chart_path: Path,
        baseline_profile_path: Path,
        output_dir: Path,
        patch_rows: int,
        patch_cols: int,
        cell_sample_ratio: float,
        min_patch_count: int,
        method: str = "linear",
    ) -> None:
        self.baseline_chart_path = baseline_chart_path
        self.baseline_profile_path = baseline_profile_path
        self.output_dir = output_dir
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols
        self.cell_sample_ratio = cell_sample_ratio
        self.min_patch_count = min_patch_count
        self.method = method
        self.sampler = ChartGridSampler(
            rows=patch_rows,
            cols=patch_cols,
            cell_sample_ratio=cell_sample_ratio,
        )

    def _read_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"unable to read image: {path}")
        return image

    def build_baseline_profile(self) -> BaselineProfile:
        baseline_chart = self._read_image(self.baseline_chart_path)
        patch_colors = self.sampler.sample(baseline_chart)
        if len(patch_colors) < self.min_patch_count:
            raise ValueError(
                f"baseline chart produced too few patches ({len(patch_colors)} < {self.min_patch_count})"
            )

        return BaselineProfile(
            baseline_chart_path=str(self.baseline_chart_path),
            patch_rows=self.patch_rows,
            patch_cols=self.patch_cols,
            cell_sample_ratio=self.cell_sample_ratio,
            patch_colors_bgr=patch_colors.tolist(),
        )

    def build_and_save_baseline_profile(self) -> BaselineProfile:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        profile = self.build_baseline_profile()
        self.baseline_profile_path.write_text(
            json.dumps(profile.to_dict(), indent=2),
            encoding="utf-8",
        )
        return profile

    def load_baseline_profile(self) -> BaselineProfile:
        if not self.baseline_profile_path.exists():
            return self.build_and_save_baseline_profile()

        raw = json.loads(self.baseline_profile_path.read_text(encoding="utf-8"))
        return BaselineProfile(**raw)

    def _build_corrector(self, source_patch_colors: np.ndarray, baseline_profile: BaselineProfile) -> LinearColorCorrector:
        target_patch_colors = np.array(baseline_profile.patch_colors_bgr, dtype=np.float32)

        if len(source_patch_colors) != len(target_patch_colors):
            patch_count = min(len(source_patch_colors), len(target_patch_colors))
            source_patch_colors = source_patch_colors[:patch_count]
            target_patch_colors = target_patch_colors[:patch_count]

        if len(source_patch_colors) < self.min_patch_count:
            raise ValueError(
                f"chart crop produced too few usable patches ({len(source_patch_colors)} < {self.min_patch_count})"
            )

        corrector = LinearColorCorrector()
        corrector.fit(source_patch_colors, target_patch_colors)
        return corrector

    def calibrate_single(
        self,
        source_image_path: Path,
        chart_crop_path: Path,
        output_stem: str,
    ) -> dict:
        baseline_profile = self.load_baseline_profile()
        chart_crop = self._read_image(chart_crop_path)
        source_image = self._read_image(source_image_path)
        source_patch_colors = self.sampler.sample(chart_crop)
        corrector = self._build_corrector(source_patch_colors, baseline_profile)
        calibrated_image = corrector.transform_image(source_image)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{output_stem}_calibrated.jpg"
        cv2.imwrite(str(output_path), calibrated_image)

        return {
            "source_image_path": str(source_image_path),
            "chart_crop_path": str(chart_crop_path),
            "output_path": str(output_path),
            "patch_count": int(len(source_patch_colors)),
            "method": self.method,
        }
