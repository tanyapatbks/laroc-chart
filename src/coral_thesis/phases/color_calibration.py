from __future__ import annotations

from collections import Counter
import json
import multiprocessing as mp
import signal
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty
from typing import Callable

import cv2
import numpy as np

from coral_thesis.config import IMAGE_SUFFIXES


@dataclass(frozen=True)
class BaselineProfile:
    baseline_chart_path: str
    patch_rows: int
    patch_cols: int
    cell_sample_ratio: float
    normalized_chart_width: int
    normalized_chart_height: int
    patch_colors_bgr: list[list[float]]

    def to_dict(self) -> dict:
        return asdict(self)


def _calibrate_sample_worker(payload: dict, result_queue: mp.Queue) -> None:
    try:
        phase = ColorCalibrationPhase(
            baseline_chart_path=Path(payload["baseline_chart_path"]),
            baseline_profile_path=Path(payload["baseline_profile_path"]),
            output_dir=Path(payload["output_dir"]),
            patch_rows=int(payload["patch_rows"]),
            patch_cols=int(payload["patch_cols"]),
            cell_sample_ratio=float(payload["cell_sample_ratio"]),
            min_patch_count=int(payload["min_patch_count"]),
            method=str(payload["method"]),
            sample_timeout_seconds=None,
        )
        baseline_profile = BaselineProfile(**payload["baseline_profile"])
        result = phase.calibrate_single(
            source_image_path=Path(payload["source_image_path"]),
            chart_crop_path=Path(payload["chart_crop_path"]),
            output_stem=str(payload["output_stem"]),
            baseline_profile=baseline_profile,
        )
        result_queue.put({"status": "ok", "result": result})
    except Exception as exc:
        result_queue.put(
            {
                "status": "error",
                "error_type": exc.__class__.__name__,
                "message": str(exc),
            }
        )


class ChartCropNormalizer:
    def __init__(self, target_width: int, target_height: int) -> None:
        self.target_width = target_width
        self.target_height = target_height

    @staticmethod
    def _order_points(points: np.ndarray) -> np.ndarray:
        points = points.astype(np.float32)
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).reshape(-1)

        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = points[np.argmin(sums)]
        ordered[2] = points[np.argmax(sums)]
        ordered[1] = points[np.argmin(diffs)]
        ordered[3] = points[np.argmax(diffs)]
        return ordered

    def _warp(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        destination = np.array(
            [
                [0, 0],
                [self.target_width - 1, 0],
                [self.target_width - 1, self.target_height - 1],
                [0, self.target_height - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(self._order_points(corners), destination)
        return cv2.warpPerspective(image, transform, (self.target_width, self.target_height))

    def _detect_corners(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((5, 5), dtype=np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_area = float(height * width)
        best_corners: np.ndarray | None = None
        best_meta = {"strategy": "full_image", "area_ratio": 1.0}
        best_area = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * 0.1:
                continue

            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                if area > best_area:
                    best_area = area
                    best_corners = polygon.reshape(4, 2).astype(np.float32)
                    best_meta = {"strategy": "quadrilateral", "area_ratio": area / image_area}

        if best_corners is not None:
            return best_corners, best_meta

        if contours:
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.float32)
            return box, {"strategy": "min_area_rect", "area_ratio": area / image_area}

        fallback = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )
        return fallback, best_meta

    def normalize(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        corners, meta = self._detect_corners(image)
        normalized = self._warp(image, corners)
        return normalized, meta


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

    def transform_colors(self, colors: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("fit must be called before transform_colors")
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError("expected color arrays shaped as (n, 3)")

        design = np.concatenate(
            [colors.astype(np.float32), np.ones((colors.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        corrected = design @ self.coefficients
        return np.clip(corrected, 0, 255).astype(np.float32)

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("fit must be called before transform_image")

        flat = image.reshape(-1, 3).astype(np.float32)
        corrected = self.transform_colors(flat).astype(np.uint8)
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
        sample_timeout_seconds: int | None = None,
    ) -> None:
        self.baseline_chart_path = baseline_chart_path
        self.baseline_profile_path = baseline_profile_path
        self.output_dir = output_dir
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols
        self.cell_sample_ratio = cell_sample_ratio
        self.min_patch_count = min_patch_count
        self.method = method
        self.sample_timeout_seconds = sample_timeout_seconds
        self.grid_sampler = ChartGridSampler(
            rows=patch_rows,
            cols=patch_cols,
            cell_sample_ratio=cell_sample_ratio,
        )

    def _read_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"unable to read image: {path}")
        return image

    def _run_with_timeout(self, operation: str, callback: Callable[[], dict]) -> dict:
        if self.sample_timeout_seconds is None or not hasattr(signal, "SIGALRM"):
            return callback()

        def _handle_timeout(signum, frame) -> None:
            raise TimeoutError(
                f"{operation} timed out after {self.sample_timeout_seconds} seconds"
            )

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(self.sample_timeout_seconds)
        try:
            return callback()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)

    def build_baseline_profile(self) -> BaselineProfile:
        baseline_chart = self._read_image(self.baseline_chart_path)
        patch_colors = self.grid_sampler.sample(baseline_chart)
        if len(patch_colors) < self.min_patch_count:
            raise ValueError(
                f"baseline chart produced too few patches ({len(patch_colors)} < {self.min_patch_count})"
            )

        return BaselineProfile(
            baseline_chart_path=str(self.baseline_chart_path),
            patch_rows=self.patch_rows,
            patch_cols=self.patch_cols,
            cell_sample_ratio=self.cell_sample_ratio,
            normalized_chart_width=int(baseline_chart.shape[1]),
            normalized_chart_height=int(baseline_chart.shape[0]),
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
        if "normalized_chart_width" not in raw or "normalized_chart_height" not in raw:
            baseline_chart = self._read_image(self.baseline_chart_path)
            raw["normalized_chart_width"] = int(baseline_chart.shape[1])
            raw["normalized_chart_height"] = int(baseline_chart.shape[0])
            self.baseline_profile_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        return BaselineProfile(**raw)

    def _normalizer_for_profile(self, baseline_profile: BaselineProfile) -> ChartCropNormalizer:
        return ChartCropNormalizer(
            target_width=baseline_profile.normalized_chart_width,
            target_height=baseline_profile.normalized_chart_height,
        )

    def _align_patch_colors(
        self,
        source_patch_colors: np.ndarray,
        baseline_profile: BaselineProfile,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_patch_colors = np.array(baseline_profile.patch_colors_bgr, dtype=np.float32)

        if len(source_patch_colors) != len(target_patch_colors):
            patch_count = min(len(source_patch_colors), len(target_patch_colors))
            source_patch_colors = source_patch_colors[:patch_count]
            target_patch_colors = target_patch_colors[:patch_count]

        if len(source_patch_colors) < self.min_patch_count:
            raise ValueError(
                f"chart crop produced too few usable patches ({len(source_patch_colors)} < {self.min_patch_count})"
            )

        return source_patch_colors.astype(np.float32), target_patch_colors.astype(np.float32)

    def _build_corrector(
        self,
        source_patch_colors: np.ndarray,
        target_patch_colors: np.ndarray,
    ) -> LinearColorCorrector:
        corrector = LinearColorCorrector()
        corrector.fit(source_patch_colors, target_patch_colors)
        return corrector

    @staticmethod
    def _error_summary(candidate_colors: np.ndarray, target_colors: np.ndarray) -> dict:
        differences = target_colors.astype(np.float32) - candidate_colors.astype(np.float32)
        absolute = np.abs(differences)
        channel_mae = absolute.mean(axis=0)
        return {
            "mae": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(differences ** 2))),
            "max_abs_error": float(absolute.max()),
            "channel_mae_bgr": {
                "b": float(channel_mae[0]),
                "g": float(channel_mae[1]),
                "r": float(channel_mae[2]),
            },
        }

    def _quality_metrics(
        self,
        source_patch_colors: np.ndarray,
        corrected_patch_colors: np.ndarray,
        target_patch_colors: np.ndarray,
    ) -> dict:
        before = self._error_summary(source_patch_colors, target_patch_colors)
        after = self._error_summary(corrected_patch_colors, target_patch_colors)
        mae_delta = before["mae"] - after["mae"]
        rmse_delta = before["rmse"] - after["rmse"]
        mae_improvement_ratio = mae_delta / before["mae"] if before["mae"] > 0 else 0.0
        return {
            "patch_count_evaluated": int(len(target_patch_colors)),
            "before": before,
            "after": after,
            "mae_delta": float(mae_delta),
            "rmse_delta": float(rmse_delta),
            "mae_improvement_ratio": float(mae_improvement_ratio),
            "improved": bool(after["mae"] < before["mae"]),
        }

    @staticmethod
    def _classify_failure(reason: str, source_image_path: Path | None, chart_crop_path: Path) -> str:
        if "source image not found" in reason:
            return "missing_source_image"
        if "timed out after" in reason:
            return "sample_timeout"
        if "unable to read image" in reason:
            if source_image_path is not None and str(source_image_path) in reason:
                return "unreadable_source_image"
            if str(chart_crop_path) in reason:
                return "unreadable_chart_crop"
            return "unreadable_image"
        if "too few usable patches" in reason or "too few patches" in reason:
            return "insufficient_chart_patches"
        return "calibration_failed"

    def _failure_record(
        self,
        image_id: str,
        chart_crop_path: Path,
        reason: str,
        source_image_path: Path | None = None,
    ) -> dict:
        record = {
            "image_id": image_id,
            "chart_crop_path": str(chart_crop_path),
            "reason": reason,
            "failure_type": self._classify_failure(
                reason=reason,
                source_image_path=source_image_path,
                chart_crop_path=chart_crop_path,
            ),
        }
        if source_image_path is not None:
            record["source_image_path"] = str(source_image_path)
        return record

    @staticmethod
    def _batch_quality_summary(successes: list[dict]) -> dict:
        if not successes:
            return {
                "sample_count": 0,
                "improved_count": 0,
                "improved_ratio": 0.0,
                "mean_before_mae": None,
                "mean_after_mae": None,
                "mean_mae_delta": None,
                "mean_before_rmse": None,
                "mean_after_rmse": None,
                "mean_rmse_delta": None,
            }

        before_mae = np.array([item["quality_metrics"]["before"]["mae"] for item in successes], dtype=np.float32)
        after_mae = np.array([item["quality_metrics"]["after"]["mae"] for item in successes], dtype=np.float32)
        before_rmse = np.array([item["quality_metrics"]["before"]["rmse"] for item in successes], dtype=np.float32)
        after_rmse = np.array([item["quality_metrics"]["after"]["rmse"] for item in successes], dtype=np.float32)
        improved_count = sum(1 for item in successes if item["quality_metrics"]["improved"])

        return {
            "sample_count": len(successes),
            "improved_count": improved_count,
            "improved_ratio": float(improved_count / len(successes)),
            "mean_before_mae": float(before_mae.mean()),
            "mean_after_mae": float(after_mae.mean()),
            "mean_mae_delta": float((before_mae - after_mae).mean()),
            "mean_before_rmse": float(before_rmse.mean()),
            "mean_after_rmse": float(after_rmse.mean()),
            "mean_rmse_delta": float((before_rmse - after_rmse).mean()),
        }

    @staticmethod
    def _batch_metrics(successes: list[dict], failures: list[dict]) -> tuple[dict, list[dict]]:
        strategy_counts = Counter(item["normalization_strategy"] for item in successes)
        failure_counts = Counter(item["failure_type"] for item in failures)
        unstable_failures = [
            failure
            for failure in failures
            if failure["failure_type"] in {
                "sample_timeout",
                "unreadable_source_image",
                "unreadable_chart_crop",
                "unreadable_image",
            }
        ]
        metrics = {
            "quality": ColorCalibrationPhase._batch_quality_summary(successes),
            "normalization_strategy_counts": dict(strategy_counts),
            "failure_type_counts": dict(failure_counts),
            "unstable_input_count": len(unstable_failures),
        }
        return metrics, unstable_failures

    def _resolve_source_image(self, source_dir: Path, image_id: str) -> Path | None:
        for candidate in sorted(source_dir.iterdir()):
            if candidate.is_file() and candidate.stem == image_id and candidate.suffix in IMAGE_SUFFIXES:
                return candidate
        return None

    def _batch_sample_payload(
        self,
        source_image_path: Path,
        chart_crop_path: Path,
        output_stem: str,
        baseline_profile: BaselineProfile,
    ) -> dict:
        return {
            "baseline_chart_path": str(self.baseline_chart_path),
            "baseline_profile_path": str(self.baseline_profile_path),
            "output_dir": str(self.output_dir),
            "patch_rows": self.patch_rows,
            "patch_cols": self.patch_cols,
            "cell_sample_ratio": self.cell_sample_ratio,
            "min_patch_count": self.min_patch_count,
            "method": self.method,
            "baseline_profile": baseline_profile.to_dict(),
            "source_image_path": str(source_image_path),
            "chart_crop_path": str(chart_crop_path),
            "output_stem": output_stem,
        }

    def _calibrate_batch_sample(
        self,
        source_image_path: Path,
        chart_crop_path: Path,
        output_stem: str,
        baseline_profile: BaselineProfile,
    ) -> dict:
        if self.sample_timeout_seconds is None:
            return self.calibrate_single(
                source_image_path=source_image_path,
                chart_crop_path=chart_crop_path,
                output_stem=output_stem,
                baseline_profile=baseline_profile,
            )

        context = mp.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(
            target=_calibrate_sample_worker,
            args=(
                self._batch_sample_payload(
                    source_image_path=source_image_path,
                    chart_crop_path=chart_crop_path,
                    output_stem=output_stem,
                    baseline_profile=baseline_profile,
                ),
                result_queue,
            ),
        )
        process.start()
        process.join(self.sample_timeout_seconds)

        try:
            if process.is_alive():
                process.terminate()
                process.join()
                raise TimeoutError(
                    f"Phase 2 calibration for {chart_crop_path} timed out after {self.sample_timeout_seconds} seconds"
                )

            try:
                outcome = result_queue.get(timeout=1)
            except Empty:
                raise RuntimeError(
                    f"Phase 2 worker exited without a result for {chart_crop_path} (exit_code={process.exitcode})"
                )

            if outcome["status"] == "ok":
                return outcome["result"]

            message = outcome["message"]
            error_type = outcome["error_type"]
            if error_type == "TimeoutError":
                raise TimeoutError(message)
            if error_type == "ValueError":
                raise ValueError(message)
            raise RuntimeError(message)
        finally:
            result_queue.close()
            result_queue.join_thread()

    def _normalize_chart_crop(
        self,
        chart_crop: np.ndarray,
        baseline_profile: BaselineProfile,
    ) -> tuple[np.ndarray, dict]:
        normalizer = self._normalizer_for_profile(baseline_profile)
        return normalizer.normalize(chart_crop)

    def calibrate_single(
        self,
        source_image_path: Path,
        chart_crop_path: Path,
        output_stem: str,
        baseline_profile: BaselineProfile | None = None,
    ) -> dict:
        def _calibrate() -> dict:
            resolved_profile = baseline_profile or self.load_baseline_profile()
            chart_crop = self._read_image(chart_crop_path)
            source_image = self._read_image(source_image_path)
            normalized_chart, normalization_meta = self._normalize_chart_crop(
                chart_crop=chart_crop,
                baseline_profile=resolved_profile,
            )
            source_patch_colors = self.grid_sampler.sample(normalized_chart)
            aligned_source_patch_colors, target_patch_colors = self._align_patch_colors(
                source_patch_colors=source_patch_colors,
                baseline_profile=resolved_profile,
            )
            corrector = self._build_corrector(aligned_source_patch_colors, target_patch_colors)
            corrected_patch_colors = corrector.transform_colors(aligned_source_patch_colors)
            quality_metrics = self._quality_metrics(
                source_patch_colors=aligned_source_patch_colors,
                corrected_patch_colors=corrected_patch_colors,
                target_patch_colors=target_patch_colors,
            )
            calibrated_image = corrector.transform_image(source_image)

            self.output_dir.mkdir(parents=True, exist_ok=True)
            calibrated_dir = self.output_dir / "calibrated_images"
            normalized_dir = self.output_dir / "normalized_charts"
            calibrated_dir.mkdir(parents=True, exist_ok=True)
            normalized_dir.mkdir(parents=True, exist_ok=True)
            output_path = calibrated_dir / f"{output_stem}_calibrated.jpg"
            normalized_chart_path = normalized_dir / f"{output_stem}_normalized_chart.jpg"
            cv2.imwrite(str(output_path), calibrated_image)
            cv2.imwrite(str(normalized_chart_path), normalized_chart)

            return {
                "source_image_path": str(source_image_path),
                "chart_crop_path": str(chart_crop_path),
                "output_path": str(output_path),
                "normalized_chart_path": str(normalized_chart_path),
                "patch_count": int(len(aligned_source_patch_colors)),
                "normalization_strategy": normalization_meta["strategy"],
                "normalization_area_ratio": float(normalization_meta["area_ratio"]),
                "method": self.method,
                "quality_metrics": quality_metrics,
            }

        return self._run_with_timeout(
            operation=f"Phase 2 calibration for {chart_crop_path}",
            callback=_calibrate,
        )

    def calibrate_batch(
        self,
        source_dir: Path,
        crops_dir: Path,
        report_name: str = "batch_report",
        crop_glob: str = "*.jpg",
        limit: int | None = None,
    ) -> dict:
        baseline_profile = self.load_baseline_profile()
        crop_paths = sorted(crops_dir.glob(crop_glob))
        if limit is not None:
            crop_paths = crop_paths[:limit]

        successes: list[dict] = []
        failures: list[dict] = []

        for crop_path in crop_paths:
            image_id = crop_path.stem.split("_chart_")[0]
            source_image_path = self._resolve_source_image(source_dir, image_id)
            if source_image_path is None:
                failures.append(
                    self._failure_record(
                        image_id=image_id,
                        chart_crop_path=crop_path,
                        reason=f"source image not found for image_id={image_id}",
                    )
                )
                continue

            try:
                result = self._calibrate_batch_sample(
                    source_image_path=source_image_path,
                    chart_crop_path=crop_path,
                    output_stem=crop_path.stem,
                    baseline_profile=baseline_profile,
                )
                successes.append(result)
            except Exception as exc:
                failures.append(
                    self._failure_record(
                        image_id=image_id,
                        chart_crop_path=crop_path,
                        source_image_path=source_image_path,
                        reason=str(exc),
                    )
                )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / f"{report_name}.json"
        metrics, unstable_inputs = self._batch_metrics(successes, failures)
        report = {
            "source_dir": str(source_dir),
            "crops_dir": str(crops_dir),
            "crop_glob": crop_glob,
            "attempted_count": len(crop_paths),
            "success_count": len(successes),
            "failure_count": len(failures),
            "metrics": metrics,
            "unstable_inputs": unstable_inputs,
            "successes": successes,
            "failures": failures,
            "report_path": str(report_path),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
