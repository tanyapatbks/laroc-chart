from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DetectionResult:
    image_id: str
    source_image_path: Path
    crop_paths: list[Path]
    visualization_path: Path | None


@dataclass(frozen=True)
class CalibrationResult:
    image_id: str
    source_image_path: Path
    chart_crop_path: Path
    calibrated_image_path: Path


@dataclass(frozen=True)
class SegmentationResult:
    image_id: str
    source_image_path: Path
    mask_path: Path
    masked_image_path: Path


@dataclass(frozen=True)
class FeatureVector:
    image_id: str
    values: dict[str, float]


@dataclass(frozen=True)
class EstimationResult:
    image_id: str
    hue_group: str
    health_score: float
    category: str | None = None

