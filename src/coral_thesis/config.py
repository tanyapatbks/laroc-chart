from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG")


@dataclass(frozen=True)
class PathConfig:
    dataset_dir: Path
    baseline_chart_path: Path
    artifacts_dir: Path
    outputs_dir: Path
    temp_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    chart_detector_weights: Path | None
    coral_segmenter_weights: Path | None
    hue_model_path: Path | None
    health_model_path: Path | None
    detection_backbone: Path
    segmentation_backbone: Path


@dataclass(frozen=True)
class RuntimeConfig:
    confidence_threshold: float
    image_size: int
    device: str


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    config_path: Path
    paths: PathConfig
    models: ModelConfig
    runtime: RuntimeConfig

    def ensure_workspace(self) -> None:
        for directory in (
            self.paths.artifacts_dir,
            self.paths.outputs_dir,
            self.paths.temp_dir,
            self.paths.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def validate(self) -> list[str]:
        issues: list[str] = []

        if not self.paths.dataset_dir.exists():
            issues.append(f"dataset_dir does not exist: {self.paths.dataset_dir}")
        if not self.paths.baseline_chart_path.exists():
            issues.append(
                f"baseline_chart_path does not exist: {self.paths.baseline_chart_path}"
            )
        if self.runtime.confidence_threshold <= 0 or self.runtime.confidence_threshold > 1:
            issues.append(
                "runtime.confidence_threshold must be within the interval (0, 1]."
            )
        if self.runtime.image_size <= 0:
            issues.append("runtime.image_size must be a positive integer.")
        if not self.models.detection_backbone.exists():
            issues.append(
                f"detection_backbone does not exist: {self.models.detection_backbone}"
            )
        if not self.models.segmentation_backbone.exists():
            issues.append(
                "segmentation_backbone does not exist: "
                f"{self.models.segmentation_backbone}"
            )

        for label, path in (
            ("chart_detector_weights", self.models.chart_detector_weights),
            ("coral_segmenter_weights", self.models.coral_segmenter_weights),
            ("hue_model_path", self.models.hue_model_path),
            ("health_model_path", self.models.health_model_path),
        ):
            if path is not None and not path.exists():
                issues.append(f"{label} does not exist: {path}")

        return issues

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["project_root"] = str(self.project_root)
        data["config_path"] = str(self.config_path)
        _stringify_paths(data)
        return data


def _stringify_paths(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if isinstance(item, Path):
                value[key] = str(item)
            else:
                _stringify_paths(item)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, Path):
                value[index] = str(item)
            else:
                _stringify_paths(item)


def _resolve_path(raw_value: str | None, project_root: Path) -> Path | None:
    if raw_value is None:
        return None

    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def load_config(config_path: str | Path) -> PipelineConfig:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    paths_raw = raw.get("paths", {})
    models_raw = raw.get("models", {})
    runtime_raw = raw.get("runtime", {})

    paths = PathConfig(
        dataset_dir=_resolve_path(paths_raw["dataset_dir"], project_root),
        baseline_chart_path=_resolve_path(paths_raw["baseline_chart_path"], project_root),
        artifacts_dir=_resolve_path(paths_raw["artifacts_dir"], project_root),
        outputs_dir=_resolve_path(paths_raw["outputs_dir"], project_root),
        temp_dir=_resolve_path(paths_raw["temp_dir"], project_root),
        reports_dir=_resolve_path(paths_raw["reports_dir"], project_root),
    )
    models = ModelConfig(
        chart_detector_weights=_resolve_path(
            models_raw.get("chart_detector_weights"), project_root
        ),
        coral_segmenter_weights=_resolve_path(
            models_raw.get("coral_segmenter_weights"), project_root
        ),
        hue_model_path=_resolve_path(models_raw.get("hue_model_path"), project_root),
        health_model_path=_resolve_path(models_raw.get("health_model_path"), project_root),
        detection_backbone=_resolve_path(models_raw["detection_backbone"], project_root),
        segmentation_backbone=_resolve_path(
            models_raw["segmentation_backbone"], project_root
        ),
    )
    runtime = RuntimeConfig(
        confidence_threshold=float(runtime_raw.get("confidence_threshold", 0.25)),
        image_size=int(runtime_raw.get("image_size", 640)),
        device=str(runtime_raw.get("device", "cpu")),
    )

    return PipelineConfig(
        project_root=project_root,
        config_path=config_path,
        paths=paths,
        models=models,
        runtime=runtime,
    )

