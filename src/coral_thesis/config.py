from __future__ import annotations

from dataclasses import asdict, dataclass
import os
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
class Phase1TrainConfig:
    run_name: str
    epochs: int
    batch_size: int
    patience: int
    augment: bool
    workers: int
    plots: bool
    prewarm_font_cache: bool
    prewarm_dataset: bool
    drop_corrupt_samples: bool
    sample_timeout_seconds: int
    warmup_workers: int
    warmup_log_interval: int


@dataclass(frozen=True)
class Phase1Config:
    class_name: str
    prepared_dataset_dir: Path
    training_dir: Path
    inference_dir: Path
    val_split: float
    seed: int
    excluded_image_ids: tuple[str, ...]
    use_symlinks: bool
    skip_unlabeled_images: bool
    train: Phase1TrainConfig


@dataclass(frozen=True)
class Phase2Config:
    output_dir: Path
    baseline_profile_path: Path
    evaluation_manifest_path: Path
    evaluation_reports_dir: Path
    patch_rows: int
    patch_cols: int
    cell_sample_ratio: float
    min_patch_count: int
    method: str
    sample_timeout_seconds: int | None


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    config_path: Path
    paths: PathConfig
    models: ModelConfig
    runtime: RuntimeConfig
    phase1: Phase1Config
    phase2: Phase2Config

    def ensure_workspace(self) -> None:
        for directory in (
            self.paths.artifacts_dir,
            self.paths.outputs_dir,
            self.paths.temp_dir,
            self.paths.reports_dir,
            self.phase1.prepared_dataset_dir,
            self.phase1.training_dir,
            self.phase1.inference_dir,
            self.phase2.output_dir,
            self.phase2.evaluation_reports_dir,
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

        if not 0 < self.phase1.val_split < 1:
            issues.append("phase1.val_split must be within the interval (0, 1).")
        if self.phase1.seed < 0:
            issues.append("phase1.seed must be non-negative.")
        if self.phase1.train.epochs <= 0:
            issues.append("phase1.train.epochs must be a positive integer.")
        if self.phase1.train.batch_size <= 0:
            issues.append("phase1.train.batch_size must be a positive integer.")
        if self.phase1.train.patience < 0:
            issues.append("phase1.train.patience must be zero or greater.")
        if self.phase1.train.workers < 0:
            issues.append("phase1.train.workers must be zero or greater.")
        if self.phase1.train.warmup_workers <= 0:
            issues.append("phase1.train.warmup_workers must be a positive integer.")
        if self.phase1.train.warmup_log_interval <= 0:
            issues.append("phase1.train.warmup_log_interval must be a positive integer.")
        if self.phase1.train.sample_timeout_seconds <= 0:
            issues.append("phase1.train.sample_timeout_seconds must be a positive integer.")
        if not self.phase1.class_name:
            issues.append("phase1.class_name must not be empty.")
        if self.phase2.patch_rows <= 0 or self.phase2.patch_cols <= 0:
            issues.append("phase2.patch_rows and phase2.patch_cols must be positive integers.")
        if not 0 < self.phase2.cell_sample_ratio <= 1:
            issues.append("phase2.cell_sample_ratio must be within the interval (0, 1].")
        if self.phase2.min_patch_count <= 0:
            issues.append("phase2.min_patch_count must be a positive integer.")
        if self.phase2.method not in {"linear"}:
            issues.append("phase2.method currently supports only 'linear'.")
        if not self.phase2.evaluation_manifest_path.exists():
            issues.append(
                "phase2.evaluation_manifest_path does not exist: "
                f"{self.phase2.evaluation_manifest_path}"
            )
        if (
            self.phase2.sample_timeout_seconds is not None
            and self.phase2.sample_timeout_seconds <= 0
        ):
            issues.append("phase2.sample_timeout_seconds must be a positive integer when set.")

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
    phase1_raw = raw.get("phase1", {})
    phase1_train_raw = phase1_raw.get("train", {})
    phase2_raw = raw.get("phase2", {})

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
    phase1 = Phase1Config(
        class_name=str(phase1_raw.get("class_name", "chart")),
        prepared_dataset_dir=_resolve_path(
            phase1_raw.get("prepared_dataset_dir", "artifacts/outputs/phase1/dataset"),
            project_root,
        ),
        training_dir=_resolve_path(
            phase1_raw.get("training_dir", "artifacts/outputs/phase1/training"),
            project_root,
        ),
        inference_dir=_resolve_path(
            phase1_raw.get("inference_dir", "artifacts/outputs/phase1/inference"),
            project_root,
        ),
        val_split=float(phase1_raw.get("val_split", 0.2)),
        seed=int(phase1_raw.get("seed", 42)),
        excluded_image_ids=tuple(str(value) for value in phase1_raw.get("excluded_image_ids", [])),
        use_symlinks=bool(phase1_raw.get("use_symlinks", True)),
        skip_unlabeled_images=bool(phase1_raw.get("skip_unlabeled_images", True)),
        train=Phase1TrainConfig(
            run_name=str(phase1_train_raw.get("run_name", "chart_detector")),
            epochs=int(phase1_train_raw.get("epochs", 10)),
            batch_size=int(phase1_train_raw.get("batch_size", 8)),
            patience=int(phase1_train_raw.get("patience", 20)),
            augment=bool(phase1_train_raw.get("augment", True)),
            workers=int(phase1_train_raw.get("workers", 0)),
            plots=bool(phase1_train_raw.get("plots", False)),
            prewarm_font_cache=bool(phase1_train_raw.get("prewarm_font_cache", True)),
            prewarm_dataset=bool(phase1_train_raw.get("prewarm_dataset", True)),
            drop_corrupt_samples=bool(phase1_train_raw.get("drop_corrupt_samples", True)),
            sample_timeout_seconds=int(phase1_train_raw.get("sample_timeout_seconds", 15)),
            warmup_workers=int(phase1_train_raw.get("warmup_workers", 1)),
            warmup_log_interval=int(phase1_train_raw.get("warmup_log_interval", 1)),
        ),
    )
    phase2 = Phase2Config(
        output_dir=_resolve_path(
            phase2_raw.get("output_dir", "artifacts/outputs/phase2"),
            project_root,
        ),
        baseline_profile_path=_resolve_path(
            phase2_raw.get("baseline_profile_path", "artifacts/outputs/phase2/baseline_profile.json"),
            project_root,
        ),
        evaluation_manifest_path=_resolve_path(
            phase2_raw.get("evaluation_manifest_path", "configs/phase2_evaluation_manifest.yaml"),
            project_root,
        ),
        evaluation_reports_dir=_resolve_path(
            phase2_raw.get("evaluation_reports_dir", "artifacts/reports/phase2"),
            project_root,
        ),
        patch_rows=int(phase2_raw.get("patch_rows", 6)),
        patch_cols=int(phase2_raw.get("patch_cols", 4)),
        cell_sample_ratio=float(phase2_raw.get("cell_sample_ratio", 0.5)),
        min_patch_count=int(phase2_raw.get("min_patch_count", 8)),
        method=str(phase2_raw.get("method", "linear")),
        sample_timeout_seconds=(
            int(phase2_raw["sample_timeout_seconds"])
            if phase2_raw.get("sample_timeout_seconds") is not None
            else None
        ),
    )

    os.environ.setdefault("MPLCONFIGDIR", str((project_root / "artifacts" / ".matplotlib").resolve()))

    return PipelineConfig(
        project_root=project_root,
        config_path=config_path,
        paths=paths,
        models=models,
        runtime=runtime,
        phase1=phase1,
        phase2=phase2,
    )
