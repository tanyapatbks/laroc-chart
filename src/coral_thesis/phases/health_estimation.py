from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from coral_thesis.domain import EstimationResult, FeatureVector
from coral_thesis.phases.category_mapping import map_to_category
from coral_thesis.phases.feature_extraction import FEATURE_COLUMNS

HUE_GROUPS = ("B", "C", "D", "E")
CATEGORY_PATTERN = re.compile(r"^([BCDE])([1-6])$")


@dataclass(frozen=True)
class Phase5Issue:
    path: Path
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "message": self.message}


@dataclass(frozen=True)
class CoralWatchReferencePatch:
    category: str
    hue_group: str
    health_level: int
    health_score: float
    mean_bgr: tuple[float, float, float]
    mean_lab: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoralWatchReference:
    baseline_chart_path: Path
    patches: tuple[CoralWatchReferencePatch, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "baseline_chart_path": str(self.baseline_chart_path),
            "patch_count": len(self.patches),
            "categories": [patch.category for patch in self.patches],
        }


@dataclass(frozen=True)
class Phase5Inventory:
    features_csv_path: Path
    labels_csv_path: Path | None
    feature_row_count: int
    label_row_count: int
    matched_row_count: int
    unlabeled_feature_ids: tuple[str, ...]
    label_only_ids: tuple[str, ...]
    label_source_format: str | None
    feature_columns: tuple[str, ...]
    issues: tuple[Phase5Issue, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "features_csv_path": str(self.features_csv_path),
            "labels_csv_path": str(self.labels_csv_path) if self.labels_csv_path is not None else None,
            "feature_row_count": self.feature_row_count,
            "label_row_count": self.label_row_count,
            "matched_row_count": self.matched_row_count,
            "unlabeled_feature_count": len(self.unlabeled_feature_ids),
            "label_only_count": len(self.label_only_ids),
            "label_source_format": self.label_source_format,
            "feature_columns": list(self.feature_columns),
            "unlabeled_feature_ids": list(self.unlabeled_feature_ids[:20]),
            "label_only_ids": list(self.label_only_ids[:20]),
            "issue_count": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues[:20]],
        }


@dataclass(frozen=True)
class Phase5TrainingArtifacts:
    hue_model_path: Path
    health_model_path: Path
    report_path: Path
    train_row_count: int
    validation_row_count: int
    feature_columns: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "hue_model_path": str(self.hue_model_path),
            "health_model_path": str(self.health_model_path),
            "report_path": str(self.report_path),
            "train_row_count": self.train_row_count,
            "validation_row_count": self.validation_row_count,
            "feature_columns": list(self.feature_columns),
        }


@dataclass(frozen=True)
class Phase5PredictionDataset:
    csv_path: Path
    report_path: Path
    estimated_count: int
    strategy_used: str
    feature_columns: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "csv_path": str(self.csv_path),
            "report_path": str(self.report_path),
            "estimated_count": self.estimated_count,
            "strategy_used": self.strategy_used,
            "feature_columns": list(self.feature_columns),
        }


def health_level_to_score(level: int) -> float:
    if level < 1 or level > 6:
        raise ValueError(f"health_level must be within [1, 6], found {level}")
    return float((level - 0.5) / 6.0)


def parse_coralwatch_category(category: str) -> tuple[str, int]:
    match = CATEGORY_PATTERN.fullmatch(category.strip().upper())
    if match is None:
        raise ValueError(f"invalid CoralWatch category: {category!r}")
    return match.group(1), int(match.group(2))


def _category_from_hue_and_score(hue_group: str, health_score: float) -> str:
    category = map_to_category(hue_group, health_score)
    if category in {"Unknown", "Error"}:
        raise ValueError(
            f"unable to derive CoralWatch category from hue_group={hue_group!r}, health_score={health_score!r}"
        )
    return category


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"unable to read image: {path}")
    return image


def _patch_sample_bounds(height: int, width: int, orientation: str) -> tuple[slice, slice]:
    if orientation == "top":
        return slice(0, max(1, int(height * 0.35))), slice(int(width * 0.10), max(int(width * 0.90), 1))
    if orientation == "right":
        return slice(int(height * 0.10), max(int(height * 0.90), 1)), slice(int(width * 0.60), width)
    if orientation == "bottom":
        return slice(int(height * 0.65), height), slice(int(width * 0.10), max(int(width * 0.90), 1))
    if orientation == "left":
        return slice(int(height * 0.10), max(int(height * 0.90), 1)), slice(0, max(1, int(width * 0.40)))
    raise ValueError(f"unsupported reference patch orientation: {orientation}")


def _dominant_patch_color(cell: np.ndarray, orientation: str) -> tuple[float, float, float]:
    row_slice, col_slice = _patch_sample_bounds(cell.shape[0], cell.shape[1], orientation)
    sample = cell[row_slice, col_slice]
    if sample.size == 0:
        sample = cell

    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY).reshape(-1)
    pixels = sample.reshape(-1, 3).astype(np.float32)
    threshold = float(np.quantile(gray, 0.35))
    filtered_pixels = pixels[gray >= threshold]
    if len(filtered_pixels) < max(25, len(pixels) // 5):
        filtered_pixels = pixels
    mean_bgr = filtered_pixels.mean(axis=0)
    return float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])


def _reference_patch_specs() -> list[tuple[str, str, int, int, int, str]]:
    specs: list[tuple[str, str, int, int, int, str]] = []
    for level in range(1, 7):
        specs.append((f"B{level}", "B", level, 0, level - 1, "top"))
        specs.append((f"C{level}", "C", level, level - 1, 6, "right"))
        specs.append((f"D{level}", "D", level, 6, 7 - level, "bottom"))
        specs.append((f"E{level}", "E", level, 7 - level, 0, "left"))
    return specs


def build_coralwatch_reference(baseline_chart_path: Path) -> CoralWatchReference:
    baseline_chart = _read_image(baseline_chart_path)
    row_edges = np.linspace(0, baseline_chart.shape[0], 8, dtype=int)
    col_edges = np.linspace(0, baseline_chart.shape[1], 8, dtype=int)

    patches: list[CoralWatchReferencePatch] = []
    for category, hue_group, health_level, row_index, col_index, orientation in _reference_patch_specs():
        cell = baseline_chart[
            row_edges[row_index]:row_edges[row_index + 1],
            col_edges[col_index]:col_edges[col_index + 1],
        ]
        mean_bgr = _dominant_patch_color(cell, orientation=orientation)
        mean_lab_pixel = cv2.cvtColor(
            np.array([[np.round(mean_bgr)]], dtype=np.uint8),
            cv2.COLOR_BGR2Lab,
        )[0, 0]
        patches.append(
            CoralWatchReferencePatch(
                category=category,
                hue_group=hue_group,
                health_level=health_level,
                health_score=health_level_to_score(health_level),
                mean_bgr=mean_bgr,
                mean_lab=(float(mean_lab_pixel[0]), float(mean_lab_pixel[1]), float(mean_lab_pixel[2])),
            )
        )

    return CoralWatchReference(
        baseline_chart_path=baseline_chart_path.resolve(),
        patches=tuple(sorted(patches, key=lambda patch: patch.category)),
    )


def _feature_vector_from_series(row: pd.Series) -> FeatureVector:
    values = {column: float(row[column]) for column in FEATURE_COLUMNS}
    return FeatureVector(image_id=str(row["image_id"]), values=values)


def _load_feature_table(features_csv_path: Path) -> pd.DataFrame:
    if not features_csv_path.exists():
        raise FileNotFoundError(f"Phase 5 features CSV does not exist: {features_csv_path}")

    frame = pd.read_csv(features_csv_path)
    required_columns = {"image_id", *FEATURE_COLUMNS}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Phase 5 features CSV is missing required columns: {', '.join(missing_columns)}"
        )

    duplicated_ids = frame["image_id"][frame["image_id"].duplicated()].astype(str).unique().tolist()
    if duplicated_ids:
        raise ValueError(
            "Phase 5 features CSV contains duplicated image_id values: "
            + ", ".join(duplicated_ids[:10])
        )

    return frame[["image_id", *FEATURE_COLUMNS]].copy()


def normalize_phase5_labels(labels_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "image_id" not in labels_df.columns:
        raise ValueError("Phase 5 labels CSV must contain an 'image_id' column.")

    normalized_rows: list[dict[str, Any]] = []
    source_format: str | None = None

    def _update_source_format(current: str) -> None:
        nonlocal source_format
        if source_format is None:
            source_format = current
        elif source_format != current:
            source_format = "mixed"

    for _, row in labels_df.iterrows():
        image_id = str(row["image_id"]).strip()
        if not image_id:
            raise ValueError("Phase 5 labels CSV contains an empty image_id value.")

        category_value = row.get("category")
        hue_group_value = row.get("hue_group")
        health_level_value = row.get("health_level")
        health_score_value = row.get("health_score")

        if pd.notna(category_value) and str(category_value).strip():
            hue_group, health_level = parse_coralwatch_category(str(category_value))
            health_score = health_level_to_score(health_level)
            _update_source_format("category")
        elif pd.notna(hue_group_value) and str(hue_group_value).strip():
            hue_group = str(hue_group_value).strip().upper()
            if hue_group not in HUE_GROUPS:
                raise ValueError(f"invalid hue_group value: {hue_group!r}")

            if pd.notna(health_level_value) and str(health_level_value).strip():
                health_level = int(health_level_value)
                health_score = health_level_to_score(health_level)
                _update_source_format("hue_group+health_level")
            elif pd.notna(health_score_value) and str(health_score_value).strip():
                health_score = float(health_score_value)
                category = _category_from_hue_and_score(hue_group, health_score)
                _, health_level = parse_coralwatch_category(category)
                _update_source_format("hue_group+health_score")
            else:
                raise ValueError(
                    "Phase 5 labels CSV must provide category, or hue_group with health_level/health_score."
                )
        else:
            raise ValueError(
                "Phase 5 labels CSV must provide category, or hue_group with health_level/health_score."
            )

        category = _category_from_hue_and_score(hue_group, health_score)
        normalized_rows.append(
            {
                "image_id": image_id,
                "hue_group": hue_group,
                "health_level": health_level,
                "health_score": float(np.clip(health_score, 0.0, 1.0)),
                "category": category,
            }
        )

    normalized = pd.DataFrame(
        normalized_rows,
        columns=["image_id", "hue_group", "health_level", "health_score", "category"],
    )
    duplicated_ids = normalized["image_id"][normalized["image_id"].duplicated()].astype(str).unique().tolist()
    if duplicated_ids:
        raise ValueError(
            "Phase 5 labels CSV contains duplicated image_id values: "
            + ", ".join(duplicated_ids[:10])
        )
    return normalized, source_format or "unknown"


def build_phase5_inventory(
    features_csv_path: Path,
    labels_csv_path: Path | None = None,
) -> Phase5Inventory:
    features_df = _load_feature_table(features_csv_path)
    issues: list[Phase5Issue] = []
    label_row_count = 0
    matched_row_count = 0
    unlabeled_feature_ids = tuple(features_df["image_id"].astype(str).tolist())
    label_only_ids: tuple[str, ...] = ()
    label_source_format: str | None = None

    if labels_csv_path is not None:
        if labels_csv_path.exists():
            normalized_labels, label_source_format = normalize_phase5_labels(
                pd.read_csv(labels_csv_path)
            )
            label_row_count = len(normalized_labels)
            merged = features_df.merge(normalized_labels, on="image_id", how="inner")
            matched_row_count = len(merged)
            feature_ids = set(features_df["image_id"].astype(str))
            label_ids = set(normalized_labels["image_id"].astype(str))
            unlabeled_feature_ids = tuple(sorted(feature_ids - label_ids))
            label_only_ids = tuple(sorted(label_ids - feature_ids))
        else:
            issues.append(
                Phase5Issue(
                    path=labels_csv_path,
                    message="Phase 5 labels CSV does not exist yet.",
                )
            )

    return Phase5Inventory(
        features_csv_path=features_csv_path.resolve(),
        labels_csv_path=labels_csv_path.resolve() if labels_csv_path is not None else None,
        feature_row_count=len(features_df),
        label_row_count=label_row_count,
        matched_row_count=matched_row_count,
        unlabeled_feature_ids=unlabeled_feature_ids,
        label_only_ids=label_only_ids,
        label_source_format=label_source_format,
        feature_columns=FEATURE_COLUMNS,
        issues=tuple(issues),
    )


def export_phase5_label_template(features_csv_path: Path, template_path: Path) -> Path:
    features_df = _load_feature_table(features_csv_path)
    template = pd.DataFrame(
        {
            "image_id": features_df["image_id"].astype(str),
            "category": "",
            "hue_group": "",
            "health_level": "",
            "health_score": "",
            "notes": "",
        }
    )
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(template_path, index=False)
    return template_path.resolve()


def _safe_stratify_labels(labels: pd.Series, validation_split: float) -> pd.Series | None:
    counts = labels.value_counts()
    if len(counts) < 2 or int(counts.min()) < 2:
        return None
    validation_count = int(math.ceil(len(labels) * validation_split))
    if validation_count < len(counts):
        return None
    return labels


def _feature_importance_summary(model: Any, feature_columns: tuple[str, ...], limit: int = 5) -> list[dict[str, Any]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    ordered_indices = np.argsort(importances)[::-1][:limit]
    return [
        {
            "feature": feature_columns[index],
            "importance": float(importances[index]),
        }
        for index in ordered_indices
    ]


def _evaluation_metrics(
    y_hue_true: pd.Series,
    y_hue_pred: np.ndarray,
    y_health_true: pd.Series,
    y_health_pred: np.ndarray,
    y_category_true: pd.Series,
    y_category_pred: list[str],
) -> dict[str, Any]:
    metrics = {
        "sample_count": int(len(y_hue_true)),
        "hue_accuracy": float(accuracy_score(y_hue_true, y_hue_pred)),
        "hue_balanced_accuracy": float(balanced_accuracy_score(y_hue_true, y_hue_pred)),
        "health_mae": float(mean_absolute_error(y_health_true, y_health_pred)),
        "health_rmse": float(np.sqrt(mean_squared_error(y_health_true, y_health_pred))),
        "category_accuracy": float(np.mean(np.array(y_category_true) == np.array(y_category_pred))),
    }
    if len(y_health_true) >= 2:
        metrics["health_r2"] = float(r2_score(y_health_true, y_health_pred))
    else:
        metrics["health_r2"] = None
    return metrics


def _series_distribution(series: pd.Series) -> dict[str, int]:
    return {str(index): int(value) for index, value in series.value_counts().sort_index().items()}


def train_health_models(
    features_csv_path: Path,
    labels_csv_path: Path,
    baseline_chart_path: Path,
    hue_model_path: Path,
    health_model_path: Path,
    report_path: Path,
    seed: int,
    validation_split: float,
    classifier_estimators: int,
    regressor_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
) -> Phase5TrainingArtifacts:
    features_df = _load_feature_table(features_csv_path)
    normalized_labels, label_source_format = normalize_phase5_labels(pd.read_csv(labels_csv_path))
    merged = features_df.merge(normalized_labels, on="image_id", how="inner")
    if len(merged) < 8:
        raise ValueError(
            f"Phase 5 training requires at least 8 matched rows, found {len(merged)}."
        )

    X = merged[list(FEATURE_COLUMNS)]
    y_hue = merged["hue_group"]
    y_health = merged["health_score"]
    y_category = merged["category"]

    stratify_labels = _safe_stratify_labels(y_category, validation_split=validation_split)
    (
        X_train,
        X_val,
        y_hue_train,
        y_hue_val,
        y_health_train,
        y_health_val,
        y_category_train,
        y_category_val,
    ) = train_test_split(
        X,
        y_hue,
        y_health,
        y_category,
        test_size=validation_split,
        random_state=seed,
        stratify=stratify_labels,
    )

    hue_model = RandomForestClassifier(
        n_estimators=classifier_estimators,
        random_state=seed,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    health_model = RandomForestRegressor(
        n_estimators=regressor_estimators,
        random_state=seed,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    hue_model.fit(X_train, y_hue_train)
    health_model.fit(X_train, y_health_train)

    hue_predictions = hue_model.predict(X_val)
    health_predictions = np.clip(health_model.predict(X_val), 0.0, 1.0)
    category_predictions = [
        _category_from_hue_and_score(hue_group, float(health_score))
        for hue_group, health_score in zip(hue_predictions, health_predictions, strict=True)
    ]
    model_metrics = _evaluation_metrics(
        y_hue_true=y_hue_val,
        y_hue_pred=hue_predictions,
        y_health_true=y_health_val,
        y_health_pred=health_predictions,
        y_category_true=y_category_val,
        y_category_pred=category_predictions,
    )

    reference = build_coralwatch_reference(baseline_chart_path)
    heuristic_estimator = HealthEstimationPhase(
        hue_model_path=None,
        health_model_path=None,
        baseline_chart_path=baseline_chart_path,
        strategy="heuristic",
        reference=reference,
    )
    heuristic_predictions = [heuristic_estimator.run(_feature_vector_from_series(row)) for _, row in X_val.join(
        merged.loc[X_val.index, ["image_id"]]
    ).iterrows()]
    heuristic_metrics = _evaluation_metrics(
        y_hue_true=y_hue_val,
        y_hue_pred=np.array([prediction.hue_group for prediction in heuristic_predictions]),
        y_health_true=y_health_val,
        y_health_pred=np.array([prediction.health_score for prediction in heuristic_predictions], dtype=np.float32),
        y_category_true=y_category_val,
        y_category_pred=[prediction.category or "Unknown" for prediction in heuristic_predictions],
    )

    hue_model_path.parent.mkdir(parents=True, exist_ok=True)
    health_model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(hue_model, hue_model_path)
    joblib.dump(health_model, health_model_path)

    report = {
        "features_csv_path": str(features_csv_path.resolve()),
        "labels_csv_path": str(labels_csv_path.resolve()),
        "label_source_format": label_source_format,
        "baseline_reference": reference.summary(),
        "feature_columns": list(FEATURE_COLUMNS),
        "train_row_count": int(len(X_train)),
        "validation_row_count": int(len(X_val)),
        "train_category_distribution": _series_distribution(y_category_train),
        "validation_category_distribution": _series_distribution(y_category_val),
        "model_paths": {
            "hue_model_path": str(hue_model_path.resolve()),
            "health_model_path": str(health_model_path.resolve()),
        },
        "training_parameters": {
            "seed": seed,
            "validation_split": validation_split,
            "classifier_estimators": classifier_estimators,
            "regressor_estimators": regressor_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
        },
        "validation_metrics": {
            "model": model_metrics,
            "heuristic_baseline": heuristic_metrics,
        },
        "feature_importance": {
            "hue_model": _feature_importance_summary(hue_model, FEATURE_COLUMNS),
            "health_model": _feature_importance_summary(health_model, FEATURE_COLUMNS),
        },
        "report_path": str(report_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return Phase5TrainingArtifacts(
        hue_model_path=hue_model_path.resolve(),
        health_model_path=health_model_path.resolve(),
        report_path=report_path.resolve(),
        train_row_count=len(X_train),
        validation_row_count=len(X_val),
        feature_columns=FEATURE_COLUMNS,
    )


class HealthEstimationPhase:
    phase_name = "health-estimation"

    def __init__(
        self,
        hue_model_path: Path | None,
        health_model_path: Path | None,
        baseline_chart_path: Path | None = None,
        strategy: str = "auto",
        reference: CoralWatchReference | None = None,
    ) -> None:
        if strategy not in {"auto", "heuristic", "model"}:
            raise ValueError("strategy must be one of {'auto', 'heuristic', 'model'}")
        self.hue_model_path = hue_model_path
        self.health_model_path = health_model_path
        self.baseline_chart_path = baseline_chart_path
        self.strategy = strategy
        self.hue_model = self._load_model(hue_model_path)
        self.health_model = self._load_model(health_model_path)
        self.reference = reference
        if self.reference is None and baseline_chart_path is not None:
            self.reference = build_coralwatch_reference(baseline_chart_path)

    @staticmethod
    def _load_model(model_path: Path | None):
        if model_path is None:
            return None
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        return joblib.load(model_path)

    def is_model_ready(self) -> bool:
        return self.hue_model is not None and self.health_model is not None

    def resolved_strategy(self) -> str:
        if self.strategy == "model":
            if not self.is_model_ready():
                raise RuntimeError(
                    "Health estimation models are not configured. "
                    "Provide both hue_model_path and health_model_path."
                )
            return "model"
        if self.strategy == "heuristic":
            if self.reference is None:
                raise RuntimeError(
                    "Heuristic health estimation requires baseline_chart_path or a prebuilt reference."
                )
            return "heuristic"
        if self.is_model_ready():
            return "model"
        if self.reference is not None:
            return "heuristic"
        raise RuntimeError(
            "Health estimation is not ready. Provide models or baseline_chart_path for heuristic estimation."
        )

    def _run_model(self, features: FeatureVector) -> EstimationResult:
        frame = pd.DataFrame([features.values], columns=list(FEATURE_COLUMNS))
        hue_group = str(self.hue_model.predict(frame)[0])
        health_score = float(np.clip(self.health_model.predict(frame)[0], 0.0, 1.0))
        category = map_to_category(hue_group, health_score)
        return EstimationResult(
            image_id=features.image_id,
            hue_group=hue_group,
            health_score=health_score,
            category=category,
        )

    def _run_heuristic(self, features: FeatureVector) -> EstimationResult:
        if self.reference is None:
            raise RuntimeError(
                "Heuristic health estimation requires baseline_chart_path or a prebuilt reference."
            )

        lab_vector = np.array(
            [
                features.values["L_mean"],
                features.values["a_mean"],
                features.values["b_mean"],
            ],
            dtype=np.float32,
        )
        best_patch = min(
            self.reference.patches,
            key=lambda patch: float(np.linalg.norm(lab_vector - np.array(patch.mean_lab, dtype=np.float32))),
        )
        return EstimationResult(
            image_id=features.image_id,
            hue_group=best_patch.hue_group,
            health_score=best_patch.health_score,
            category=best_patch.category,
        )

    def run(self, features: FeatureVector) -> EstimationResult:
        strategy = self.resolved_strategy()
        if strategy == "model":
            return self._run_model(features)
        return self._run_heuristic(features)


def estimate_health_dataset(
    features_csv_path: Path,
    baseline_chart_path: Path,
    csv_path: Path,
    report_path: Path,
    strategy: str,
    hue_model_path: Path | None,
    health_model_path: Path | None,
) -> Phase5PredictionDataset:
    features_df = _load_feature_table(features_csv_path)
    estimator = HealthEstimationPhase(
        hue_model_path=hue_model_path,
        health_model_path=health_model_path,
        baseline_chart_path=baseline_chart_path,
        strategy=strategy,
    )
    strategy_used = estimator.resolved_strategy()

    prediction_rows: list[dict[str, Any]] = []
    for _, row in features_df.iterrows():
        result = estimator.run(_feature_vector_from_series(row))
        prediction_rows.append(
            {
                "image_id": result.image_id,
                "hue_group": result.hue_group,
                "health_score": result.health_score,
                "category": result.category,
                "strategy": strategy_used,
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(
        prediction_rows,
        columns=["image_id", "hue_group", "health_score", "category", "strategy"],
    )
    prediction_frame.to_csv(csv_path, index=False)

    category_counts = _series_distribution(prediction_frame["category"])
    report = {
        "features_csv_path": str(features_csv_path.resolve()),
        "csv_path": str(csv_path.resolve()),
        "report_path": str(report_path.resolve()),
        "strategy_requested": strategy,
        "strategy_used": strategy_used,
        "estimated_count": int(len(prediction_frame)),
        "category_counts": category_counts,
        "hue_group_counts": _series_distribution(prediction_frame["hue_group"]),
        "baseline_reference": estimator.reference.summary() if estimator.reference is not None else None,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return Phase5PredictionDataset(
        csv_path=csv_path.resolve(),
        report_path=report_path.resolve(),
        estimated_count=len(prediction_frame),
        strategy_used=strategy_used,
        feature_columns=FEATURE_COLUMNS,
    )
