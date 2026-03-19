from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from coral_thesis.config import IMAGE_SUFFIXES
from coral_thesis.domain import FeatureVector


FEATURE_COLUMNS: tuple[str, ...] = (
    "R_mean",
    "G_mean",
    "B_mean",
    "R_std",
    "G_std",
    "B_std",
    "H_mean",
    "S_mean",
    "V_mean",
    "L_mean",
    "a_mean",
    "b_mean",
    "R_p10",
    "R_p90",
)


@dataclass(frozen=True)
class FeatureExtractionIssue:
    path: Path
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "message": self.message}


@dataclass(frozen=True)
class FeatureExtractionPair:
    image_id: str
    source_image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class FeatureExtractionInventory:
    source_dir: Path
    mask_dir: Path
    source_image_paths: tuple[Path, ...]
    pairs: tuple[FeatureExtractionPair, ...]
    orphan_mask_paths: tuple[Path, ...]
    missing_mask_image_ids: tuple[str, ...]
    issues: tuple[FeatureExtractionIssue, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "source_dir": str(self.source_dir),
            "mask_dir": str(self.mask_dir),
            "source_image_count": len(self.source_image_paths),
            "mask_count": len(self.pairs) + len(self.orphan_mask_paths),
            "matched_pair_count": len(self.pairs),
            "missing_mask_count": len(self.missing_mask_image_ids),
            "orphan_mask_count": len(self.orphan_mask_paths),
            "issue_count": len(self.issues),
            "missing_mask_image_ids": list(self.missing_mask_image_ids[:20]),
            "orphan_mask_paths": [str(path) for path in self.orphan_mask_paths[:20]],
            "issues": [issue.to_dict() for issue in self.issues[:20]],
        }


@dataclass(frozen=True)
class ExtractedFeatureDataset:
    csv_path: Path
    report_path: Path
    extracted_count: int
    skipped_count: int
    feature_columns: tuple[str, ...]
    inventory: FeatureExtractionInventory

    def summary(self) -> dict[str, Any]:
        return {
            "csv_path": str(self.csv_path),
            "report_path": str(self.report_path),
            "extracted_count": self.extracted_count,
            "skipped_count": self.skipped_count,
            "feature_columns": list(self.feature_columns),
            "inventory": self.inventory.summary(),
        }


class FeatureExtractionPhase:
    phase_name = "feature-extraction"

    def run(self, image_path: Path, mask_path: Path) -> FeatureVector:
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        if mask is None:
            raise ValueError(f"Unable to read mask: {mask_path}")

        return FeatureVector(
            image_id=image_path.stem,
            values=self.extract_from_arrays(image=image, mask=mask),
        )

    @staticmethod
    def extract_from_arrays(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        if image is None or mask is None:
            raise ValueError("image and mask are required")

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask_bool = mask > 127
        if not np.any(mask_bool):
            raise ValueError("mask does not contain any foreground pixels")

        pixels_bgr = image[mask_bool]
        pixels_rgb = cv2.cvtColor(pixels_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2RGB)[0]
        pixels_hsv = cv2.cvtColor(pixels_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)[0]
        pixels_lab = cv2.cvtColor(pixels_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2Lab)[0]

        features = {
            "R_mean": float(np.mean(pixels_rgb[:, 0])),
            "G_mean": float(np.mean(pixels_rgb[:, 1])),
            "B_mean": float(np.mean(pixels_rgb[:, 2])),
            "R_std": float(np.std(pixels_rgb[:, 0])),
            "G_std": float(np.std(pixels_rgb[:, 1])),
            "B_std": float(np.std(pixels_rgb[:, 2])),
            "H_mean": float(np.mean(pixels_hsv[:, 0])),
            "S_mean": float(np.mean(pixels_hsv[:, 1])),
            "V_mean": float(np.mean(pixels_hsv[:, 2])),
            "L_mean": float(np.mean(pixels_lab[:, 0])),
            "a_mean": float(np.mean(pixels_lab[:, 1])),
            "b_mean": float(np.mean(pixels_lab[:, 2])),
            "R_p10": float(np.percentile(pixels_rgb[:, 0], 10)),
            "R_p90": float(np.percentile(pixels_rgb[:, 0], 90)),
        }
        return {column: features[column] for column in FEATURE_COLUMNS}


def build_feature_extraction_inventory(
    source_dir: Path,
    mask_dir: Path,
) -> FeatureExtractionInventory:
    if not source_dir.exists():
        raise FileNotFoundError(f"Feature extraction source_dir does not exist: {source_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Feature extraction mask_dir does not exist: {mask_dir}")

    source_image_paths = tuple(
        sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix in IMAGE_SUFFIXES
        )
    )
    source_image_by_id = {path.stem: path for path in source_image_paths}
    mask_paths = tuple(sorted(path for path in mask_dir.glob("*.png") if path.is_file()))

    pairs: list[FeatureExtractionPair] = []
    orphan_mask_paths: list[Path] = []
    issues: list[FeatureExtractionIssue] = []

    for mask_path in mask_paths:
        source_image_path = source_image_by_id.get(mask_path.stem)
        if source_image_path is None:
            orphan_mask_paths.append(mask_path)
            issues.append(
                FeatureExtractionIssue(
                    path=mask_path,
                    message="mask does not have a matching source image",
                )
            )
            continue

        pairs.append(
            FeatureExtractionPair(
                image_id=mask_path.stem,
                source_image_path=source_image_path,
                mask_path=mask_path,
            )
        )

    paired_ids = {pair.image_id for pair in pairs}
    missing_mask_image_ids = tuple(
        sorted(path.stem for path in source_image_paths if path.stem not in paired_ids)
    )

    return FeatureExtractionInventory(
        source_dir=source_dir.resolve(),
        mask_dir=mask_dir.resolve(),
        source_image_paths=source_image_paths,
        pairs=tuple(pairs),
        orphan_mask_paths=tuple(orphan_mask_paths),
        missing_mask_image_ids=missing_mask_image_ids,
        issues=tuple(issues),
    )


def extract_feature_dataset(
    inventory: FeatureExtractionInventory,
    csv_path: Path,
    report_path: Path,
) -> ExtractedFeatureDataset:
    phase = FeatureExtractionPhase()
    rows: list[dict[str, float | str]] = []
    issues = [issue.to_dict() for issue in inventory.issues]

    for pair in inventory.pairs:
        try:
            feature_vector = phase.run(pair.source_image_path, pair.mask_path)
        except ValueError as exc:
            issues.append({"path": str(pair.mask_path), "message": str(exc)})
            continue

        row: dict[str, float | str] = {"image_id": feature_vector.image_id}
        for column in FEATURE_COLUMNS:
            row[column] = feature_vector.values[column]
        rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(rows, columns=["image_id", *FEATURE_COLUMNS])
    frame.to_csv(csv_path, index=False)

    report = {
        "inventory": inventory.summary(),
        "csv_path": str(csv_path),
        "report_path": str(report_path),
        "extracted_count": len(rows),
        "skipped_count": len(issues) - len(inventory.issues),
        "feature_columns": list(FEATURE_COLUMNS),
        "issues": issues[:50],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return ExtractedFeatureDataset(
        csv_path=csv_path.resolve(),
        report_path=report_path.resolve(),
        extracted_count=len(rows),
        skipped_count=len(issues) - len(inventory.issues),
        feature_columns=FEATURE_COLUMNS,
        inventory=inventory,
    )
