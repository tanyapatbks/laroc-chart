from __future__ import annotations

from collections import Counter
import json
import os
import random
import signal
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import cv2
import numpy as np
import yaml

from coral_thesis.config import IMAGE_SUFFIXES
from coral_thesis.domain import SegmentationResult

BOUNDARY_EPSILON = 1e-6


@dataclass(frozen=True)
class SegmentationPolygon:
    class_id: int
    points: tuple[tuple[float, float], ...]
    source_format: str = "polygon"

    def as_yolo_row(self) -> str:
        point_tokens = " ".join(f"{x:.6f} {y:.6f}" for x, y in self.points)
        return f"{self.class_id} {point_tokens}"


@dataclass(frozen=True)
class SegmentationDatasetIssue:
    path: Path
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "message": self.message}


@dataclass(frozen=True)
class SegmentationDatasetItem:
    image_path: Path
    label_path: Path | None
    polygons: tuple[SegmentationPolygon, ...]
    issues: tuple[SegmentationDatasetIssue, ...]

    @property
    def image_id(self) -> str:
        return self.image_path.stem

    @property
    def is_labeled(self) -> bool:
        return self.label_path is not None

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0

    @property
    def polygon_count(self) -> int:
        return len(self.polygons)

    @property
    def total_point_count(self) -> int:
        return sum(len(polygon.points) for polygon in self.polygons)

    @property
    def annotation_format_counts(self) -> dict[str, int]:
        counts = Counter(polygon.source_format for polygon in self.polygons)
        return dict(sorted(counts.items()))


@dataclass(frozen=True)
class SegmentationDatasetInventory:
    dataset_dir: Path
    label_dir: Path
    class_name: str
    items: tuple[SegmentationDatasetItem, ...]
    orphan_label_paths: tuple[Path, ...]
    duplicate_label_paths: tuple[Path, ...]
    json_annotation_paths: tuple[Path, ...]
    issues: tuple[SegmentationDatasetIssue, ...]

    @property
    def labeled_items(self) -> tuple[SegmentationDatasetItem, ...]:
        return tuple(item for item in self.items if item.is_labeled)

    @property
    def unlabeled_items(self) -> tuple[SegmentationDatasetItem, ...]:
        return tuple(item for item in self.items if not item.is_labeled)

    @property
    def valid_labeled_items(self) -> tuple[SegmentationDatasetItem, ...]:
        return tuple(item for item in self.labeled_items if item.is_valid and item.polygons)

    def summary(self) -> dict[str, Any]:
        annotation_format_counts = Counter()
        for item in self.valid_labeled_items:
            annotation_format_counts.update(item.annotation_format_counts)

        return {
            "dataset_dir": str(self.dataset_dir),
            "label_dir": str(self.label_dir),
            "class_name": self.class_name,
            "image_count": len(self.items),
            "labeled_image_count": len(self.labeled_items),
            "unlabeled_image_count": len(self.unlabeled_items),
            "valid_labeled_image_count": len(self.valid_labeled_items),
            "polygon_count": sum(item.polygon_count for item in self.valid_labeled_items),
            "point_count": sum(item.total_point_count for item in self.valid_labeled_items),
            "orphan_label_count": len(self.orphan_label_paths),
            "duplicate_label_count": len(self.duplicate_label_paths),
            "json_annotation_count": len(self.json_annotation_paths),
            "issue_count": len(self.issues),
            "annotation_format_counts": dict(sorted(annotation_format_counts.items())),
            "unlabeled_image_ids": [item.image_id for item in self.unlabeled_items],
            "issues": [issue.to_dict() for issue in self.issues[:20]],
        }


@dataclass(frozen=True)
class LegacySegmentationSplitReference:
    root_dir: Path
    train_image_ids: tuple[str, ...]
    val_image_ids: tuple[str, ...]
    duplicate_alias_names: tuple[str, ...]
    conflicting_image_ids: tuple[str, ...]

    @property
    def assigned_image_ids(self) -> tuple[str, ...]:
        assigned = set(self.train_image_ids) | set(self.val_image_ids)
        return tuple(sorted(assigned))

    @property
    def split_by_image_id(self) -> dict[str, str]:
        mapping = {image_id: "train" for image_id in self.train_image_ids}
        mapping.update({image_id: "val" for image_id in self.val_image_ids})
        return mapping

    def summary(self) -> dict[str, Any]:
        return {
            "root_dir": str(self.root_dir),
            "train_count": len(self.train_image_ids),
            "val_count": len(self.val_image_ids),
            "assigned_count": len(self.assigned_image_ids),
            "duplicate_alias_count": len(self.duplicate_alias_names),
            "duplicate_alias_names": list(self.duplicate_alias_names[:20]),
            "conflicting_image_count": len(self.conflicting_image_ids),
            "conflicting_image_ids": list(self.conflicting_image_ids),
        }


@dataclass(frozen=True)
class PreparedSegmentationDataset:
    root_dir: Path
    data_yaml_path: Path
    manifest_path: Path
    train_count: int
    val_count: int
    unassigned_count: int
    split_strategy: str

    def summary(self) -> dict[str, Any]:
        return {
            "root_dir": str(self.root_dir),
            "data_yaml_path": str(self.data_yaml_path),
            "manifest_path": str(self.manifest_path),
            "train_count": self.train_count,
            "val_count": self.val_count,
            "unassigned_count": self.unassigned_count,
            "split_strategy": self.split_strategy,
        }


@dataclass(frozen=True)
class PreparedSegmentationLabelPair:
    image_path: Path
    label_path: Path
    split: str


def parse_yolo_segmentation_label(
    label_path: Path,
) -> tuple[tuple[SegmentationPolygon, ...], tuple[SegmentationDatasetIssue, ...]]:
    polygons: list[SegmentationPolygon] = []
    issues: list[SegmentationDatasetIssue] = []

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) == 5:
            try:
                class_id = int(parts[0])
                center_x, center_y, width, height = (float(value) for value in parts[1:])
            except ValueError:
                issues.append(
                    SegmentationDatasetIssue(
                        path=label_path,
                        message=f"line {line_number}: non-numeric segmentation annotation values",
                    )
                )
                continue

            if class_id != 0:
                issues.append(
                    SegmentationDatasetIssue(
                        path=label_path,
                        message=f"line {line_number}: expected class id 0, found {class_id}",
                    )
                )

            if width <= 0 or height <= 0:
                issues.append(
                    SegmentationDatasetIssue(
                        path=label_path,
                        message=f"line {line_number}: bbox width and height must be positive",
                    )
                )
                continue

            min_x = center_x - (width / 2.0)
            max_x = center_x + (width / 2.0)
            min_y = center_y - (height / 2.0)
            max_y = center_y + (height / 2.0)
            points = (
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
            )

            for point_index, (x, y) in enumerate(points, start=1):
                if not (-BOUNDARY_EPSILON) <= x <= (1.0 + BOUNDARY_EPSILON):
                    issues.append(
                        SegmentationDatasetIssue(
                            path=label_path,
                            message=f"line {line_number}: point {point_index} x must be within [0, 1]",
                        )
                    )
                if not (-BOUNDARY_EPSILON) <= y <= (1.0 + BOUNDARY_EPSILON):
                    issues.append(
                        SegmentationDatasetIssue(
                            path=label_path,
                            message=f"line {line_number}: point {point_index} y must be within [0, 1]",
                        )
                    )

            polygons.append(
                SegmentationPolygon(
                    class_id=class_id,
                    points=points,
                    source_format="bbox",
                )
            )
            continue

        if len(parts) < 7:
            issues.append(
                SegmentationDatasetIssue(
                    path=label_path,
                    message=(
                        f"line {line_number}: expected 5 (bbox) or at least 7 (polygon) tokens, "
                        f"found {len(parts)}"
                    ),
                )
            )
            continue

        if (len(parts) - 1) % 2 != 0:
            issues.append(
                SegmentationDatasetIssue(
                    path=label_path,
                    message=f"line {line_number}: polygon coordinate count must be even",
                )
            )
            continue

        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError:
            issues.append(
                SegmentationDatasetIssue(
                    path=label_path,
                    message=f"line {line_number}: non-numeric segmentation annotation values",
                )
            )
            continue

        if class_id != 0:
            issues.append(
                SegmentationDatasetIssue(
                    path=label_path,
                    message=f"line {line_number}: expected class id 0, found {class_id}",
                )
            )

        points = tuple((coordinates[index], coordinates[index + 1]) for index in range(0, len(coordinates), 2))
        if len(points) < 3:
            issues.append(
                SegmentationDatasetIssue(
                    path=label_path,
                    message=f"line {line_number}: polygons require at least 3 points",
                )
            )

        for point_index, (x, y) in enumerate(points, start=1):
            if not (-BOUNDARY_EPSILON) <= x <= (1.0 + BOUNDARY_EPSILON):
                issues.append(
                    SegmentationDatasetIssue(
                        path=label_path,
                        message=f"line {line_number}: point {point_index} x must be within [0, 1]",
                    )
                )
            if not (-BOUNDARY_EPSILON) <= y <= (1.0 + BOUNDARY_EPSILON):
                issues.append(
                    SegmentationDatasetIssue(
                        path=label_path,
                        message=f"line {line_number}: point {point_index} y must be within [0, 1]",
                    )
                )

        polygons.append(
            SegmentationPolygon(
                class_id=class_id,
                points=points,
                source_format="polygon",
            )
        )

    if not polygons and not issues:
        issues.append(SegmentationDatasetIssue(path=label_path, message="label file is empty"))

    return tuple(polygons), tuple(issues)


def build_segmentation_dataset_inventory(
    dataset_dir: Path,
    label_dir: Path | None = None,
    class_name: str = "coral",
) -> SegmentationDatasetInventory:
    label_dir = label_dir or dataset_dir
    image_paths = sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix in IMAGE_SUFFIXES
    )
    label_candidates = sorted(path for path in label_dir.rglob("*.txt") if path.is_file())
    label_paths: dict[str, Path] = {}
    duplicate_label_paths: list[Path] = []
    for path in label_candidates:
        if path.stem in label_paths:
            duplicate_label_paths.append(path)
            continue
        label_paths[path.stem] = path
    json_paths = tuple(sorted(dataset_dir.glob("*.json")))
    image_stems = {path.stem for path in image_paths}
    orphan_label_paths = tuple(sorted(path for stem, path in label_paths.items() if stem not in image_stems))

    items: list[SegmentationDatasetItem] = []
    issues: list[SegmentationDatasetIssue] = []

    for image_path in image_paths:
        label_path = label_paths.get(image_path.stem)
        polygons: tuple[SegmentationPolygon, ...] = ()
        item_issues: tuple[SegmentationDatasetIssue, ...] = ()

        if label_path is not None:
            polygons, item_issues = parse_yolo_segmentation_label(label_path)
            issues.extend(item_issues)

        items.append(
            SegmentationDatasetItem(
                image_path=image_path,
                label_path=label_path,
                polygons=polygons,
                issues=item_issues,
            )
        )

    for label_path in orphan_label_paths:
        issues.append(
            SegmentationDatasetIssue(
                path=label_path,
                message="label does not have a matching image",
            )
        )
    for label_path in duplicate_label_paths:
        issues.append(
            SegmentationDatasetIssue(
                path=label_path,
                message="duplicate label stem detected; later file ignored",
            )
        )

    return SegmentationDatasetInventory(
        dataset_dir=dataset_dir,
        label_dir=label_dir,
        class_name=class_name,
        items=tuple(items),
        orphan_label_paths=orphan_label_paths,
        duplicate_label_paths=tuple(sorted(duplicate_label_paths)),
        json_annotation_paths=json_paths,
        issues=tuple(issues),
    )


def load_legacy_segmentation_split_reference(reference_dir: Path) -> LegacySegmentationSplitReference:
    images_root = reference_dir / "images"
    split_mapping: dict[str, str] = {}
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    duplicate_alias_names: list[str] = []
    conflicting_image_ids: set[str] = set()

    for split_name, target_set in (("train", train_ids), ("val", val_ids)):
        split_dir = images_root / split_name
        if not split_dir.exists():
            continue

        for entry in sorted(split_dir.iterdir()):
            if not entry.is_file() and not entry.is_symlink():
                continue
            if entry.suffix not in IMAGE_SUFFIXES:
                continue

            resolved = entry.resolve()
            image_id = resolved.stem if resolved.suffix in IMAGE_SUFFIXES else entry.stem
            if entry.stem != image_id:
                duplicate_alias_names.append(entry.name)

            existing_split = split_mapping.get(image_id)
            if existing_split is not None and existing_split != split_name:
                conflicting_image_ids.add(image_id)
                continue
            if existing_split is not None:
                continue

            split_mapping[image_id] = split_name
            target_set.add(image_id)

    for image_id in conflicting_image_ids:
        train_ids.discard(image_id)
        val_ids.discard(image_id)

    return LegacySegmentationSplitReference(
        root_dir=reference_dir.resolve(),
        train_image_ids=tuple(sorted(train_ids)),
        val_image_ids=tuple(sorted(val_ids)),
        duplicate_alias_names=tuple(sorted(set(duplicate_alias_names))),
        conflicting_image_ids=tuple(sorted(conflicting_image_ids)),
    )


def build_segmentation_inventory_report(
    inventory: SegmentationDatasetInventory,
    legacy_split_reference: LegacySegmentationSplitReference | None = None,
) -> dict[str, Any]:
    report = inventory.summary()
    if legacy_split_reference is None:
        return report

    valid_image_ids = {item.image_id for item in inventory.valid_labeled_items}
    assigned_image_ids = set(legacy_split_reference.assigned_image_ids)
    report["legacy_split"] = legacy_split_reference.summary()
    report["legacy_split"]["matched_valid_image_count"] = len(valid_image_ids & assigned_image_ids)
    report["legacy_split"]["missing_valid_image_count"] = len(valid_image_ids - assigned_image_ids)
    report["legacy_split"]["split_only_image_count"] = len(assigned_image_ids - valid_image_ids)
    report["legacy_split"]["missing_valid_image_ids"] = sorted(valid_image_ids - assigned_image_ids)[:20]
    report["legacy_split"]["split_only_image_ids"] = sorted(assigned_image_ids - valid_image_ids)[:20]
    return report


def _ensure_empty_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _link_or_copy(source: Path, destination: Path, use_symlinks: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if use_symlinks:
        try:
            destination.symlink_to(source)
            return
        except OSError:
            pass

    try:
        os.link(source, destination)
        return
    except OSError:
        pass

    shutil.copy2(source, destination)


def _write_segmentation_label_file(destination: Path, polygons: tuple[SegmentationPolygon, ...]) -> None:
    rows = [polygon.as_yolo_row() for polygon in polygons]
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _split_items_randomly(
    items: tuple[SegmentationDatasetItem, ...],
    val_split: float,
    seed: int,
) -> tuple[list[SegmentationDatasetItem], list[SegmentationDatasetItem]]:
    ordered = list(items)
    random.Random(seed).shuffle(ordered)

    if not ordered:
        return [], []
    if len(ordered) == 1:
        return ordered, []

    val_count = max(1, int(len(ordered) * val_split))
    val_count = min(val_count, len(ordered) - 1)
    return ordered[val_count:], ordered[:val_count]


def prepare_segmentation_dataset(
    inventory: SegmentationDatasetInventory,
    output_dir: Path,
    class_name: str,
    split_strategy: str,
    use_symlinks: bool,
    val_split: float,
    seed: int,
    legacy_split_reference: LegacySegmentationSplitReference | None = None,
) -> PreparedSegmentationDataset:
    dataset_root = output_dir.resolve()
    _ensure_empty_directory(dataset_root)

    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    for split in ("train", "val"):
        (images_root / split).mkdir(parents=True, exist_ok=True)
        (labels_root / split).mkdir(parents=True, exist_ok=True)

    selected_items = inventory.valid_labeled_items
    if not selected_items:
        raise ValueError("No valid Phase 3 dataset items are available for preparation.")

    unassigned_image_ids: list[str] = []
    if split_strategy == "legacy":
        if legacy_split_reference is None:
            raise ValueError("legacy_split_reference is required when split_strategy='legacy'")

        split_by_image_id = legacy_split_reference.split_by_image_id
        conflicting_image_ids = set(legacy_split_reference.conflicting_image_ids)
        train_items: list[SegmentationDatasetItem] = []
        val_items: list[SegmentationDatasetItem] = []

        for item in selected_items:
            if item.image_id in conflicting_image_ids:
                unassigned_image_ids.append(item.image_id)
                continue

            split_name = split_by_image_id.get(item.image_id)
            if split_name == "train":
                train_items.append(item)
            elif split_name == "val":
                val_items.append(item)
            else:
                unassigned_image_ids.append(item.image_id)
    else:
        train_items, val_items = _split_items_randomly(
            selected_items,
            val_split=val_split,
            seed=seed,
        )

    if not train_items:
        raise ValueError("Phase 3 preparation produced no training items.")
    if split_strategy == "random" and not val_items:
        raise ValueError("Phase 3 preparation produced no validation items.")

    print(
        "Preparing Phase 3 dataset "
        f"(train={len(train_items)}, val={len(val_items)}, split_strategy={split_strategy}, "
        f"unassigned={len(unassigned_image_ids)}, symlinks={use_symlinks})..."
    )

    for split_name, split_items in (("train", train_items), ("val", val_items)):
        for index, item in enumerate(split_items, start=1):
            destination_image = images_root / split_name / item.image_path.name
            destination_label = labels_root / split_name / f"{item.image_id}.txt"
            _link_or_copy(item.image_path, destination_image, use_symlinks=use_symlinks)
            _write_segmentation_label_file(destination_label, item.polygons)

            if index % 25 == 0 or index == len(split_items):
                print(f"  {split_name}: prepared {index}/{len(split_items)}")

    data_yaml_path = dataset_root / "data.yaml"
    data_yaml = {
        "path": str(dataset_root),
        "train": str((images_root / "train").resolve()),
        "val": str((images_root / "val").resolve()),
        "names": {0: class_name},
    }
    data_yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    manifest_path = dataset_root / "manifest.json"
    manifest = {
        "dataset_dir": str(inventory.dataset_dir),
        "class_name": class_name,
        "split_strategy": split_strategy,
        "seed": seed,
        "val_split": val_split,
        "use_symlinks": use_symlinks,
        "source_summary": inventory.summary(),
        "legacy_split_summary": legacy_split_reference.summary() if legacy_split_reference is not None else None,
        "train_image_ids": [item.image_id for item in train_items],
        "val_image_ids": [item.image_id for item in val_items],
        "unassigned_image_ids": sorted(unassigned_image_ids),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return PreparedSegmentationDataset(
        root_dir=dataset_root,
        data_yaml_path=data_yaml_path,
        manifest_path=manifest_path,
        train_count=len(train_items),
        val_count=len(val_items),
        unassigned_count=len(unassigned_image_ids),
        split_strategy=split_strategy,
    )


def _collect_prepared_segmentation_label_pairs(
    prepared_dataset: PreparedSegmentationDataset,
    split: str,
) -> list[PreparedSegmentationLabelPair]:
    image_dir = prepared_dataset.root_dir / "images" / split
    label_dir = prepared_dataset.root_dir / "labels" / split
    pairs: list[PreparedSegmentationLabelPair] = []

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(
                f"Prepared segmentation dataset is missing label for {image_path.name}: {label_path}"
            )
        pairs.append(
            PreparedSegmentationLabelPair(
                image_path=image_path,
                label_path=label_path,
                split=split,
            )
        )

    return pairs


def prewarm_segmentation_dataset(
    prepared_dataset: PreparedSegmentationDataset,
    num_classes: int,
    drop_corrupt_samples: bool,
    sample_timeout_seconds: int,
    log_interval: int,
) -> None:
    from ultralytics.data.utils import verify_image_label

    def _timeout_handler(signum, frame):
        raise TimeoutError("sample verification timed out")

    for split in ("train", "val"):
        pairs = _collect_prepared_segmentation_label_pairs(prepared_dataset, split)
        if not pairs:
            continue

        print(f"Prewarming {split} segmentation dataset verification ({len(pairs)} images)...")
        start = perf_counter()
        warnings: list[str] = []
        removed_pairs: list[PreparedSegmentationLabelPair] = []

        for index, pair in enumerate(pairs, start=1):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(sample_timeout_seconds)
            try:
                _, _, _, _, _, _, _, _, corrupt_count, message = verify_image_label(
                    (
                        str(pair.image_path),
                        str(pair.label_path),
                        "",
                        False,
                        num_classes,
                        0,
                        0,
                        False,
                    )
                )
            except TimeoutError:
                corrupt_count = 1
                message = f"{pair.image_path}: verification exceeded {sample_timeout_seconds}s timeout"
            finally:
                signal.alarm(0)

            if corrupt_count:
                details = message or f"{pair.image_path} failed verification"
                if drop_corrupt_samples:
                    pair.image_path.unlink(missing_ok=True)
                    pair.label_path.unlink(missing_ok=True)
                    removed_pairs.append(pair)
                    print(f"  {split}: removed corrupt sample {pair.image_path.name}")
                    print(f"    reason: {details}")
                else:
                    warnings.append(details)
            elif message:
                warnings.append(message)

            if index % log_interval == 0 or index == len(pairs):
                elapsed = perf_counter() - start
                print(f"  {split}: verified {index}/{len(pairs)} in {elapsed:.1f}s")

        if removed_pairs:
            print(f"  {split}: removed {len(removed_pairs)} corrupt samples before training")
        if warnings:
            sample = "\n".join(f"- {warning}" for warning in warnings[:10])
            print(f"  {split}: warnings during verification:\n{sample}")


class SegmentationTrainer:
    def __init__(
        self,
        backbone_path: Path,
        training_dir: Path,
        run_name: str,
        image_size: int,
        batch_size: int,
        epochs: int,
        patience: int,
        seed: int,
        augment: bool,
        workers: int,
        plots: bool,
        prewarm_font_cache: bool,
        prewarm_dataset: bool,
        drop_corrupt_samples: bool,
        sample_timeout_seconds: int,
        warmup_log_interval: int,
        device: str,
    ) -> None:
        self.backbone_path = backbone_path
        self.training_dir = training_dir
        self.run_name = run_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.augment = augment
        self.workers = workers
        self.plots = plots
        self.prewarm_font_cache = prewarm_font_cache
        self.prewarm_dataset = prewarm_dataset
        self.drop_corrupt_samples = drop_corrupt_samples
        self.sample_timeout_seconds = sample_timeout_seconds
        self.warmup_log_interval = warmup_log_interval
        self.device = device

    def run(self, prepared_dataset: PreparedSegmentationDataset) -> Path:
        from ultralytics import YOLO

        from coral_thesis.phases.chart_detection import prewarm_ultralytics_runtime

        if not self.backbone_path.exists():
            raise FileNotFoundError(f"Segmentation backbone not found: {self.backbone_path}")

        self.training_dir.mkdir(parents=True, exist_ok=True)
        matplotlib_dir = self.training_dir / ".matplotlib"
        matplotlib_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(matplotlib_dir.resolve())

        if self.prewarm_font_cache:
            prewarm_ultralytics_runtime(matplotlib_dir)
        if self.prewarm_dataset:
            prewarm_segmentation_dataset(
                prepared_dataset=prepared_dataset,
                num_classes=1,
                drop_corrupt_samples=self.drop_corrupt_samples,
                sample_timeout_seconds=self.sample_timeout_seconds,
                log_interval=self.warmup_log_interval,
            )

        model = YOLO(str(self.backbone_path))
        model.train(
            data=str(prepared_dataset.data_yaml_path),
            epochs=self.epochs,
            imgsz=self.image_size,
            patience=self.patience,
            batch=self.batch_size,
            seed=self.seed,
            project=str(self.training_dir),
            name=self.run_name,
            exist_ok=True,
            augment=self.augment,
            workers=self.workers,
            plots=self.plots,
            device=self.device,
        )
        return self.training_dir / self.run_name


def evaluate_segmentation_model(
    model_path: Path,
    prepared_dataset: PreparedSegmentationDataset,
    reports_dir: Path,
    report_name: str,
    image_size: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    from coral_thesis.phases.chart_detection import prewarm_ultralytics_runtime

    if not model_path.exists():
        raise FileNotFoundError(f"Segmentation weights not found: {model_path}")

    reports_dir = reports_dir.resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir = reports_dir / ".matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_dir)
    prewarm_ultralytics_runtime(matplotlib_dir)

    ultralytics_project_dir = reports_dir / "ultralytics"
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(prepared_dataset.data_yaml_path),
        imgsz=image_size,
        batch=batch_size,
        workers=0,
        device=device,
        split="val",
        plots=False,
        project=str(ultralytics_project_dir),
        name=report_name,
    )

    manifest_data = json.loads(prepared_dataset.manifest_path.read_text(encoding="utf-8"))
    serialized_results = {key: float(value) for key, value in metrics.results_dict.items()}
    speed = {
        key: float(value)
        for key, value in getattr(metrics, "speed", {}).items()
    }
    save_dir = Path(getattr(metrics, "save_dir", ultralytics_project_dir / report_name))
    report = {
        "report_name": report_name,
        "weights_path": str(model_path.resolve()),
        "prepared_dataset": prepared_dataset.summary(),
        "prepared_manifest_path": str(prepared_dataset.manifest_path),
        "source_summary": manifest_data.get("source_summary"),
        "legacy_split_summary": manifest_data.get("legacy_split_summary"),
        "results_dict": serialized_results,
        "box_metrics": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        },
        "mask_metrics": {
            "precision": float(metrics.seg.mp),
            "recall": float(metrics.seg.mr),
            "map50": float(metrics.seg.map50),
            "map50_95": float(metrics.seg.map),
        },
        "speed_ms_per_image": speed,
        "ultralytics_save_dir": str(save_dir),
    }

    report_path = reports_dir / f"{report_name}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _resolve_segmentation_result_source_path(
    source_paths: Sequence[Path],
    result_index: int,
    fallback_result_path: str | os.PathLike[str] | None,
) -> Path:
    if result_index < len(source_paths):
        return source_paths[result_index]
    if fallback_result_path is None:
        raise IndexError(
            "Segmentation inference returned more results than the number of provided source images."
        )
    return Path(fallback_result_path)


class CoralSegmentationPhase:
    phase_name = "coral-segmentation"

    def __init__(self, model_path: Path, confidence_threshold: float = 0.25) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

    def run_from_source(self, source: Path, output_dir: Path) -> list[SegmentationResult]:
        if source.is_dir():
            source_paths = sorted(
                path
                for path in source.iterdir()
                if path.is_file() and path.suffix in IMAGE_SUFFIXES
            )
        else:
            source_paths = [source]
        return self.run(source_paths=source_paths, output_dir=output_dir)

    def run(self, source_paths: Sequence[Path], output_dir: Path) -> list[SegmentationResult]:
        from ultralytics import YOLO

        if not self.model_path.exists():
            raise FileNotFoundError(f"Coral segmentation weights not found: {self.model_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = output_dir / "masks"
        masked_dir = output_dir / "masked_images"
        _ensure_empty_directory(masks_dir)
        _ensure_empty_directory(masked_dir)

        model = YOLO(str(self.model_path))
        results = model(
            [str(path) for path in source_paths],
            stream=True,
            conf=self.confidence_threshold,
            task="segment",
        )

        phase_results: list[SegmentationResult] = []
        for result_index, result in enumerate(results):
            source_image_path = _resolve_segmentation_result_source_path(
                source_paths=source_paths,
                result_index=result_index,
                fallback_result_path=getattr(result, "path", None),
            )
            if result.masks is None:
                continue

            image_id = source_image_path.stem
            full_mask = np.zeros(result.orig_shape[:2], dtype=np.uint8)

            for mask in result.masks.data.cpu().numpy():
                resized = cv2.resize(mask, (result.orig_shape[1], result.orig_shape[0]))
                full_mask = np.maximum(full_mask, (resized > 0.5).astype(np.uint8) * 255)

            mask_path = masks_dir / f"{image_id}.png"
            masked_image_path = masked_dir / f"{image_id}.png"
            cv2.imwrite(str(mask_path), full_mask)
            masked_image = cv2.bitwise_and(result.orig_img, result.orig_img, mask=full_mask)
            cv2.imwrite(str(masked_image_path), masked_image)

            phase_results.append(
                SegmentationResult(
                    image_id=image_id,
                    source_image_path=source_image_path,
                    mask_path=mask_path,
                    masked_image_path=masked_image_path,
                )
            )

        return sorted(phase_results, key=lambda item: item.image_id)
