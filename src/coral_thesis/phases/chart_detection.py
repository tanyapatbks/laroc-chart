from __future__ import annotations

import json
import os
import random
import signal
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import cv2
import yaml

from coral_thesis.config import IMAGE_SUFFIXES
from coral_thesis.domain import DetectionResult

BOUNDARY_EPSILON = 1e-6


@dataclass(frozen=True)
class ChartAnnotation:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def as_yolo_row(self) -> str:
        return (
            f"{self.class_id} "
            f"{self.x_center:.6f} "
            f"{self.y_center:.6f} "
            f"{self.width:.6f} "
            f"{self.height:.6f}"
        )


@dataclass(frozen=True)
class ChartDatasetIssue:
    path: Path
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "message": self.message}


@dataclass(frozen=True)
class ChartDatasetItem:
    image_path: Path
    label_path: Path | None
    annotations: tuple[ChartAnnotation, ...]
    issues: tuple[ChartDatasetIssue, ...]

    @property
    def image_id(self) -> str:
        return self.image_path.stem

    @property
    def is_labeled(self) -> bool:
        return self.label_path is not None

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0


@dataclass(frozen=True)
class ChartDatasetInventory:
    dataset_dir: Path
    class_name: str
    items: tuple[ChartDatasetItem, ...]
    orphan_label_paths: tuple[Path, ...]
    json_annotation_paths: tuple[Path, ...]
    issues: tuple[ChartDatasetIssue, ...]

    @property
    def labeled_items(self) -> tuple[ChartDatasetItem, ...]:
        return tuple(item for item in self.items if item.is_labeled)

    @property
    def unlabeled_items(self) -> tuple[ChartDatasetItem, ...]:
        return tuple(item for item in self.items if not item.is_labeled)

    @property
    def valid_labeled_items(self) -> tuple[ChartDatasetItem, ...]:
        return tuple(item for item in self.labeled_items if item.is_valid and item.annotations)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "class_name": self.class_name,
            "image_count": len(self.items),
            "labeled_image_count": len(self.labeled_items),
            "unlabeled_image_count": len(self.unlabeled_items),
            "valid_labeled_image_count": len(self.valid_labeled_items),
            "orphan_label_count": len(self.orphan_label_paths),
            "json_annotation_count": len(self.json_annotation_paths),
            "issue_count": len(self.issues),
            "unlabeled_image_ids": [item.image_id for item in self.unlabeled_items],
            "issues": [issue.to_dict() for issue in self.issues[:20]],
        }


@dataclass(frozen=True)
class PreparedChartDataset:
    root_dir: Path
    data_yaml_path: Path
    manifest_path: Path
    train_count: int
    val_count: int
    skipped_unlabeled_count: int

    def summary(self) -> dict[str, Any]:
        return {
            "root_dir": str(self.root_dir),
            "data_yaml_path": str(self.data_yaml_path),
            "manifest_path": str(self.manifest_path),
            "train_count": self.train_count,
            "val_count": self.val_count,
            "skipped_unlabeled_count": self.skipped_unlabeled_count,
        }


@dataclass(frozen=True)
class PreparedLabelPair:
    image_path: Path
    label_path: Path
    split: str


def parse_yolo_detection_label(label_path: Path) -> tuple[tuple[ChartAnnotation, ...], tuple[ChartDatasetIssue, ...]]:
    annotations: list[ChartAnnotation] = []
    issues: list[ChartDatasetIssue] = []

    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            issues.append(
                ChartDatasetIssue(
                    path=label_path,
                    message=f"line {line_number}: expected 5 tokens, found {len(parts)}",
                )
            )
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError:
            issues.append(
                ChartDatasetIssue(
                    path=label_path,
                    message=f"line {line_number}: non-numeric YOLO annotation values",
                )
            )
            continue

        if class_id != 0:
            issues.append(
                ChartDatasetIssue(
                    path=label_path,
                    message=f"line {line_number}: expected class id 0, found {class_id}",
                )
            )
        for field_name, value in (
            ("x_center", x_center),
            ("y_center", y_center),
            ("width", width),
            ("height", height),
        ):
            if not (-BOUNDARY_EPSILON) <= value <= (1.0 + BOUNDARY_EPSILON):
                issues.append(
                    ChartDatasetIssue(
                        path=label_path,
                        message=f"line {line_number}: {field_name} must be within [0, 1]",
                    )
                )

        if width <= 0 or height <= 0:
            issues.append(
                ChartDatasetIssue(
                    path=label_path,
                    message=f"line {line_number}: width and height must be positive",
                )
            )

        if (
            x_center - (width / 2) < -BOUNDARY_EPSILON
            or x_center + (width / 2) > 1.0 + BOUNDARY_EPSILON
        ):
            issues.append(
                ChartDatasetIssue(
                    path=label_path,
                    message=f"line {line_number}: x bounds exceed image extent",
                )
            )
        if (
            y_center - (height / 2) < -BOUNDARY_EPSILON
            or y_center + (height / 2) > 1.0 + BOUNDARY_EPSILON
        ):
            issues.append(
                ChartDatasetIssue(
                    path=label_path,
                    message=f"line {line_number}: y bounds exceed image extent",
                )
            )

        annotations.append(
            ChartAnnotation(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )

    if not annotations and not issues:
        issues.append(
            ChartDatasetIssue(
                path=label_path,
                message="label file is empty",
            )
        )

    return tuple(annotations), tuple(issues)


def build_chart_dataset_inventory(dataset_dir: Path, class_name: str = "chart") -> ChartDatasetInventory:
    image_paths = sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix in IMAGE_SUFFIXES
    )
    label_paths = {path.stem: path for path in dataset_dir.glob("*.txt")}
    json_paths = tuple(sorted(dataset_dir.glob("*.json")))
    image_stems = {path.stem for path in image_paths}
    orphan_label_paths = tuple(sorted(path for stem, path in label_paths.items() if stem not in image_stems))

    items: list[ChartDatasetItem] = []
    issues: list[ChartDatasetIssue] = []

    for image_path in image_paths:
        label_path = label_paths.get(image_path.stem)
        annotations: tuple[ChartAnnotation, ...] = ()
        item_issues: tuple[ChartDatasetIssue, ...] = ()

        if label_path is not None:
            annotations, item_issues = parse_yolo_detection_label(label_path)
            issues.extend(item_issues)

        items.append(
            ChartDatasetItem(
                image_path=image_path,
                label_path=label_path,
                annotations=annotations,
                issues=item_issues,
            )
        )

    for label_path in orphan_label_paths:
        issues.append(
            ChartDatasetIssue(
                path=label_path,
                message="label does not have a matching image",
            )
        )

    return ChartDatasetInventory(
        dataset_dir=dataset_dir,
        class_name=class_name,
        items=tuple(items),
        orphan_label_paths=orphan_label_paths,
        json_annotation_paths=json_paths,
        issues=tuple(issues),
    )


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


def _write_label_file(destination: Path, annotations: tuple[ChartAnnotation, ...]) -> None:
    rows = [annotation.as_yolo_row() for annotation in annotations]
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _split_items(items: tuple[ChartDatasetItem, ...], val_split: float, seed: int) -> tuple[list[ChartDatasetItem], list[ChartDatasetItem]]:
    ordered = list(items)
    random.Random(seed).shuffle(ordered)

    if not ordered:
        return [], []

    if len(ordered) == 1:
        return ordered, []

    val_count = max(1, int(len(ordered) * val_split))
    val_count = min(val_count, len(ordered) - 1)

    val_items = ordered[:val_count]
    train_items = ordered[val_count:]
    return train_items, val_items


def prepare_chart_detection_dataset(
    inventory: ChartDatasetInventory,
    output_dir: Path,
    class_name: str,
    val_split: float,
    seed: int,
    excluded_image_ids: Sequence[str],
    use_symlinks: bool,
    skip_unlabeled_images: bool,
) -> PreparedChartDataset:
    dataset_root = output_dir.resolve()
    _ensure_empty_directory(dataset_root)

    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    for split in ("train", "val"):
        (images_root / split).mkdir(parents=True, exist_ok=True)
        (labels_root / split).mkdir(parents=True, exist_ok=True)

    if skip_unlabeled_images:
        selected_items = inventory.valid_labeled_items
    else:
        selected_items = tuple(item for item in inventory.items if item.is_valid)

    excluded_ids = set(excluded_image_ids)
    if excluded_ids:
        selected_items = tuple(item for item in selected_items if item.image_id not in excluded_ids)

    if not selected_items:
        raise ValueError("No valid Phase 1 dataset items are available for preparation.")

    train_items, val_items = _split_items(selected_items, val_split=val_split, seed=seed)
    print(
        "Preparing Phase 1 dataset "
        f"(train={len(train_items)}, val={len(val_items)}, "
        f"skip_unlabeled={skip_unlabeled_images}, excluded={len(excluded_ids)}, "
        f"symlinks={use_symlinks})..."
    )

    for split_name, split_items in (("train", train_items), ("val", val_items)):
        for index, item in enumerate(split_items, start=1):
            destination_image = images_root / split_name / item.image_path.name
            _link_or_copy(item.image_path, destination_image, use_symlinks=use_symlinks)

            if item.label_path is not None:
                destination_label = labels_root / split_name / item.label_path.name
                _write_label_file(destination_label, item.annotations)

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
        "seed": seed,
        "val_split": val_split,
        "skip_unlabeled_images": skip_unlabeled_images,
        "source_summary": inventory.summary(),
        "train_image_ids": [item.image_id for item in train_items],
        "val_image_ids": [item.image_id for item in val_items],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return PreparedChartDataset(
        root_dir=dataset_root,
        data_yaml_path=data_yaml_path,
        manifest_path=manifest_path,
        train_count=len(train_items),
        val_count=len(val_items),
        skipped_unlabeled_count=len(inventory.unlabeled_items) if skip_unlabeled_images else 0,
    )


def _collect_prepared_label_pairs(prepared_dataset: PreparedChartDataset, split: str) -> list[PreparedLabelPair]:
    image_dir = prepared_dataset.root_dir / "images" / split
    label_dir = prepared_dataset.root_dir / "labels" / split
    pairs: list[PreparedLabelPair] = []

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(
                f"Prepared dataset is missing label for {image_path.name}: {label_path}"
            )
        pairs.append(PreparedLabelPair(image_path=image_path, label_path=label_path, split=split))

    return pairs


def prewarm_ultralytics_runtime(cache_dir: Path) -> None:
    from matplotlib import font_manager
    from ultralytics.data import utils as data_utils
    from ultralytics.utils import checks as check_module

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir.resolve())

    print("Prewarming matplotlib font cache...")
    system_fonts = font_manager.findSystemFonts()
    if not system_fonts:
        raise RuntimeError("No system fonts were discovered; Ultralytics font initialization cannot proceed.")

    def _safe_check_font(font: str = "Arial.ttf"):
        target = Path(font).stem.lower()
        for candidate in system_fonts:
            if target in Path(candidate).stem.lower():
                return candidate
        return system_fonts[0]

    data_utils.check_font = _safe_check_font
    check_module.check_font = _safe_check_font
    print(f"Font cache ready with {len(system_fonts)} system fonts.")


def prewarm_chart_detection_dataset(
    prepared_dataset: PreparedChartDataset,
    num_classes: int,
    drop_corrupt_samples: bool,
    sample_timeout_seconds: int,
    max_workers: int,
    log_interval: int,
) -> None:
    from ultralytics.data.utils import verify_image_label

    def _timeout_handler(signum, frame):
        raise TimeoutError("sample verification timed out")

    for split in ("train", "val"):
        pairs = _collect_prepared_label_pairs(prepared_dataset, split)
        if not pairs:
            continue

        print(f"Prewarming {split} dataset verification ({len(pairs)} images)...")
        start = perf_counter()
        warnings: list[str] = []
        removed_pairs: list[PreparedLabelPair] = []

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


class ChartDetectionTrainer:
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
        warmup_workers: int,
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
        self.warmup_workers = warmup_workers
        self.warmup_log_interval = warmup_log_interval
        self.device = device

    def run(self, prepared_dataset: PreparedChartDataset) -> Path:
        from ultralytics import YOLO

        if not self.backbone_path.exists():
            raise FileNotFoundError(f"Detection backbone not found: {self.backbone_path}")

        self.training_dir.mkdir(parents=True, exist_ok=True)
        matplotlib_dir = self.training_dir / ".matplotlib"
        matplotlib_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(matplotlib_dir.resolve())

        if self.prewarm_font_cache:
            prewarm_ultralytics_runtime(matplotlib_dir)
        if self.prewarm_dataset:
            prewarm_chart_detection_dataset(
                prepared_dataset=prepared_dataset,
                num_classes=1,
                drop_corrupt_samples=self.drop_corrupt_samples,
                sample_timeout_seconds=self.sample_timeout_seconds,
                max_workers=self.warmup_workers,
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


class ChartDetectionPhase:
    phase_name = "chart-detection"

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.25,
        expected_class_name: str = "chart",
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.expected_class_name = expected_class_name

    def run_from_source(self, source: Path, output_dir: Path) -> list[DetectionResult]:
        if source.is_dir():
            source_paths = sorted(
                path
                for path in source.iterdir()
                if path.is_file() and path.suffix in IMAGE_SUFFIXES
            )
        else:
            source_paths = [source]
        return self.run(source_paths=source_paths, output_dir=output_dir)

    def run(self, source_paths: Sequence[Path], output_dir: Path) -> list[DetectionResult]:
        from ultralytics import YOLO

        if not self.model_path.exists():
            raise FileNotFoundError(f"Chart detector weights not found: {self.model_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        crops_dir = output_dir / "crops"
        viz_dir = output_dir / "viz"
        crops_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(str(self.model_path))
        collected: dict[str, dict[str, Path | list[Path]]] = defaultdict(
            lambda: {"crop_paths": [], "source_image_path": None, "visualization_path": None}
        )

        results = model(
            [str(path) for path in source_paths],
            stream=True,
            conf=self.confidence_threshold,
        )
        for result in results:
            image_path = Path(result.path)
            image_id = image_path.stem
            visualization_path = viz_dir / image_path.name
            result.save(filename=str(visualization_path))

            record = collected[image_id]
            record["source_image_path"] = image_path
            record["visualization_path"] = visualization_path

            for index, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name != self.expected_class_name:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = result.orig_img[y1:y2, x1:x2]
                crop_path = crops_dir / f"{image_id}_chart_{index}.jpg"
                cv2.imwrite(str(crop_path), crop)
                record["crop_paths"].append(crop_path)

        phase_results: list[DetectionResult] = []
        for image_id, record in collected.items():
            phase_results.append(
                DetectionResult(
                    image_id=image_id,
                    source_image_path=record["source_image_path"],
                    crop_paths=list(record["crop_paths"]),
                    visualization_path=record["visualization_path"],
                )
            )

        return sorted(phase_results, key=lambda item: item.image_id)
