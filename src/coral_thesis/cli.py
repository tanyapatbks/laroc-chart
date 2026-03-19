from __future__ import annotations

import argparse
import json
from pathlib import Path

from coral_thesis.config import load_config
from coral_thesis.phases.chart_detection import (
    ChartDetectionPhase,
    ChartDetectionTrainer,
    build_chart_dataset_inventory,
    prepare_chart_detection_dataset,
)
from coral_thesis.phases.color_calibration import ColorCalibrationPhase
from coral_thesis.phases.coral_segmentation import (
    build_segmentation_dataset_inventory,
    build_segmentation_inventory_report,
    load_legacy_segmentation_split_reference,
    prepare_segmentation_dataset,
)
from coral_thesis.pipeline import CoralPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="coral-thesis V2 command line tools")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to a YAML configuration file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show-config", help="Print the resolved configuration.")
    subparsers.add_parser("validate-config", help="Validate configuration paths and values.")
    subparsers.add_parser("bootstrap", help="Create workspace artifact directories.")
    subparsers.add_parser(
        "describe-pipeline",
        help="Print a high-level description of the current V2 pipeline.",
    )
    subparsers.add_parser(
        "phase1-inventory",
        help="Inspect raw Phase 1 annotations and print a dataset summary.",
    )
    subparsers.add_parser(
        "phase1-prepare",
        help="Prepare the Phase 1 YOLO dataset under the configured artifact path.",
    )
    subparsers.add_parser(
        "phase1-train",
        help="Prepare and train the Phase 1 chart detector.",
    )
    phase1_train = subparsers.choices["phase1-train"]
    phase1_train.add_argument("--epochs", type=int, help="Override training epochs for this run.")
    phase1_train.add_argument("--batch-size", type=int, help="Override batch size for this run.")
    phase1_train.add_argument("--image-size", type=int, help="Override image size for this run.")
    phase1_train.add_argument("--run-name", help="Override the training run name for this run.")
    phase1_infer = subparsers.add_parser(
        "phase1-infer",
        help="Run Phase 1 inference using configured chart detector weights.",
    )
    phase1_infer.add_argument(
        "--source",
        required=True,
        help="Path to a source image or directory to run inference on.",
    )
    subparsers.add_parser(
        "phase2-baseline",
        help="Analyze the baseline chart and save a reusable Phase 2 calibration profile.",
    )
    phase2_calibrate = subparsers.add_parser(
        "phase2-calibrate",
        help="Calibrate a source image using a detected chart crop.",
    )
    phase2_calibrate.add_argument("--source-image", required=True, help="Path to the raw source image.")
    phase2_calibrate.add_argument("--chart-crop", required=True, help="Path to the detected chart crop.")
    phase2_calibrate.add_argument(
        "--output-name",
        required=True,
        help="Filename stem for the calibrated output image inside the Phase 2 output directory.",
    )
    phase2_batch = subparsers.add_parser(
        "phase2-calibrate-batch",
        help="Calibrate many source images using a directory of detected chart crops.",
    )
    phase2_batch.add_argument("--source-dir", required=True, help="Directory containing raw source images.")
    phase2_batch.add_argument("--crops-dir", required=True, help="Directory containing chart crop images.")
    phase2_batch.add_argument(
        "--report-name",
        default="batch_report",
        help="Filename stem for the batch calibration report inside the Phase 2 output directory.",
    )
    phase2_batch.add_argument(
        "--crop-glob",
        default="*.jpg",
        help="Glob pattern used to select crop images inside the crops directory.",
    )
    phase2_batch.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of crop files to process.",
    )
    phase2_evaluate = subparsers.add_parser(
        "phase2-evaluate-manifest",
        help="Run Phase 2 evaluation from a curated manifest of source images and chart crops.",
    )
    phase2_evaluate.add_argument(
        "--manifest",
        help="Path to a Phase 2 evaluation manifest YAML file. Defaults to the configured manifest path.",
    )
    phase2_evaluate.add_argument(
        "--report-name",
        default="phase2_evaluation",
        help="Filename stem for the evaluation report inside the Phase 2 evaluation reports directory.",
    )
    subparsers.add_parser(
        "phase3-inventory",
        help="Inspect raw Phase 3 segmentation labels and legacy split metadata.",
    )
    subparsers.add_parser(
        "phase3-prepare",
        help="Prepare the Phase 3 YOLO segmentation dataset under the configured artifact path.",
    )
    return parser


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2))


def _resolve_chart_detector_weights(config) -> Path:
    if config.models.chart_detector_weights is not None:
        return config.models.chart_detector_weights

    candidate = (
        config.phase1.training_dir
        / config.phase1.train.run_name
        / "weights"
        / "best.pt"
    )
    if candidate.exists():
        return candidate

    raise RuntimeError(
        "No chart detector weights were found. "
        "Set models.chart_detector_weights or run `phase1-train` first."
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(Path(args.config))
    pipeline = CoralPipeline(config)

    if args.command == "show-config":
        _print_json(config.to_dict())
        return

    if args.command == "validate-config":
        issues = config.validate()
        if issues:
            print("Config validation failed:")
            for issue in issues:
                print(f"- {issue}")
        else:
            print("Config validation passed.")
        return

    if args.command == "bootstrap":
        pipeline.bootstrap()
        print(f"Workspace initialized at {config.project_root}")
        return

    if args.command == "describe-pipeline":
        print(pipeline.describe())
        return

    if args.command == "phase1-inventory":
        inventory = build_chart_dataset_inventory(
            dataset_dir=config.paths.dataset_dir,
            class_name=config.phase1.class_name,
        )
        _print_json(inventory.summary())
        return

    if args.command == "phase1-prepare":
        inventory = build_chart_dataset_inventory(
            dataset_dir=config.paths.dataset_dir,
            class_name=config.phase1.class_name,
        )
        prepared = prepare_chart_detection_dataset(
            inventory=inventory,
            output_dir=config.phase1.prepared_dataset_dir,
            class_name=config.phase1.class_name,
            val_split=config.phase1.val_split,
            seed=config.phase1.seed,
            excluded_image_ids=config.phase1.excluded_image_ids,
            use_symlinks=config.phase1.use_symlinks,
            skip_unlabeled_images=config.phase1.skip_unlabeled_images,
        )
        _print_json(prepared.summary())
        return

    if args.command == "phase1-train":
        inventory = build_chart_dataset_inventory(
            dataset_dir=config.paths.dataset_dir,
            class_name=config.phase1.class_name,
        )
        prepared = prepare_chart_detection_dataset(
            inventory=inventory,
            output_dir=config.phase1.prepared_dataset_dir,
            class_name=config.phase1.class_name,
            val_split=config.phase1.val_split,
            seed=config.phase1.seed,
            excluded_image_ids=config.phase1.excluded_image_ids,
            use_symlinks=config.phase1.use_symlinks,
            skip_unlabeled_images=config.phase1.skip_unlabeled_images,
        )
        trainer = ChartDetectionTrainer(
            backbone_path=config.models.detection_backbone,
            training_dir=config.phase1.training_dir,
            run_name=args.run_name or config.phase1.train.run_name,
            image_size=args.image_size or config.runtime.image_size,
            batch_size=args.batch_size or config.phase1.train.batch_size,
            epochs=args.epochs or config.phase1.train.epochs,
            patience=config.phase1.train.patience,
            seed=config.phase1.seed,
            augment=config.phase1.train.augment,
            workers=config.phase1.train.workers,
            plots=config.phase1.train.plots,
            prewarm_font_cache=config.phase1.train.prewarm_font_cache,
            prewarm_dataset=config.phase1.train.prewarm_dataset,
            drop_corrupt_samples=config.phase1.train.drop_corrupt_samples,
            sample_timeout_seconds=config.phase1.train.sample_timeout_seconds,
            warmup_workers=config.phase1.train.warmup_workers,
            warmup_log_interval=config.phase1.train.warmup_log_interval,
            device=config.runtime.device,
        )
        run_dir = trainer.run(prepared)
        print(f"Phase 1 training complete: {run_dir}")
        return

    if args.command == "phase1-infer":
        phase = ChartDetectionPhase(
            model_path=_resolve_chart_detector_weights(config),
            confidence_threshold=config.runtime.confidence_threshold,
            expected_class_name=config.phase1.class_name,
        )
        results = phase.run_from_source(
            source=Path(args.source),
            output_dir=config.phase1.inference_dir,
        )
        print(f"Processed {len(results)} source images.")
        return

    if args.command == "phase2-baseline":
        phase = ColorCalibrationPhase(
            baseline_chart_path=config.paths.baseline_chart_path,
            baseline_profile_path=config.phase2.baseline_profile_path,
            output_dir=config.phase2.output_dir,
            evaluation_reports_dir=config.phase2.evaluation_reports_dir,
            patch_rows=config.phase2.patch_rows,
            patch_cols=config.phase2.patch_cols,
            cell_sample_ratio=config.phase2.cell_sample_ratio,
            min_patch_count=config.phase2.min_patch_count,
            method=config.phase2.method,
            sample_timeout_seconds=config.phase2.sample_timeout_seconds,
        )
        profile = phase.build_and_save_baseline_profile()
        _print_json(profile.to_dict())
        return

    if args.command == "phase2-calibrate":
        phase = ColorCalibrationPhase(
            baseline_chart_path=config.paths.baseline_chart_path,
            baseline_profile_path=config.phase2.baseline_profile_path,
            output_dir=config.phase2.output_dir,
            evaluation_reports_dir=config.phase2.evaluation_reports_dir,
            patch_rows=config.phase2.patch_rows,
            patch_cols=config.phase2.patch_cols,
            cell_sample_ratio=config.phase2.cell_sample_ratio,
            min_patch_count=config.phase2.min_patch_count,
            method=config.phase2.method,
            sample_timeout_seconds=config.phase2.sample_timeout_seconds,
        )
        result = phase.calibrate_single(
            source_image_path=Path(args.source_image),
            chart_crop_path=Path(args.chart_crop),
            output_stem=args.output_name,
        )
        _print_json(result)
        return

    if args.command == "phase2-calibrate-batch":
        phase = ColorCalibrationPhase(
            baseline_chart_path=config.paths.baseline_chart_path,
            baseline_profile_path=config.phase2.baseline_profile_path,
            output_dir=config.phase2.output_dir,
            evaluation_reports_dir=config.phase2.evaluation_reports_dir,
            patch_rows=config.phase2.patch_rows,
            patch_cols=config.phase2.patch_cols,
            cell_sample_ratio=config.phase2.cell_sample_ratio,
            min_patch_count=config.phase2.min_patch_count,
            method=config.phase2.method,
            sample_timeout_seconds=config.phase2.sample_timeout_seconds,
        )
        report = phase.calibrate_batch(
            source_dir=Path(args.source_dir),
            crops_dir=Path(args.crops_dir),
            report_name=args.report_name,
            crop_glob=args.crop_glob,
            limit=args.limit,
        )
        _print_json(report)
        return

    if args.command == "phase2-evaluate-manifest":
        phase = ColorCalibrationPhase(
            baseline_chart_path=config.paths.baseline_chart_path,
            baseline_profile_path=config.phase2.baseline_profile_path,
            output_dir=config.phase2.output_dir,
            evaluation_reports_dir=config.phase2.evaluation_reports_dir,
            patch_rows=config.phase2.patch_rows,
            patch_cols=config.phase2.patch_cols,
            cell_sample_ratio=config.phase2.cell_sample_ratio,
            min_patch_count=config.phase2.min_patch_count,
            method=config.phase2.method,
            sample_timeout_seconds=config.phase2.sample_timeout_seconds,
        )
        report = phase.evaluate_manifest(
            manifest_path=Path(args.manifest) if args.manifest else config.phase2.evaluation_manifest_path,
            report_name=args.report_name,
        )
        _print_json(report)
        return

    if args.command == "phase3-inventory":
        inventory = build_segmentation_dataset_inventory(
            dataset_dir=config.paths.dataset_dir,
            label_dir=config.phase3.labels_dir,
            class_name=config.phase3.class_name,
        )
        legacy_reference = None
        if config.phase3.legacy_dataset_dir is not None and config.phase3.legacy_dataset_dir.exists():
            legacy_reference = load_legacy_segmentation_split_reference(config.phase3.legacy_dataset_dir)
        _print_json(build_segmentation_inventory_report(inventory, legacy_reference))
        return

    if args.command == "phase3-prepare":
        inventory = build_segmentation_dataset_inventory(
            dataset_dir=config.paths.dataset_dir,
            label_dir=config.phase3.labels_dir,
            class_name=config.phase3.class_name,
        )
        legacy_reference = None
        if config.phase3.split_strategy == "legacy":
            legacy_reference = load_legacy_segmentation_split_reference(config.phase3.legacy_dataset_dir)
        prepared = prepare_segmentation_dataset(
            inventory=inventory,
            output_dir=config.phase3.prepared_dataset_dir,
            class_name=config.phase3.class_name,
            split_strategy=config.phase3.split_strategy,
            use_symlinks=config.phase3.use_symlinks,
            val_split=config.phase3.val_split,
            seed=config.phase3.seed,
            legacy_split_reference=legacy_reference,
        )
        _print_json(prepared.summary())
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
