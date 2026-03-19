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
    phase1_infer = subparsers.add_parser(
        "phase1-infer",
        help="Run Phase 1 inference using configured chart detector weights.",
    )
    phase1_infer.add_argument(
        "--source",
        required=True,
        help="Path to a source image or directory to run inference on.",
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
            use_symlinks=config.phase1.use_symlinks,
            skip_unlabeled_images=config.phase1.skip_unlabeled_images,
        )
        trainer = ChartDetectionTrainer(
            backbone_path=config.models.detection_backbone,
            training_dir=config.phase1.training_dir,
            run_name=config.phase1.train.run_name,
            image_size=config.runtime.image_size,
            batch_size=config.phase1.train.batch_size,
            epochs=config.phase1.train.epochs,
            patience=config.phase1.train.patience,
            seed=config.phase1.seed,
            augment=config.phase1.train.augment,
            workers=config.phase1.train.workers,
            plots=config.phase1.train.plots,
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

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
