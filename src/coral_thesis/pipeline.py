from __future__ import annotations

from pathlib import Path

from coral_thesis.config import IMAGE_SUFFIXES, PipelineConfig


class CoralPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def bootstrap(self) -> None:
        self.config.ensure_workspace()

    def discover_source_images(self) -> list[Path]:
        dataset_dir = self.config.paths.dataset_dir
        if not dataset_dir.exists():
            return []

        return sorted(
            path
            for path in dataset_dir.iterdir()
            if path.is_file() and path.suffix in IMAGE_SUFFIXES
        )

    def describe(self) -> str:
        source_images = self.discover_source_images()
        lines = [
            "Coral Thesis V2 Pipeline",
            f"project_root: {self.config.project_root}",
            f"config_path: {self.config.config_path}",
            f"dataset_dir: {self.config.paths.dataset_dir}",
            f"baseline_chart_path: {self.config.paths.baseline_chart_path}",
            f"source_images_detected: {len(source_images)}",
            f"phase1_prepared_dataset_dir: {self.config.phase1.prepared_dataset_dir}",
            f"phase1_training_dir: {self.config.phase1.training_dir}",
            f"phase1_inference_dir: {self.config.phase1.inference_dir}",
            f"phase2_output_dir: {self.config.phase2.output_dir}",
            f"phase2_baseline_profile_path: {self.config.phase2.baseline_profile_path}",
            f"phase2_evaluation_manifest_path: {self.config.phase2.evaluation_manifest_path}",
            f"phase2_evaluation_reports_dir: {self.config.phase2.evaluation_reports_dir}",
            f"phase3_prepared_dataset_dir: {self.config.phase3.prepared_dataset_dir}",
            f"phase3_training_dir: {self.config.phase3.training_dir}",
            f"phase3_inference_dir: {self.config.phase3.inference_dir}",
            f"phase3_labels_dir: {self.config.phase3.labels_dir}",
            f"phase3_split_strategy: {self.config.phase3.split_strategy}",
            "",
            "Phases",
            "Phase 1: Chart detection",
            "Phase 2: Color calibration",
            "Phase 3: Coral segmentation",
            "Phase 4: Feature extraction",
            "Phase 5: Health estimation",
            "Phase 6: Category mapping",
            "",
            "Status",
            "- Foundation scaffold ready",
            "- Phase 1 inventory, preparation, training, and inference commands implemented",
            "- Phase 2 baseline analysis, crop normalization, and batch calibration commands implemented",
            "- Phase 2 quality metrics, unreadable-input reporting, and per-sample timeouts implemented",
            "- Phase 2 curated manifest evaluation implemented",
            "- Phase 3 segmentation inventory, mixed-format label normalization, and dataset preparation commands implemented",
            "- Phase 4 and Phase 6 deterministic logic implemented",
            "- Phase 3 and Phase 5 wrappers prepared",
            "- Phase 2 model-based chart localization inside the crop remains open",
        ]
        return "\n".join(lines)
