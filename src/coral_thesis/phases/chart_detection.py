from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import cv2

from coral_thesis.domain import DetectionResult


class ChartDetectionPhase:
    phase_name = "chart-detection"

    def __init__(self, model_path: Path, confidence_threshold: float = 0.25) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

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
                if class_name != "chart":
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

