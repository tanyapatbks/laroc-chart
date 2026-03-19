from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from coral_thesis.domain import SegmentationResult


class CoralSegmentationPhase:
    phase_name = "coral-segmentation"

    def __init__(self, model_path: Path, confidence_threshold: float = 0.25) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

    def run(self, source_paths: Sequence[Path], output_dir: Path) -> list[SegmentationResult]:
        from ultralytics import YOLO

        if not self.model_path.exists():
            raise FileNotFoundError(f"Coral segmentation weights not found: {self.model_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = output_dir / "masks"
        masked_dir = output_dir / "masked_images"
        masks_dir.mkdir(parents=True, exist_ok=True)
        masked_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(str(self.model_path))
        results = model(
            [str(path) for path in source_paths],
            stream=True,
            conf=self.confidence_threshold,
            task="segment",
        )

        phase_results: list[SegmentationResult] = []
        for result in results:
            if result.masks is None:
                continue

            image_path = Path(result.path)
            image_id = image_path.stem
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
                    source_image_path=image_path,
                    mask_path=mask_path,
                    masked_image_path=masked_image_path,
                )
            )

        return sorted(phase_results, key=lambda item: item.image_id)

