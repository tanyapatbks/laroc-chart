from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from coral_thesis.domain import FeatureVector


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

        return {
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

