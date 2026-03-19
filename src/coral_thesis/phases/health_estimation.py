from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from coral_thesis.domain import EstimationResult, FeatureVector
from coral_thesis.phases.category_mapping import map_to_category


class HealthEstimationPhase:
    phase_name = "health-estimation"

    def __init__(
        self,
        hue_model_path: Path | None,
        health_model_path: Path | None,
    ) -> None:
        self.hue_model_path = hue_model_path
        self.health_model_path = health_model_path
        self.hue_model = self._load_model(hue_model_path)
        self.health_model = self._load_model(health_model_path)

    @staticmethod
    def _load_model(model_path: Path | None):
        if model_path is None:
            return None
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        return joblib.load(model_path)

    def is_ready(self) -> bool:
        return self.hue_model is not None and self.health_model is not None

    def run(self, features: FeatureVector) -> EstimationResult:
        if not self.is_ready():
            raise RuntimeError(
                "Health estimation models are not configured. "
                "Provide both hue_model_path and health_model_path."
            )

        frame = pd.DataFrame([features.values])
        hue_group = self.hue_model.predict(frame)[0]
        health_score = float(self.health_model.predict(frame)[0])
        category = map_to_category(hue_group, health_score)

        return EstimationResult(
            image_id=features.image_id,
            hue_group=hue_group,
            health_score=health_score,
            category=category,
        )

