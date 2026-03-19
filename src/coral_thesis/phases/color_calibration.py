from __future__ import annotations

from coral_thesis.domain import CalibrationResult, DetectionResult


class ColorCalibrationPhase:
    phase_name = "color-calibration"

    def run(self, detections: list[DetectionResult]) -> list[CalibrationResult]:
        raise NotImplementedError(
            "Phase 2 has not been ported from the legacy prototype. "
            "The V2 rewrite will implement chart patch extraction, color correspondence, "
            "and image-wide correction as a dedicated step."
        )

