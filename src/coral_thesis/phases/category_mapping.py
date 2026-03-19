from __future__ import annotations


def map_to_category(hue_group: str, health_score: float) -> str:
    if hue_group not in {"B", "C", "D", "E"}:
        return "Unknown"

    if not isinstance(health_score, (float, int)):
        return "Error"

    score = max(0.0, min(1.0, float(health_score)))
    thresholds = (0.166, 0.333, 0.500, 0.666, 0.833)

    if score <= thresholds[0]:
        level = 1
    elif score <= thresholds[1]:
        level = 2
    elif score <= thresholds[2]:
        level = 3
    elif score <= thresholds[3]:
        level = 4
    elif score <= thresholds[4]:
        level = 5
    else:
        level = 6

    return f"{hue_group}{level}"


class CategoryMappingPhase:
    phase_name = "category-mapping"

    def run(self, hue_group: str, health_score: float) -> str:
        return map_to_category(hue_group, health_score)

