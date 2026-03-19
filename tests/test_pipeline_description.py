import unittest
from pathlib import Path

from coral_thesis.config import load_config
from coral_thesis.pipeline import CoralPipeline


class PipelineDescriptionTests(unittest.TestCase):
    def test_pipeline_description_mentions_all_phases(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_config(project_root / "configs" / "base.yaml")
        pipeline = CoralPipeline(config)

        description = pipeline.describe()

        self.assertIn("Phase 1", description)
        self.assertIn("Phase 6", description)
        self.assertIn("dataset_dir", description)


if __name__ == "__main__":
    unittest.main()

