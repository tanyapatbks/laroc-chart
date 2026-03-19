import unittest

from coral_thesis.phases.category_mapping import map_to_category


class CategoryMappingTests(unittest.TestCase):
    def test_invalid_hue_group_returns_unknown(self) -> None:
        self.assertEqual(map_to_category("X", 0.5), "Unknown")

    def test_invalid_health_score_returns_error(self) -> None:
        self.assertEqual(map_to_category("B", "bad"), "Error")

    def test_level_boundaries(self) -> None:
        self.assertEqual(map_to_category("C", 0.0), "C1")
        self.assertEqual(map_to_category("C", 0.166), "C1")
        self.assertEqual(map_to_category("D", 0.2), "D2")
        self.assertEqual(map_to_category("E", 0.5), "E3")
        self.assertEqual(map_to_category("B", 0.9), "B6")

    def test_clamps_out_of_range_scores(self) -> None:
        self.assertEqual(map_to_category("E", -1.0), "E1")
        self.assertEqual(map_to_category("E", 1.5), "E6")


if __name__ == "__main__":
    unittest.main()

