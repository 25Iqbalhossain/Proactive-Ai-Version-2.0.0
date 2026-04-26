import unittest

import pandas as pd

from data_processing.data_cleaning import DataCleaner
from data_processing.dataset_analyzer import DatasetAnalyzer
from pipeline.training_pipeline import TrainingConfig, TrainingPipeline


class DatasetValidationTests(unittest.TestCase):
    def test_analyzer_warns_when_item_id_is_missing(self):
        df = pd.DataFrame(
            {
                "user_type": ["a", "b", "a", "c"],
                "rating": [5, 4, 3, 5],
                "created_at": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            }
        )
        analyzer = DatasetAnalyzer()
        mapping = analyzer.detect_columns(df)
        warnings = analyzer.validate_mapping(df, mapping)

        self.assertIsNone(mapping.itemID)
        self.assertTrue(any("No itemID column could be detected" in warning for warning in warnings))

    def test_data_cleaner_split_rejects_empty_dataframe(self):
        cleaner = DataCleaner()
        df = pd.DataFrame(columns=["userID", "itemID", "rating"])

        with self.assertRaisesRegex(ValueError, "no interactions remained after cleaning"):
            cleaner.split(df)

    def test_training_pipeline_fails_early_for_invalid_interaction_dataset(self):
        df = pd.DataFrame(
            {
                "user_type": ["guest", "member", "guest", "member"],
                "action": ["view", "click", "view", "click"],
                "rating": [5, 4, 5, 4],
                "created_at": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
                "id": [101, 102, 103, 104],
            }
        )
        pipeline = TrainingPipeline(TrainingConfig(n_tuning_trials=0))

        with self.assertRaisesRegex(ValueError, "0 interactions remained"):
            pipeline.run_from_dataframe(df)


if __name__ == "__main__":
    unittest.main()
