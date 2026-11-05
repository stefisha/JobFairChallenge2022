"""Tests for forecast contracts and leakage prevention using small synthetic cohorts."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from jobfair.data import (
    CATEGORICAL,
    daily_metrics,
    daily_retention,
    features,
    split_forecast,
    validate_data,
    write_submission,
)
from jobfair.experiment import partition
from jobfair.models import RetentionModel


def cohort_data():
    days = pd.date_range("2022-08-01", "2022-09-15")
    frame = pd.DataFrame(
        {
            "date": days.repeat(2),
            "registrations": [10, 30] * len(days),
            "returned": [2, 12] * len(days),
            "screen_dpi": 300.0,
            "device_memory_size_mb": 4096.0,
        }
    )
    for column in CATEGORICAL:
        frame[column] = ["A", "B"] * len(days)
    frame.loc[frame.date > "2022-08-31", "returned"] = 0
    return frame


class DataTests(unittest.TestCase):
    def test_daily_retention_sums_counts_before_dividing(self):
        frame = cohort_data().iloc[:2]
        # Cohort rates are 20% and 40%; the weighted daily rate is 35%, not 30%.
        self.assertAlmostEqual(daily_retention(frame).iloc[0], 35)

    def test_target_never_enters_features(self):
        frame = cohort_data()
        expected = features(frame)
        frame["returned"] = frame.registrations
        pd.testing.assert_frame_equal(features(frame), expected)
        self.assertNotIn("returned", expected)
        self.assertIn("day_of_week", expected)
        self.assertIn("days_since_start", expected)

    def test_placeholder_period_is_excluded_from_training(self):
        train, future = split_forecast(cohort_data())
        self.assertEqual(train.date.max(), pd.Timestamp("2022-08-31"))
        self.assertEqual(future.date.min(), pd.Timestamp("2022-09-01"))
        self.assertEqual(future.date.nunique(), 15)
        with self.assertRaisesRegex(ValueError, "unlabeled"):
            split_forecast(cohort_data(), "2022-09-02")

    def test_sixteen_day_variant_excludes_august31_from_training(self):
        train, future = split_forecast(cohort_data(), "2022-08-31")
        self.assertEqual(train.date.max(), pd.Timestamp("2022-08-30"))
        self.assertEqual(future.date.nunique(), 16)

    def test_temporal_validation_keeps_entire_days_separate(self):
        train, val = partition(cohort_data(), "2022-08-17", "2022-08-31")
        self.assertLess(train.date.max(), val.date.min())
        self.assertEqual(val.date.nunique(), 15)
        self.assertTrue(set(train.date).isdisjoint(val.date))
        with self.assertRaisesRegex(ValueError, "unlabeled"):
            partition(cohort_data(), "2022-09-01", "2022-09-15")

    def test_submission_has_one_row_per_day_and_four_decimals(self):
        _, future = split_forecast(cohort_data())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            daily = write_submission(future, future.registrations * 0.35, path)
            self.assertEqual(len(daily), 15)
            self.assertTrue(daily.date.is_unique)
            self.assertEqual(path.read_text().splitlines()[1], "2022-09-01,35.0000")

    def test_invalid_counts_and_missing_columns_are_rejected(self):
        frame = cohort_data()
        validate_data(frame)
        for column, value in [
            ("registrations", 0),
            ("returned", -1),
            ("returned", 100),
            ("returned", np.nan),
        ]:
            invalid = frame.copy()
            invalid.loc[0, column] = value
            with (
                self.subTest(column=column, value=value),
                self.assertRaises(ValueError),
            ):
                validate_data(invalid)
        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_data(frame.drop(columns=["device_model"]))

    def test_daily_metrics_use_percentage_points(self):
        actual = pd.Series([30.0, 40.0], index=["a", "b"])
        predicted = pd.Series([33.0, 36.0], index=["a", "b"])
        metrics = daily_metrics(actual, predicted)
        self.assertAlmostEqual(metrics["rmse_pp"], np.sqrt(12.5))
        self.assertEqual(metrics["mae_pp"], 3.5)
        with self.assertRaisesRegex(ValueError, "dates"):
            daily_metrics(actual, predicted.rename(index={"b": "c"}))


class ModelTests(unittest.TestCase):
    def test_models_handle_unseen_categories_and_produce_bounded_counts(self):
        train, future = split_forecast(cohort_data())
        future.loc[:, "device_model"] = "new-device"
        future.loc[:, "screen_dpi"] = np.nan
        for name in (
            "historical_rate",
            "recent_14d",
            "segment_28d",
            "random_forest",
            "ridge_rate",
            "catboost_rate",
            "catboost_recent",
        ):
            with self.subTest(model=name):
                predictions = RetentionModel(name, threads=1).fit(train).predict(future)
                self.assertEqual(len(predictions), len(future))
                self.assertTrue(np.isfinite(predictions).all())
                self.assertTrue((predictions >= 0).all())
                self.assertTrue((predictions <= future.registrations).all())

    def test_unseen_segments_fall_back_to_training_prior(self):
        train, future = split_forecast(cohort_data())
        future.loc[:, "registration_type"] = "unseen"
        model = RetentionModel("segment_28d").fit(train)
        np.testing.assert_allclose(model.predict(future), future.registrations * 0.35)


if __name__ == "__main__":
    unittest.main()
