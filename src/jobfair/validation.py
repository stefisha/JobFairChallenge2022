"""Check the published evidence against the dataset without retraining models."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import daily_metrics, daily_retention
from .experiment import HOLDOUT, SELECTION_FOLDS


def check_reports(frame, data_path, reports, submission_path):
    reports = Path(reports)
    manifest = json.loads((reports / "run_manifest.json").read_text())
    digest = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    if manifest["dataset_sha256"] != digest:
        raise ValueError("Report dataset hash does not match the input CSV.")
    choice = json.loads((reports / "selection.json").read_text())
    leaderboard = pd.read_csv(reports / "selection_leaderboard.csv")
    if (
        choice["selected_model"]
        != leaderboard.sort_values(["rmse_pp", "model"]).iloc[0].model
    ):
        raise ValueError("Selected model does not match the development leaderboard.")
    for stage in ("selection", "holdout"):
        metrics = pd.read_csv(
            reports
            / f"{stage}_{'leaderboard' if stage == 'selection' else 'metrics'}.csv"
        )
        predictions = pd.read_csv(
            reports / f"{stage}_predictions.csv", parse_dates=["date"]
        )
        for row in metrics.itertuples():
            rows = predictions[predictions.model == row.model]
            expected_dates = []
            for start, end in SELECTION_FOLDS if stage == "selection" else [HOLDOUT]:
                expected_dates.extend(pd.date_range(start, end))
            if sorted(rows.date.tolist()) != sorted(expected_dates):
                raise ValueError(
                    f"Unexpected prediction dates for {stage}/{row.model}."
                )
            actual = daily_retention(frame[frame.date.isin(rows.date)])
            dates = pd.DatetimeIndex(rows.date)
            np.testing.assert_allclose(rows.actual, actual.reindex(dates), atol=1e-10)
            scores = daily_metrics(
                pd.Series(rows.actual.to_numpy(), index=dates),
                pd.Series(rows.predicted.to_numpy(), index=dates),
            )
            np.testing.assert_allclose(
                [scores["rmse_pp"], scores["mae_pp"]],
                [row.rmse_pp, row.mae_pp],
                atol=1e-10,
            )
    for path, start in (
        (Path(submission_path), "2022-09-01"),
        (reports / "submission_16_days.csv", "2022-08-31"),
    ):
        submission = pd.read_csv(path, parse_dates=["date"])
        if submission.columns.tolist() != ["date", "retention_d1"]:
            raise ValueError(f"Invalid submission columns: {path}")
        if submission.date.tolist() != list(pd.date_range(start, "2022-09-15")):
            raise ValueError(f"Invalid submission dates: {path}")
        values = submission.retention_d1.to_numpy()
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 100)).all():
            raise ValueError(f"Invalid retention percentages: {path}")
    print(
        "Dataset fingerprint, published metrics, split dates, and submission contracts verified."
    )
