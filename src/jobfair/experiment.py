"""Rolling-origin model selection followed by one untouched chronological holdout."""

import hashlib
import json
import platform
from pathlib import Path
import time
from importlib.metadata import version

import pandas as pd

from .data import KNOWN_END, daily_metrics, daily_retention
from .models import MODEL_NAMES, RetentionModel

SELECTION_FOLDS = (("2022-07-18", "2022-08-01"), ("2022-08-02", "2022-08-16"))
HOLDOUT = ("2022-08-17", "2022-08-31")


def partition(frame, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end > KNOWN_END:
        raise ValueError("Evaluation cannot use the unlabeled forecast period.")
    train = frame[frame.date < start].copy()
    val = frame[(frame.date >= start) & (frame.date <= end)].copy()
    if train.empty or val.empty or train.date.max() >= val.date.min():
        raise ValueError("Expected a nonempty, strictly chronological split.")
    return train, val


def evaluate(frame, name, start, end, threads):
    train, val = partition(frame, start, end)
    tick = time.perf_counter()
    model = RetentionModel(name, threads).fit(train)
    predicted = daily_retention(val, model.predict(val))
    actual = daily_retention(val)
    metrics = daily_metrics(actual, predicted)
    print(
        f"{name:20s} {start}..{end}: RMSE {metrics['rmse_pp']:.4f} pp ({time.perf_counter()-tick:.1f}s)",
        flush=True,
    )
    predictions = pd.DataFrame(
        {
            "date": actual.index,
            "actual": actual.to_numpy(),
            "predicted": predicted.to_numpy(),
            "model": name,
            "fold_start": start,
        }
    )
    return metrics, predictions


def select_models(frame, output, threads=4, names=MODEL_NAMES):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scores, predictions = [], []
    for start, end in SELECTION_FOLDS:
        for name in names:
            metrics, result = evaluate(frame, name, start, end, threads)
            scores.append({"model": name, "start": start, "end": end, **metrics})
            predictions.append(result)
            # Persist progress after each candidate; long runs remain inspectable.
            pd.DataFrame(scores).to_csv(output / "selection_folds.csv", index=False)
    selection = pd.concat(predictions, ignore_index=True)
    selection.to_csv(output / "selection_predictions.csv", index=False)
    summary = []
    for name, result in selection.groupby("model"):
        metrics = daily_metrics(
            pd.Series(result.actual.to_numpy(), index=result.date),
            pd.Series(result.predicted.to_numpy(), index=result.date),
        )
        summary.append({"model": name, **metrics})
    leaderboard = pd.DataFrame(summary).sort_values(["rmse_pp", "model"])
    leaderboard.to_csv(output / "selection_leaderboard.csv", index=False)
    winner = str(leaderboard.iloc[0].model)
    selection_record = {
        "selected_model": winner,
        "selection_folds": SELECTION_FOLDS,
        "holdout": HOLDOUT,
        "metric": "daily_retention_rmse_percentage_points",
        "seed": 42,
        "threads": threads,
    }
    (output / "selection.json").write_text(
        json.dumps(selection_record, indent=2) + "\n"
    )
    print(f"Selected on development folds: {winner}", flush=True)
    return winner


def evaluate_holdout(frame, output, selected, threads=4):
    output = Path(output)
    scores, predictions = [], []
    # Fixed references; do not choose/re-tune a model using this holdout.
    for name in dict.fromkeys(
        ["historical_rate", "recent_28d", "random_forest", selected]
    ):
        metrics, result = evaluate(frame, name, *HOLDOUT, threads)
        scores.append({"model": name, **metrics})
        predictions.append(result)
    pd.DataFrame(scores).to_csv(output / "holdout_metrics.csv", index=False)
    pd.concat(predictions).to_csv(output / "holdout_predictions.csv", index=False)


def write_manifest(data_path, output):
    manifest = {
        "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "catboost", "matplotlib")
        },
        "known_labels_through": str(KNOWN_END.date()),
        "forecast_labels_available": False,
    }
    (Path(output) / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
