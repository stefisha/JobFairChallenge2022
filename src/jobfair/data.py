"""Challenge data contracts and registration-weighted daily evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA = Path("Challenge/2. job_fair_retention_prediction_2022_training.csv")
KNOWN_END = pd.Timestamp("2022-08-31")
CATEGORICAL = [
    "registration_type",
    "played_t11_before",
    "registration_channel",
    "network_type",
    "device_tier",
    "device_type",
    "device_manufacturer",
    "device_model",
    "os_version",
]
NUMERIC = ["screen_dpi", "device_memory_size_mb", "registrations"]


def load_data(path=DEFAULT_DATA):
    frame = pd.read_csv(path, parse_dates=["date"])
    validate_data(frame)
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def validate_data(frame):
    required = {"date", "returned", *CATEGORICAL, *NUMERIC}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty or frame.date.isna().any():
        raise ValueError("Expected nonempty data with valid dates.")
    for column in ("registrations", "returned"):
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values != np.floor(values)).any():
            raise ValueError(f"{column} must contain finite integer counts.")
    if (frame.registrations <= 0).any():
        raise ValueError("Registration counts must be positive.")
    if ((frame.returned < 0) | (frame.returned > frame.registrations)).any():
        raise ValueError("Returned counts must lie between zero and registrations.")


def split_forecast(frame, forecast_start="2022-09-01"):
    start = pd.Timestamp(forecast_start)
    if start > KNOWN_END + pd.Timedelta(days=1):
        raise ValueError(
            "Forecast starts must not move unlabeled September rows into training."
        )
    train = frame[(frame.date < start) & (frame.date <= KNOWN_END)].copy()
    future = frame[frame.date >= start].copy()
    if train.empty or future.empty:
        raise ValueError("Both labeled training data and forecast rows are required.")
    return train, future


def features(frame):
    """Only covariates known at registration time; never inspect returned counts."""
    out = frame[CATEGORICAL + NUMERIC].copy()
    for column in CATEGORICAL:
        out[column] = out[column].fillna("Unknown").astype(str)
    for column in NUMERIC:
        # Fixed sentinel; no statistics are learned from validation/forecast data.
        out[column] = (
            pd.to_numeric(out[column], errors="raise").fillna(-1).astype("float32")
        )
    out["day_of_week"] = frame.date.dt.dayofweek.astype("int32")
    out["days_since_start"] = (frame.date - pd.Timestamp("2022-06-01")).dt.days.astype(
        "int32"
    )
    return out


def daily_retention(frame, returned=None):
    """Sum counts before dividing; do not average cohort percentages."""
    counts = (
        frame.returned.to_numpy()
        if returned is None
        else np.asarray(returned, dtype=float)
    )
    if len(counts) != len(frame) or not np.isfinite(counts).all():
        raise ValueError("Expected one finite prediction per cohort row.")
    if (counts < 0).any() or (counts > frame.registrations.to_numpy() + 1e-8).any():
        raise ValueError("Predicted returned counts must be bounded by registrations.")
    totals = (
        pd.DataFrame(
            {
                "date": frame.date.to_numpy(),
                "registrations": frame.registrations.to_numpy(),
                "returned": counts,
            }
        )
        .groupby("date", sort=True)
        .sum()
    )
    return (100 * totals.returned / totals.registrations).rename("retention_d1")


def daily_metrics(actual, predicted):
    actual, predicted = actual.align(predicted, join="outer")
    if actual.isna().any() or predicted.isna().any():
        raise ValueError("Actual and predicted daily dates must match exactly.")
    error = predicted.to_numpy() - actual.to_numpy()
    return {
        "rmse_pp": float(np.sqrt(np.mean(error**2))),
        "mae_pp": float(np.mean(np.abs(error))),
    }


def write_submission(frame, predictions, path):
    daily = daily_retention(frame, predictions).reset_index()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(path, index=False, date_format="%Y-%m-%d", float_format="%.4f")
    return daily
