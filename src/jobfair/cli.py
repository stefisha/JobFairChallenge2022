"""Command-line entry points for reproducible selection, validation, and forecasting."""

import argparse
import json
from pathlib import Path

from .data import DEFAULT_DATA, load_data, split_forecast, write_submission
from .experiment import evaluate_holdout, select_models, write_manifest
from .models import RetentionModel


def forecast(frame, selected, start, output, threads):
    train, future = split_forecast(frame, start)
    model = RetentionModel(selected, threads).fit(train)
    submission = write_submission(future, model.predict(future), output)
    print(
        f"Wrote {len(submission)} daily forecasts to {output}; trained through {train.date.max().date()}",
        flush=True,
    )
    return submission


def main():
    parser = argparse.ArgumentParser(
        description="Daily Top Eleven retention forecasting"
    )
    parser.add_argument(
        "command", choices=["select", "holdout", "forecast", "run", "check"]
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--forecast-start", default="2022-09-01")
    parser.add_argument(
        "--output", type=Path, default=Path("retention_d1_predictions.csv")
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    frame = load_data(args.data)
    args.reports.mkdir(parents=True, exist_ok=True)
    if args.command == "check":
        from .validation import check_reports

        check_reports(frame, args.data, args.reports, args.output)
        return
    if args.command in {"select", "run"}:
        selected = select_models(frame, args.reports, args.threads)
        write_manifest(args.data, args.reports)
    else:
        selected = json.loads((args.reports / "selection.json").read_text())[
            "selected_model"
        ]
    if args.command in {"holdout", "run"}:
        evaluate_holdout(frame, args.reports, selected, args.threads)
    if args.command in {"forecast", "run"}:
        forecast(frame, selected, args.forecast_start, args.output, args.threads)
    if args.command == "run":
        forecast(
            frame,
            selected,
            "2022-08-31",
            args.reports / "submission_16_days.csv",
            args.threads,
        )
        from .reporting import render_report

        render_report(frame, args.reports, args.output)


if __name__ == "__main__":
    main()
