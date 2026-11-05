"""Publish aggregate validation evidence and static figures from recorded runs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .data import KNOWN_END, daily_retention

INK, TEAL, GRAY = "#172b3a", "#087e8b", "#8b98a5"


def render_report(frame, output, forecast_path="retention_d1_predictions.csv"):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    selected = json.loads((output / "selection.json").read_text())["selected_model"]
    leaderboard = pd.read_csv(output / "selection_leaderboard.csv")
    holdout = pd.read_csv(output / "holdout_metrics.csv")
    predictions = pd.read_csv(output / "holdout_predictions.csv", parse_dates=["date"])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ordered = leaderboard.sort_values("rmse_pp", ascending=False)
    bars = ax.barh(
        ordered.model,
        ordered.rmse_pp,
        color=[TEAL if name == selected else "#c5d1da" for name in ordered.model],
        height=0.65,
    )
    ax.bar_label(bars, fmt="%.3f", padding=6)
    ax.set_xlim(0, ordered.rmse_pp.max() * 1.2)
    ax.set_xlabel("Daily retention RMSE · percentage points · lower is better")
    ax.set_title(
        "Model selection on two forward validation windows", loc="left", pad=18
    )
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    fig.savefig(figures / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    actual = predictions[predictions.model == selected]
    ax.plot(
        actual.date,
        actual.actual,
        color=INK,
        marker="o",
        linewidth=2,
        label="Observed retention",
    )
    for name, color, style in (("random_forest", GRAY, "--"), (selected, TEAL, "-")):
        rows = predictions[predictions.model == name]
        ax.plot(
            rows.date,
            rows.predicted,
            color=color,
            linestyle=style,
            linewidth=2,
            label=name,
        )
    ax.set_title("Final chronological holdout · August 17 to 31", loc="left", pad=18)
    ax.set_ylabel("Retention D1 (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(frameon=False, ncol=3, loc="lower left")
    ax.grid(alpha=0.15)
    fig.savefig(figures / "holdout_forecast.png", dpi=180)
    plt.close(fig)

    observed = daily_retention(frame[frame.date <= KNOWN_END])
    forecast = pd.read_csv(forecast_path, parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ax.plot(
        observed.index,
        observed,
        color=INK,
        linewidth=1.7,
        label="Observed daily retention",
    )
    ax.plot(
        forecast.date,
        forecast.retention_d1,
        color=TEAL,
        marker=".",
        linewidth=2,
        label="September forecast",
    )
    ax.axvspan(
        pd.Timestamp("2022-09-01"), pd.Timestamp("2022-09-15"), color=TEAL, alpha=0.08
    )
    ax.set_title(
        "Daily player retention · observed history and forecast", loc="left", pad=18
    )
    ax.set_ylabel("Retention D1 (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.15)
    fig.savefig(figures / "retention_forecast.png", dpi=180)
    plt.close(fig)

    metrics = holdout.set_index("model")
    winner = metrics.loc[selected]
    forest = metrics.loc["random_forest"]
    improvement = 100 * (1 - winner.rmse_pp / forest.rmse_pp)
    selected_predictions = predictions[predictions.model == selected].copy()
    selected_predictions["error_pp"] = (
        selected_predictions.predicted - selected_predictions.actual
    )
    selected_predictions.to_csv(output / "holdout_errors.csv", index=False)
    lines = [
        "# Evaluation results",
        "",
        "All scores below are daily retention errors in **percentage points**.",
        "",
        f"Selected model: **`{selected}`**, chosen using July 18 to August 16 development predictions only.",
        "",
        "## Development model comparison",
        "",
        "| Model | RMSE (pp) | MAE (pp) |",
        "| --- | ---: | ---: |",
    ]
    for row in leaderboard.itertuples():
        lines.append(f"| `{row.model}` | {row.rmse_pp:.4f} | {row.mae_pp:.4f} |")
    lines += [
        "",
        "## Final holdout: August 17 to 31",
        "",
        "| Model | RMSE (pp) | MAE (pp) |",
        "| --- | ---: | ---: |",
    ]
    for row in holdout.itertuples():
        lines.append(f"| `{row.model}` | {row.rmse_pp:.4f} | {row.mae_pp:.4f} |")
    lines += [
        "",
        f"The selected model's holdout RMSE is {improvement:.1f}% {'lower' if improvement >= 0 else 'higher'} than the random forest reference."
        if improvement >= 0
        else f"The selected model did not beat the random forest reference on the final holdout ({-improvement:.1f}% higher RMSE).",
        "",
        f"Mean signed holdout error: {selected_predictions.error_pp.mean():+.4f} pp. Negative error indicates underprediction.",
        "",
        "The final holdout is a single 15 day window. It assesses the model selected during development; it is not used to reselect models or tune parameters. September targets are hidden, so no September accuracy is claimed.",
        "",
        "## Reproduction record",
        "",
        "`run_manifest.json` records the dataset SHA-256, package versions, and label cutoff. `selection.json` records the model choice, split dates, seed, and thread count. Daily predictions and metrics for each validation window are included alongside this report.",
        "",
        "The random forest reference uses the original notebook's configuration with 50 trees, depth 10 and minimum leaf size 4, with chronological validation and sparse preprocessing. Historical returned count RMSE is a different metric and is not compared with daily RMSE in percentage points.",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines))
