# Evaluation results

All scores below are daily retention errors in **percentage points**.

Selected model: **`segment_28d`**, chosen using July 18–August 16 development predictions only.

## Development model comparison

| Model | RMSE (pp) | MAE (pp) |
| --- | ---: | ---: |
| `segment_28d` | 1.0152 | 0.7835 |
| `random_forest` | 1.0829 | 0.8775 |
| `catboost_rate` | 1.0840 | 0.9107 |
| `catboost_recent` | 1.1777 | 1.0130 |
| `ridge_rate` | 1.2841 | 1.0471 |
| `recent_28d` | 1.3097 | 1.0156 |
| `historical_rate` | 1.3230 | 0.9953 |
| `recent_14d` | 1.3322 | 1.0737 |

## Final holdout: August 17–31

| Model | RMSE (pp) | MAE (pp) |
| --- | ---: | ---: |
| `historical_rate` | 2.1300 | 1.7432 |
| `recent_28d` | 2.0770 | 1.7020 |
| `random_forest` | 1.5038 | 1.2294 |
| `segment_28d` | 1.6224 | 1.3241 |

The selected model did not beat the random-forest reference on the final holdout (7.9% higher RMSE).

Mean signed holdout error: -1.1750 pp. Negative error indicates underprediction.

The final holdout is a single 15-day window. It assesses the development-selected model; it is not used to reselect models or tune parameters. September targets are hidden, so no September accuracy is claimed.

## Reproduction record

`run_manifest.json` records the dataset SHA-256, package versions, and label cutoff. `selection.json` records the model choice, split dates, seed, and thread count. Per-day predictions and fold-level metrics are included alongside this report.

The random-forest reference uses the original notebook's 50-tree, depth-10, minimum-leaf-4 family, with chronological validation and sparse preprocessing. Historical row-count RMSE is a different metric and is not compared with daily percentage-point RMSE.
