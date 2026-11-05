"""Simple forecasting baselines, a forest reference, and weighted cohort models."""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, features

MODEL_NAMES = (
    "historical_rate",
    "recent_14d",
    "recent_28d",
    "segment_28d",
    "random_forest",
    "ridge_rate",
    "catboost_rate",
    "catboost_recent",
)


class RetentionModel:
    def __init__(self, name, threads=4):
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {name}")
        self.name, self.threads = name, threads

    def fit(self, train):
        if self.name in {"historical_rate", "recent_14d", "recent_28d", "segment_28d"}:
            window = 14 if self.name == "recent_14d" else 28
            recent = (
                train
                if self.name == "historical_rate"
                else train[train.date > train.date.max() - pd.Timedelta(days=window)]
            )
            self.rate = float(recent.returned.sum() / recent.registrations.sum())
            if self.name == "segment_28d":
                self.group_columns = [
                    "registration_type",
                    "played_t11_before",
                    "registration_channel",
                ]
                grouped = recent.groupby(self.group_columns)[
                    ["returned", "registrations"]
                ].sum()
                # A fixed 100-registration prior stabilizes sparse/unseen cohorts.
                self.group_rates = (grouped.returned + 100 * self.rate) / (
                    grouped.registrations + 100
                )
            return self

        X = features(train)
        if self.name == "ridge_rate":
            numeric = [column for column in X if column not in CATEGORICAL]
            self.estimator = make_pipeline(
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            make_pipeline(
                                SimpleImputer(missing_values=-1, strategy="median"),
                                StandardScaler(),
                            ),
                            numeric,
                        ),
                        (
                            "categorical",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist", min_frequency=100
                            ),
                            CATEGORICAL,
                        ),
                    ],
                    sparse_threshold=1.0,
                ),
                Ridge(alpha=100, solver="lsqr", tol=1e-5),
            )
            self.estimator.fit(
                X,
                train.returned / train.registrations,
                ridge__sample_weight=train.registrations.to_numpy(),
            )
            return self
        if self.name == "random_forest":
            # Same 50-tree/depth-10 count-regression family as the original notebook.
            # Keep sparse one-hot data sparse; learn imputation within training.
            numeric = [column for column in X if column not in CATEGORICAL]
            self.estimator = make_pipeline(
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            SimpleImputer(missing_values=-1, strategy="median"),
                            numeric,
                        ),
                        (
                            "categorical",
                            OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
                            CATEGORICAL,
                        ),
                    ],
                    sparse_threshold=1.0,
                ),
                RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    min_samples_leaf=4,
                    n_jobs=self.threads,
                    random_state=42,
                ),
            )
            self.estimator.fit(X, train.returned)
            return self

        weights = train.registrations.to_numpy(dtype=float)
        if self.name == "catboost_recent":
            age_days = (train.date.max() - train.date).dt.days.to_numpy()
            weights = weights * np.power(0.5, age_days / 28)
        self.estimator = CatBoostClassifier(
            loss_function="CrossEntropy",
            iterations=400,
            depth=5,
            learning_rate=0.05,
            l2_leaf_reg=8,
            random_seed=42,
            thread_count=self.threads,
            verbose=False,
            allow_writing_files=False,
        )
        self.estimator.fit(
            X,
            train.returned / train.registrations,
            sample_weight=weights,
            cat_features=CATEGORICAL,
        )
        return self

    def predict(self, frame):
        if self.name in {"historical_rate", "recent_14d", "recent_28d"}:
            return self.rate * frame.registrations.to_numpy()
        if self.name == "segment_28d":
            keys = pd.MultiIndex.from_frame(frame[self.group_columns])
            rates = self.group_rates.reindex(keys).fillna(self.rate).to_numpy()
            return rates * frame.registrations.to_numpy()
        X = features(frame)
        if self.name == "random_forest":
            return np.clip(self.estimator.predict(X), 0, frame.registrations.to_numpy())
        if self.name == "ridge_rate":
            return (
                np.clip(self.estimator.predict(X), 0, 1)
                * frame.registrations.to_numpy()
            )
        rates = self.estimator.predict_proba(X)[:, 1]
        return np.clip(rates, 0, 1) * frame.registrations.to_numpy()
