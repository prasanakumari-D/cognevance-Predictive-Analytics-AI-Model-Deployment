"""Cleaning, feature engineering and the sklearn preprocessing pipeline.

Everything that touches data lives inside a single sklearn Pipeline, so the exact
same transformations run in training, evaluation and the production API
(no train/serve skew).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL, NUMERIC

ENGINEERED = [
    "charge_per_product", "charge_to_tenure", "ticket_rate", "is_new_customer",
    "log_total_charges", "login_gap_flag", "risky_payment", "m2m_high_bill",
    "satisfaction_missing", "tenure_x_tickets",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Cleans impossible values and creates domain-driven features."""

    def fit(self, X, y=None):
        self.median_monthly_ = float(pd.Series(X["monthly_charges"]).where(lambda s: s > 0).median())
        return self

    def transform(self, X):
        df = X.copy()
        # Data cleaning: non-positive charges are invalid -> replace with training median
        df.loc[df["monthly_charges"] <= 0, "monthly_charges"] = self.median_monthly_
        # Winsorise heavy-tailed fields
        df["avg_usage_gb"] = df["avg_usage_gb"].clip(upper=600)

        df["charge_per_product"] = df["monthly_charges"] / df["num_products"].clip(lower=1)
        df["charge_to_tenure"] = df["monthly_charges"] / (df["tenure_months"] + 1)
        df["ticket_rate"] = df["support_tickets_90d"] / (df["tenure_months"].clip(upper=3) + 1)
        df["is_new_customer"] = (df["tenure_months"] < 6).astype(int)
        df["log_total_charges"] = np.log1p(df["total_charges"].clip(lower=0))
        df["login_gap_flag"] = (df["days_since_last_login"] > 30).astype(int)
        df["risky_payment"] = (df["payment_method"] == "Electronic check").astype(int)
        df["m2m_high_bill"] = ((df["contract_type"] == "Month-to-month") & (df["monthly_charges"] > 90)).astype(int)
        df["satisfaction_missing"] = df["satisfaction_score"].isna().astype(int)
        df["tenure_x_tickets"] = df["tenure_months"] * df["support_tickets_90d"]
        return df


def build_preprocessor() -> Pipeline:
    num_cols = NUMERIC + ENGINEERED
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ct = ColumnTransformer(
        [("num", numeric_pipe, num_cols), ("cat", cat_pipe, CATEGORICAL)],
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return Pipeline([("features", FeatureEngineer()), ("encode", ct)])
