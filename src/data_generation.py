"""Generate a realistic synthetic subscription-business dataset (60k customers).

Swap this module for your own loader (CSV / SQL / API) to use real data. Any
dataset with the columns listed in config.py works with the rest of the pipeline.
"""
import numpy as np
import pandas as pd
from .config import DATA_PATH, N_ROWS, SEED


def generate(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    tenure = np.where(
        contract == "Month-to-month", rng.gamma(2.0, 8, n),
        np.where(contract == "One year", rng.gamma(4.0, 8, n), rng.gamma(6.0, 8, n)),
    ).clip(1, 72).round()
    age = rng.normal(43, 13, n).clip(18, 85).round()
    products = rng.choice([1, 2, 3, 4, 5], n, p=[0.30, 0.30, 0.20, 0.13, 0.07])
    streaming = rng.binomial(1, 0.45, n)
    security = rng.binomial(1, 0.35, n)
    monthly = (20 + 9 * products + 8 * streaming + 5 * security + rng.normal(0, 6, n)).clip(15, 200).round(2)
    total = (monthly * tenure * rng.uniform(0.92, 1.05, n)).round(2)
    tickets = rng.poisson(0.6 + 0.35 * (contract == "Month-to-month"), n)
    usage = rng.gamma(2.5, 40, n).round(1)
    late = rng.poisson(0.4 + 0.5 * (rng.random(n) < 0.2), n)
    last_login = rng.exponential(12, n).clip(0, 120).round()
    satisfaction = (7.2 - 0.45 * tickets - 0.25 * late + 0.02 * tenure + rng.normal(0, 1.4, n)).clip(1, 10).round(1)

    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "age": age,
        "region": rng.choice(["North", "South", "East", "West", "Central"], n, p=[0.22, 0.25, 0.18, 0.20, 0.15]),
        "contract_type": contract,
        "payment_method": rng.choice(["Credit card", "Bank transfer", "Electronic check", "Mailed check"], n, p=[0.3, 0.28, 0.27, 0.15]),
        "signup_channel": rng.choice(["Online", "Retail", "Referral", "Partner"], n, p=[0.45, 0.25, 0.2, 0.1]),
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "support_tickets_90d": tickets,
        "avg_usage_gb": usage,
        "late_payments_12m": late,
        "has_streaming": streaming,
        "has_security": security,
        "num_products": products,
        "days_since_last_login": last_login,
        "satisfaction_score": satisfaction,
    })

    # Hidden churn mechanism (non-linear + interactions) so models have real signal to find
    z = (
        -1.1
        + 1.0 * (contract == "Month-to-month") - 0.9 * (contract == "Two year")
        - 0.035 * tenure
        + 0.012 * (monthly - 70)
        + 0.30 * tickets + 0.35 * late
        + 0.03 * last_login
        - 0.30 * (satisfaction - 6)
        + 0.5 * (df["payment_method"] == "Electronic check")
        - 0.25 * security
        + 0.6 * ((contract == "Month-to-month") & (monthly > 90))
        + 0.4 * ((tenure < 6) & (tickets >= 2))
        + rng.normal(0, 0.6, n)
    )
    df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    # Real-world messiness: missing values and a few impossible entries
    for col, rate in [("satisfaction_score", 0.12), ("age", 0.02), ("avg_usage_gb", 0.03)]:
        df.loc[rng.random(n) < rate, col] = np.nan
    df.loc[rng.random(n) < 0.001, "monthly_charges"] = -1
    return df


def main():
    df = generate()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} rows x {df.shape[1]} cols -> {DATA_PATH} | churn rate {df.churn.mean():.1%}")


if __name__ == "__main__":
    main()
