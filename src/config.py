"""Central configuration for paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "customers.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
FIG_DIR = ROOT / "reports" / "figures"
DASHBOARD_PATH = ROOT / "reports" / "dashboard.html"

TARGET = "churn"
ID_COL = "customer_id"
SEED = 42
N_ROWS = 60_000

CATEGORICAL = ["gender", "region", "contract_type", "payment_method", "signup_channel"]
NUMERIC = [
    "tenure_months", "age", "monthly_charges", "total_charges",
    "support_tickets_90d", "avg_usage_gb", "late_payments_12m",
    "has_streaming", "has_security", "num_products",
    "days_since_last_login", "satisfaction_score",
]
