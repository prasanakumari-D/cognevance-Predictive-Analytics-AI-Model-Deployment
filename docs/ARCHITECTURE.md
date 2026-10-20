# AI Pipeline Architecture

```
 ┌──────────────┐   ┌───────────────────────────┐   ┌──────────────────────┐
 │ 1. Data      │──▶│ 2. Preprocessing          │──▶│ 3. Model training    │
 │ CSV / SQL    │   │ FeatureEngineer           │   │ LogReg, RandomForest │
 │ (60k rows)   │   │ impute → scale → one-hot  │   │ HistGB (tuned), MLP  │
 └──────────────┘   └───────────────────────────┘   └──────────┬───────────┘
                                                               ▼
 ┌──────────────┐   ┌───────────────────────────┐   ┌──────────────────────┐
 │ 6. Dashboard │◀──│ 5. Evaluation (test set)  │◀──│ 4. Selection         │
 │ HTML report  │   │ AUC, PR, F1, calibration, │   │ best val ROC-AUC +   │
 └──────────────┘   │ lift, permutation import. │   │ F1-optimal threshold │
                    └───────────────────────────┘   └──────────┬───────────┘
                                                               ▼
                           ┌───────────────────────────────────────────────┐
 client ── JSON ─────────▶ │ 7. FastAPI  /predict /predict-batch /health   │
                           │ loads models/churn_model.joblib (one pipeline)│
                           └───────────────────────────────────────────────┘
```

## Stages
1. **Data.** 60,000 customers, 17 predictors + target. Deliberately messy: ~12% missing satisfaction, ~2-3% missing age/usage, and ~0.1% invalid negative charges.
2. **Preprocessing** (`src/preprocessing.py`). `FeatureEngineer` cleans invalid values and adds `charge_per_product`, `charge_to_tenure`, `ticket_rate`, `is_new_customer`, `log_total_charges`, `login_gap_flag`, `risky_payment`, `m2m_high_bill`, `satisfaction_missing`, `tenure_x_tickets`. A `ColumnTransformer` imputes (median / mode), scales numerics and one-hot encodes categoricals. All statistics are learned on training data only.
3. **Training.** Stratified 60/20/20 train/validation/test split. Four models in identical pipelines; gradient boosting is tuned with 3-fold randomized search (8 candidates).
4. **Selection.** Winner = highest validation ROC-AUC. The decision threshold maximizes F1 on validation (class imbalance is handled by the threshold rather than class weights, which keeps probabilities calibrated).
5. **Evaluation.** The test set is touched once, after selection. Reports ROC-AUC, PR-AUC, accuracy, precision, recall, F1, Brier score, calibration, decile lift and permutation importance on raw features.
6. **Dashboard.** `reports/dashboard.html`, a single self-contained file.
7. **Serving.** The saved artifact bundles the *entire* pipeline (cleaning + features + model) with the threshold, so the API runs exactly the training-time transformations. Pydantic validates inputs; missing optional fields are imputed.

## Deployment options
- **Local / VM:** `uvicorn app.main:app --host 0.0.0.0 --port 8000` behind nginx.
- **Container:** `Dockerfile` included; push to AWS ECR + ECS/Fargate, Google Cloud Run, or Azure Container Apps.
- **Production hardening (not included):** auth/API keys, request logging, model versioning (MLflow), drift monitoring (e.g. Evidently), scheduled retraining, CI running `make test`.
