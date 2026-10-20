# Customer Churn Prediction: ML Pipeline and API

End-to-end predictive analytics project: data → feature engineering → multi-model training →
evaluation → deployed REST API → dashboard → documentation.

**Business problem:** identify subscribers likely to cancel so retention offers go to the right people.

## Quick start
```bash
pip install -r requirements.txt
make all                      # generate data, train, build dashboard, run tests
uvicorn app.main:app --reload # API + interactive docs at http://localhost:8000/docs
open reports/dashboard.html   # visualization dashboard (self-contained)
```
Docker: `docker build -t churn-api . && docker run -p 8000:8000 churn-api`

## Try the API
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
 "gender":"Female","age":34,"region":"South","contract_type":"Month-to-month",
 "payment_method":"Electronic check","signup_channel":"Online","tenure_months":4,
 "monthly_charges":95.5,"total_charges":380,"support_tickets_90d":3,"avg_usage_gb":80,
 "late_payments_12m":2,"has_streaming":1,"has_security":0,"num_products":4,
 "days_since_last_login":40,"satisfaction_score":3.5}'
```
Endpoints: `GET /health`, `GET /model-info`, `POST /predict`, `POST /predict-batch` (≤5000 rows).

## Layout
| Path | Purpose |
|---|---|
| `src/data_generation.py` | Builds the 60,000-row dataset (replace with your own loader) |
| `src/preprocessing.py` | Cleaning, 10 engineered features, imputation, scaling, encoding |
| `src/train.py` | Trains 4 models, tunes one, selects on validation, tests once |
| `src/train_keras.py` | Optional TensorFlow/Keras deep-learning model |
| `src/evaluate.py` | Figures + `reports/dashboard.html` |
| `app/main.py` | FastAPI service with validated schema |
| `models/churn_model.joblib` | Trained pipeline + decision threshold + metadata |
| `docs/` | Architecture and full project report |
| `tests/` | API tests |

## Using your own data
Provide a CSV with the columns in `src/config.py` (or edit `CATEGORICAL`/`NUMERIC` and
`FeatureEngineer`), put it at `data/customers.csv`, and run `make train report`.
The included data is **synthetic**; see the report for what that means for the results.
