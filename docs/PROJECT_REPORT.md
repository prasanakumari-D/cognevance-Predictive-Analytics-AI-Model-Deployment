# Project Report: Customer Churn Prediction

## 1. Objective
Predict which subscription customers will churn so retention spend can be targeted. Success is judged by ranking quality (ROC-AUC, lift) and by recall at a usable precision, not by raw accuracy, because only ~28% of customers churn.

## 2. Data
60,000 synthetic customers (17 features). It was generated with a known non-linear churn mechanism plus noise and realistic defects (missing values, invalid charges). Synthetic data lets the project run anywhere, but **absolute scores will differ on real data**; the pipeline, not the numbers, is the transferable asset.

## 3. Method
See `ARCHITECTURE.md`. Split 36,000 / 12,000 / 12,000 (train / validation / test), stratified on churn.

## 4. Results (held-out test set, 12,000 customers)
| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Logistic Regression** (selected) | **0.810** | **0.615** | 0.483 | 0.796 | 0.601 |
| Gradient Boosting (tuned) | 0.805 | 0.607 | 0.510 | 0.724 | 0.598 |
| Neural Network (MLP) | 0.805 | 0.603 | 0.490 | 0.759 | 0.596 |
| Random Forest | 0.803 | 0.598 | 0.487 | 0.765 | 0.595 |

Exact figures are in `reports/metrics.json`.

**Findings**
- All four models land within ~0.007 AUC of each other. That gap is small enough that it should not be treated as a real ranking; a simple, explainable linear model is competitive because the feature engineering already encodes the interactions the tree and neural models would otherwise learn. In practice this is a reason to ship the simple model.
- The top 10% of customers by risk contain about 26% of all churners (2.6x the average churn rate); the top 30% contain roughly 60%.
- Strongest drivers (permutation importance): contract type, tenure, satisfaction score, days since last login, support tickets.
- Probabilities are reasonably calibrated (Brier ≈ 0.152), so the risk percentages returned by the API can be read as likelihoods.

## 5. Business use
Rank customers weekly, contact the top deciles first, and prioritize month-to-month, low-tenure, low-satisfaction customers with unresolved tickets. Pick the threshold from offer cost versus customer lifetime value: lower it to catch more churners, raise it to cut wasted offers. The model-selected threshold (0.26) favors recall.

## 6. Limitations and next steps
- Synthetic data; validate on real data before acting on the numbers.
- Correlation, not causation: the model ranks risk but does not say which offer would retain someone. An uplift model or A/B test is the next step.
- Add monitoring for data drift and calibration, and retrain on a schedule.
- Optional: run `src/train_keras.py` for a TensorFlow network (not run in this build; the sklearn MLP covers the deep-learning requirement).
