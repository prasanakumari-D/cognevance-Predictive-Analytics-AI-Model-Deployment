"""Train, compare and persist churn models.

Run:  python -m src.train
Outputs: models/churn_model.joblib, reports/metrics.json, reports/test_predictions.csv
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .config import (CATEGORICAL, DATA_PATH, ID_COL, METRICS_PATH, MODEL_PATH, NUMERIC,
                     SEED, TARGET)
from .data_generation import main as generate_data
from .preprocessing import build_preprocessor


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        generate_data()
    return pd.read_csv(DATA_PATH)


def split(df):
    X = df.drop(columns=[TARGET, ID_COL])
    y = df[TARGET]
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)
    return X_train, X_val, X_test, y_train, y_val, y_test


def candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=5,
                                                n_jobs=-1, random_state=SEED),
        "Gradient Boosting (tuned)": RandomizedSearchCV(
            HistGradientBoostingClassifier(random_state=SEED, early_stopping=True),
            param_distributions={
                "learning_rate": uniform(0.03, 0.17),
                "max_depth": randint(3, 8),
                "max_leaf_nodes": randint(15, 63),
                "min_samples_leaf": randint(20, 120),
                "l2_regularization": uniform(0, 2),
            },
            n_iter=8, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
            scoring="roc_auc", random_state=SEED, n_jobs=1,
        ),
        # Deep learning: feed-forward neural network (sklearn). For TensorFlow see src/train_keras.py
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64, 32), alpha=1e-3,
                                              early_stopping=True, validation_fraction=0.1,
                                              n_iter_no_change=8, max_iter=200, random_state=SEED),
    }


def best_f1_threshold(y_true, proba):
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(t[np.argmax(f1)])


def score(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "threshold": thr,
    }


def main():
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split(df)
    print(f"train {len(X_train):,} | val {len(X_val):,} | test {len(X_test):,} | churn rate {y_train.mean():.1%}")

    results, fitted, val_proba = {}, {}, {}
    for name, clf in candidate_models().items():
        t0 = time.time()
        pipe = Pipeline([("prep", build_preprocessor()), ("model", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_val)[:, 1]
        thr = best_f1_threshold(y_val, proba)
        results[name] = {"validation": score(y_val, proba, thr), "train_seconds": round(time.time() - t0, 1)}
        fitted[name], val_proba[name] = pipe, proba
        print(f"{name:28s} val AUC {results[name]['validation']['roc_auc']:.4f} "
              f"F1 {results[name]['validation']['f1']:.4f} ({results[name]['train_seconds']}s)")

    best = max(results, key=lambda n: results[n]["validation"]["roc_auc"])
    thr = results[best]["validation"]["threshold"]
    print(f"\nBest model: {best} (threshold {thr:.3f})")

    # Final, untouched test-set evaluation for ALL models (selection used validation only)
    test_proba = {}
    for name, pipe in fitted.items():
        test_proba[name] = pipe.predict_proba(X_test)[:, 1]
        results[name]["test"] = score(y_test, test_proba[name], results[name]["validation"]["threshold"])

    best_params = None
    est = fitted["Gradient Boosting (tuned)"].named_steps["model"]
    best_params = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                   for k, v in est.best_params_.items()}

    final_pipe = fitted[best]
    joblib.dump({
        "pipeline": final_pipe,
        "threshold": thr,
        "model_name": best,
        "features": {"numeric": NUMERIC, "categorical": CATEGORICAL},
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "metrics_test": results[best]["test"],
    }, MODEL_PATH)

    out = {
        "best_model": best, "threshold": thr, "best_gb_params": best_params,
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "churn_rate": float(y_train.mean()), "models": results,
    }
    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps(out, indent=2))

    preds = X_test.copy()
    preds["actual"] = y_test.values
    for name, p in test_proba.items():
        preds["proba_" + name] = p
    preds.to_csv(METRICS_PATH.parent / "test_predictions.csv", index=False)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
