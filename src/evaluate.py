"""Create evaluation figures and the self-contained HTML dashboard.

Run:  python -m src.evaluate
Outputs: reports/figures/*.png, reports/dashboard.html
"""
import base64
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

from .config import DASHBOARD_PATH, FIG_DIR, METRICS_PATH, MODEL_PATH, SEED
from .train import load_data, split

INK, TEAL, RUST, GREY = "#14213d", "#0f766e", "#c2410c", "#8a94a6"
PALETTE = [TEAL, RUST, INK, "#b08900"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titleweight": "bold", "axes.titlesize": 12,
    "figure.dpi": 130, "savefig.bbox": "tight",
})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def b64(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    metrics = json.loads(METRICS_PATH.read_text())
    bundle = joblib.load(MODEL_PATH)
    preds = pd.read_csv(METRICS_PATH.parent / "test_predictions.csv")
    names = list(metrics["models"].keys())
    y = preds["actual"].values
    best, thr = metrics["best_model"], metrics["threshold"]
    figs = {}

    # 1. Model comparison (test set)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    keys = ["roc_auc", "pr_auc", "f1", "recall", "precision"]
    w = 0.16
    for i, k in enumerate(keys):
        vals = [metrics["models"][n]["test"][k] for n in names]
        ax.bar(np.arange(len(names)) + i * w, vals, w, label=k.upper().replace("_", " "),
               color=[TEAL, "#2f9e8f", INK, RUST, "#e08a5b"][i])
    ax.set_xticks(np.arange(len(names)) + 2 * w)
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylim(0.3, 0.95); ax.legend(ncol=5, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.set_title("Test-set performance by model", pad=26)
    figs["comparison"] = save(fig, "model_comparison")

    # 2. ROC + PR
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for n, c in zip(names, PALETTE):
        p = preds["proba_" + n]
        fpr, tpr, _ = roc_curve(y, p); axes[0].plot(fpr, tpr, color=c, lw=2 if n == best else 1.2, label=n)
        pr, rc, _ = precision_recall_curve(y, p); axes[1].plot(rc, pr, color=c, lw=2 if n == best else 1.2)
    axes[0].plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    axes[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].axhline(y.mean(), ls="--", color=GREY, lw=1)
    axes[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    figs["roc_pr"] = save(fig, "roc_pr_curves")

    # 3. Confusion matrix (best)
    pb = preds["proba_" + best].values
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    ConfusionMatrixDisplay.from_predictions(y, (pb >= thr).astype(int), ax=ax, cmap="Blues",
                                            display_labels=["Stay", "Churn"], colorbar=False)
    ax.set_title(f"Confusion matrix @ {thr:.2f}")
    figs["confusion"] = save(fig, "confusion_matrix")

    # 4. Calibration
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    fp, mp = calibration_curve(y, pb, n_bins=10, strategy="quantile")
    ax.plot(mp, fp, "o-", color=TEAL); ax.plot([0, 1], [0, 1], "--", color=GREY)
    ax.set(title="Calibration", xlabel="Predicted probability", ylabel="Observed churn rate")
    figs["calibration"] = save(fig, "calibration")

    # 5. Permutation importance on raw input features (model-agnostic)
    _, _, X_test, _, _, y_test = split(load_data())
    idx = np.random.default_rng(SEED).choice(len(X_test), 4000, replace=False)
    pi = permutation_importance(bundle["pipeline"], X_test.iloc[idx], y_test.iloc[idx],
                                scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh(imp.index, imp.values, color=TEAL)
    ax.set(title="What drives churn predictions", xlabel="Drop in AUC when feature is shuffled")
    figs["importance"] = save(fig, "feature_importance")
    top_features = imp.sort_values(ascending=False).head(5)

    # 6. Cumulative gains / lift by risk decile
    d = pd.DataFrame({"p": pb, "y": y}).sort_values("p", ascending=False).reset_index(drop=True)
    d["decile"] = pd.qcut(d.index, 10, labels=range(1, 11))
    g = d.groupby("decile", observed=True)["y"].agg(["sum", "mean"])
    g["cum_capture"] = g["sum"].cumsum() / d["y"].sum()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].bar(g.index.astype(int), g["mean"] * 100, color=[RUST if i < 3 else GREY for i in range(10)])
    axes[0].axhline(y.mean() * 100, ls="--", color=INK, lw=1)
    axes[0].set(title="Churn rate by risk decile (1 = highest risk)", xlabel="Risk decile", ylabel="Actual churn %")
    axes[1].plot([0] + list(g.index.astype(int)), [0] + list(g["cum_capture"] * 100), "o-", color=TEAL)
    axes[1].plot([0, 10], [0, 100], "--", color=GREY)
    axes[1].set(title="Share of all churners captured", xlabel="Customers contacted (deciles)", ylabel="% of churners")
    figs["gains"] = save(fig, "gains_by_decile")

    # 7. Segment view from the raw data
    df = load_data()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, col in zip(axes, ["contract_type", "payment_method", "region"]):
        s = df.groupby(col)["churn"].mean().sort_values() * 100
        ax.barh(s.index, s.values, color=TEAL); ax.set_title(f"Churn % by {col.replace('_', ' ')}")
        for i, v in enumerate(s.values): ax.text(v + 0.4, i, f"{v:.0f}%", va="center", fontsize=8)
        ax.set_xlim(0, s.max() * 1.2)
    figs["segments"] = save(fig, "segments")

    # Highest-risk customers table (test set)
    top = preds.assign(risk=pb).sort_values("risk", ascending=False).head(10)
    cols = ["contract_type", "tenure_months", "monthly_charges", "support_tickets_90d", "payment_method"]
    rows = "".join(
        "<tr>" + "".join(f"<td>{r[c]:g}</td>" if isinstance(r[c], (int, float)) else f"<td>{r[c]}</td>" for c in cols)
        + f"<td class='risk'>{r['risk']:.0%}</td></tr>" for _, r in top.iterrows())

    t = metrics["models"][best]["test"]
    top10 = g.loc[1, "cum_capture"] * 100
    lift = g.loc[1, "mean"] / y.mean()
    comp_rows = "".join(
        f"<tr class='{'best' if n == best else ''}'><td>{n}</td>" +
        "".join(f"<td>{metrics['models'][n]['test'][k]:.3f}</td>" for k in ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]) +
        f"<td>{metrics['models'][n]['train_seconds']}s</td></tr>" for n in names)

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Churn prediction dashboard</title>
<style>
:root{{--ink:#14213d;--teal:#0f766e;--rust:#c2410c;--bg:#eef2f6;--card:#fff;--line:#d5dde7;--mut:#5b6678}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Avenir Next","Segoe UI",system-ui,sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:32px 20px 64px}}
h1{{font-size:30px;line-height:1.15;margin:0 0 8px;letter-spacing:-.01em}}h2{{font-size:18px;margin:36px 0 12px}}
.lead{{max-width:68ch;color:var(--mut);margin:0 0 24px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:14px 16px}}
.kpi b{{display:block;font-size:28px;color:var(--teal)}}.kpi span{{color:var(--mut);font-size:13px}}
.kpi.hot b{{color:var(--rust)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:14px}}
.grid{{display:grid;gap:14px}}.g2{{grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}}
img{{max-width:100%;height:auto;display:block;margin:auto}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line)}}
th{{color:var(--mut);font-weight:600}}tr.best td{{background:#e3f3f0;font-weight:600}}td.risk{{color:var(--rust);font-weight:700}}
.scroll{{overflow-x:auto}}.note{{color:var(--mut);font-size:13px;margin-top:8px}}
</style></head><body><main>
<h1>Contacting the riskiest 10% of customers reaches {top10:.0f}% of churners</h1>
<p class="lead">A churn model scored {metrics['n_test']:,} held-out customers. The best model, {best}, ranks them well enough that retention offers can be concentrated where they matter. Numbers below are on data the model never saw during training or tuning.</p>
<div class="kpis">
<div class="kpi"><b>{t['roc_auc']:.3f}</b><span>ROC AUC</span></div>
<div class="kpi"><b>{t['recall']:.0%}</b><span>of churners caught (recall)</span></div>
<div class="kpi"><b>{t['precision']:.0%}</b><span>of flagged customers churn (precision)</span></div>
<div class="kpi hot"><b>{lift:.1f}x</b><span>churn rate in top decile vs average</span></div>
<div class="kpi"><b>{metrics['churn_rate']:.0%}</b><span>baseline churn rate</span></div></div>

<h2>Model comparison</h2>
<div class="card scroll"><table><tr><th>Model</th><th>ROC AUC</th><th>PR AUC</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Train time</th></tr>{comp_rows}</table>
<p class="note">Each model uses the F1-optimal threshold found on the validation set. Differences between models are small (under 0.01 AUC), so the simplest, most explainable model was preferred.</p></div>
<div class="card" style="margin-top:14px"><img src="{b64(figs['comparison'])}" alt="Model comparison chart"></div>

<h2>Ranking quality</h2>
<div class="card"><img src="{b64(figs['roc_pr'])}" alt="ROC and precision-recall curves"></div>
<div class="card" style="margin-top:14px"><img src="{b64(figs['gains'])}" alt="Churn by risk decile and gains curve"></div>
<div class="grid g2" style="margin-top:14px"><div class="card"><img src="{b64(figs['confusion'])}" alt="Confusion matrix"></div>
<div class="card"><img src="{b64(figs['calibration'])}" alt="Calibration plot"></div></div>

<h2>Why customers churn</h2>
<div class="grid g2"><div class="card"><img src="{b64(figs['importance'])}" alt="Feature importance"></div>
<div class="card"><img src="{b64(figs['segments'])}" alt="Churn by segment"></div></div>
<p class="note">Top drivers: {", ".join(top_features.index)}.</p>

<h2>Ten highest-risk customers in the test set</h2>
<div class="card scroll"><table><tr><th>Contract</th><th>Tenure (months)</th><th>Monthly charge</th><th>Tickets (90d)</th><th>Payment</th><th>Churn risk</th></tr>{rows}</table></div>
</main></body></html>"""
    DASHBOARD_PATH.write_text(html)
    (METRICS_PATH.parent / "summary.json").write_text(json.dumps(
        {"top_decile_capture_pct": round(top10, 1), "top_decile_lift": round(lift, 2),
         "top_features": list(top_features.index)}, indent=2))
    print("Dashboard ->", DASHBOARD_PATH, "| top-decile capture", round(top10, 1), "| lift", round(lift, 2), "|", list(top_features.index))


if __name__ == "__main__":
    main()
