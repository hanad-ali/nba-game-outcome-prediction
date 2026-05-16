"""Generate comparison figures across all 6 models.

Outputs:
  figures/model_comparison.png      (accuracy / log-loss / AUC bars)
  figures/calibration.png           (reliability diagrams)
  figures/roc_curves.png            (ROC for all models)
  figures/confusion_matrix_xgb.png  (best classical)
  figures/ablation.png              (ablation bar chart)
  figures/bias.png                  (accuracy by market size)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, confusion_matrix, log_loss, roc_auc_score, roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

# ---------- load everything ----------
classical = json.loads((RESULTS / "classical_metrics.json").read_text())
deep = json.loads((RESULTS / "deep_metrics.json").read_text())
ablations = json.loads((RESULTS / "ablation_metrics.json").read_text())
bias = pd.read_csv(RESULTS / "bias_analysis.csv")
preds_c = pd.read_csv(RESULTS / "classical_preds_test.csv")
preds_d = pd.read_csv(RESULTS / "deep_preds_test.csv")

# unify into one dict
all_metrics = {**classical, **deep}
order = ["logreg", "rf", "xgb", "lgbm", "lstm", "transformer"]
labels = {"logreg": "LogReg", "rf": "RF", "xgb": "XGBoost",
          "lgbm": "LightGBM", "lstm": "LSTM", "transformer": "Transformer"}

# ---------- 1. Model comparison ----------
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
metric_names = ["accuracy", "log_loss", "auc"]
titles = ["Accuracy (higher = better)", "Log-loss (lower = better)", "AUC (higher = better)"]
colors = ["#3a7bd5"] * 4 + ["#d54f3a"] * 2
for ax, key, title in zip(axs, metric_names, titles):
    vals = [all_metrics[m][key] for m in order]
    bars = ax.bar([labels[m] for m in order], vals, color=colors)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=20)
# legend explaining color
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color="#3a7bd5", label="Classical"),
                    Patch(color="#d54f3a", label="Deep Learning")],
           loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
plt.tight_layout()
plt.savefig(FIGS / "model_comparison.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------- 2. Calibration ----------
y_true = preds_c["y_true"].to_numpy()
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfectly calibrated")
pred_cols = {"LogReg": preds_c["logreg_p"], "RF": preds_c["rf_p"],
             "XGBoost": preds_c["xgb_p"], "LightGBM": preds_c["lgbm_p"],
             "LSTM": preds_d["lstm_p"], "Transformer": preds_d["transformer_p"]}
for name, p in pred_cols.items():
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", label=name)
ax.set_xlabel("Mean predicted P(home win)")
ax.set_ylabel("Fraction of actual home wins")
ax.set_title("Calibration curves (2023-24 test set)")
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / "calibration.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------- 3. ROC curves ----------
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
for name, p in pred_cols.items():
    fpr, tpr, _ = roc_curve(y_true, p)
    auc_val = roc_auc_score(y_true, p)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves — 2023-24 test set")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / "roc_curves.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------- 4. Confusion matrix for best model (RF — highest accuracy) ----------
best_name = max(all_metrics, key=lambda k: all_metrics[k]["accuracy"])
print(f"best by accuracy: {best_name} ({all_metrics[best_name]['accuracy']:.4f})")
p_best = pred_cols[labels[best_name]] if best_name in labels else preds_c[f"{best_name}_p"]
y_pred = (p_best >= 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["Away win", "Home win"])
ax.set_yticks([0, 1], ["Away win", "Home win"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion matrix — {labels[best_name]}")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center",
            color="white" if v > cm.max() * 0.6 else "black", fontsize=14)
plt.colorbar(im, ax=ax, fraction=0.04)
plt.tight_layout()
plt.savefig(FIGS / f"confusion_matrix_{best_name}.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------- 5. Ablation chart ----------
abl_order = ["full", "drop_rest", "drop_h2h", "diff_only", "drop_rolling"]
abl_pretty = {"full": "Full (108 feats)", "drop_rest": "− rest/B2B",
              "drop_h2h": "− head-to-head", "diff_only": "DIFF feats only (36)",
              "drop_rolling": "− rolling avgs (18)"}
fig, ax = plt.subplots(figsize=(8, 4))
vals = [ablations[k]["accuracy"] for k in abl_order]
bars = ax.bar([abl_pretty[k] for k in abl_order], vals, color="#4a90c4")
ax.axhline(ablations["full"]["accuracy"], color="red", linestyle="--", alpha=0.5,
           label=f"full baseline ({ablations['full']['accuracy']:.3f})")
ax.set_title("XGBoost ablation studies — test accuracy")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.50, 0.70)
ax.grid(axis="y", alpha=0.3)
ax.legend()
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(FIGS / "ablation.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------- 6. Bias chart ----------
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(bias))
ax.bar(x - 0.2, bias["accuracy"], width=0.4, label="Model accuracy", color="#3a7bd5")
ax.bar(x + 0.2, bias["actual_home_win_rate"], width=0.4, label="Actual home win rate", color="#d54f3a")
ax.set_xticks(x, bias["HOME_MARKET"])
ax.set_xlabel("Home team market size")
ax.set_ylabel("Rate")
ax.set_title("Bias check: model accuracy across home-team market sizes (XGBoost)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
for i, (acc, hwr, n) in enumerate(zip(bias["accuracy"], bias["actual_home_win_rate"], bias["n"])):
    ax.text(i - 0.2, acc, f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
    ax.text(i + 0.2, hwr, f"{hwr:.3f}", ha="center", va="bottom", fontsize=9)
    ax.text(i, 0.05, f"n={n}", ha="center", fontsize=8, color="gray")
plt.tight_layout()
plt.savefig(FIGS / "bias.png", dpi=120, bbox_inches="tight")
plt.close()

print("\nFigures saved:")
for p in sorted(FIGS.glob("*.png")):
    print(f"  {p.name}")
