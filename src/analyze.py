"""SHAP + ablation studies + bias analysis.

Outputs:
  results/shap_top_features.csv
  results/ablation_metrics.json
  results/bias_analysis.csv
  figures/shap_summary_xgb.png
  figures/shap_bar_xgb.png
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)


# ---------- load ----------
train = pd.read_csv(DATA / "tabular_train.csv").sort_values("GAME_DATE").reset_index(drop=True)
test = pd.read_csv(DATA / "tabular_test.csv").sort_values("GAME_DATE").reset_index(drop=True)
feat = json.loads((DATA / "feature_names.json").read_text())
markets = pd.read_csv(DATA / "team_markets.csv")
preds_classical = pd.read_csv(RESULTS / "classical_preds_test.csv")
preds_deep = pd.read_csv(RESULTS / "deep_preds_test.csv")

X_tr = train[feat].to_numpy(dtype=np.float32)
y_tr = train["HOME_WIN"].to_numpy(dtype=np.int32)
X_te = test[feat].to_numpy(dtype=np.float32)
y_te = test["HOME_WIN"].to_numpy(dtype=np.int32)


# =================== 1. SHAP on best tree model (XGBoost) ===================
print("[1/3] SHAP analysis on XGBoost...")
xgb = joblib.load(MODELS / "xgb.joblib")
explainer = shap.TreeExplainer(xgb)
# subsample test for speed
sample_idx = np.random.default_rng(0).choice(len(X_te), size=min(500, len(X_te)), replace=False)
shap_vals = explainer.shap_values(X_te[sample_idx])

# top features by mean(|SHAP|)
mean_abs = np.abs(shap_vals).mean(axis=0)
order = np.argsort(mean_abs)[::-1]
top = pd.DataFrame({
    "feature": [feat[i] for i in order],
    "mean_abs_shap": mean_abs[order],
}).head(30)
top.to_csv(RESULTS / "shap_top_features.csv", index=False)
print("  top 10 SHAP features:")
print(top.head(10).to_string(index=False))

# summary plot (bar)
shap.summary_plot(shap_vals, X_te[sample_idx], feature_names=feat,
                  plot_type="bar", max_display=15, show=False)
plt.tight_layout()
plt.savefig(FIGS / "shap_bar_xgb.png", dpi=120, bbox_inches="tight")
plt.close()

# beeswarm
shap.summary_plot(shap_vals, X_te[sample_idx], feature_names=feat,
                  max_display=15, show=False)
plt.tight_layout()
plt.savefig(FIGS / "shap_summary_xgb.png", dpi=120, bbox_inches="tight")
plt.close()
print("  saved figures/shap_*.png")


# =================== 2. Ablation studies ===================
print("\n[2/3] Ablation studies...")

def eval_split(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)),
    }

ablations = {}

# (a) Drop all rolling features (R5/R10/R20) — only static + rest + h2h remain
rolling_cols = [c for c in feat if "_R5" in c or "_R10" in c or "_R20" in c]
static_only = [c for c in feat if c not in rolling_cols]
print(f"  (a) drop rolling -> {len(static_only)} features remain")
xgb_a = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                       subsample=0.9, colsample_bytree=0.8, n_jobs=-1,
                       eval_metric="logloss", random_state=0)
xgb_a.fit(train[static_only].to_numpy(dtype=np.float32), y_tr)
ablations["drop_rolling"] = eval_split(xgb_a, test[static_only].to_numpy(dtype=np.float32), y_te)

# (b) Drop H2H + opponent context
no_h2h = [c for c in feat if "H2H" not in c]
print(f"  (b) drop H2H -> {len(no_h2h)} features remain")
xgb_b = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                      subsample=0.9, colsample_bytree=0.8, n_jobs=-1,
                      eval_metric="logloss", random_state=0)
xgb_b.fit(train[no_h2h].to_numpy(dtype=np.float32), y_tr)
ablations["drop_h2h"] = eval_split(xgb_b, test[no_h2h].to_numpy(dtype=np.float32), y_te)

# (c) Drop rest/B2B features
no_rest = [c for c in feat if "REST" not in c and "B2B" not in c]
print(f"  (c) drop rest/B2B -> {len(no_rest)} features remain")
xgb_c = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                      subsample=0.9, colsample_bytree=0.8, n_jobs=-1,
                      eval_metric="logloss", random_state=0)
xgb_c.fit(train[no_rest].to_numpy(dtype=np.float32), y_tr)
ablations["drop_rest"] = eval_split(xgb_c, test[no_rest].to_numpy(dtype=np.float32), y_te)

# (d) Use ONLY DIFF features (home-minus-away)
diff_only = [c for c in feat if c.startswith("DIFF_")]
print(f"  (d) DIFF features only -> {len(diff_only)} features remain")
xgb_d = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                      subsample=0.9, colsample_bytree=0.8, n_jobs=-1,
                      eval_metric="logloss", random_state=0)
xgb_d.fit(train[diff_only].to_numpy(dtype=np.float32), y_tr)
ablations["diff_only"] = eval_split(xgb_d, test[diff_only].to_numpy(dtype=np.float32), y_te)

# (e) full model (baseline)
ablations["full"] = eval_split(xgb, X_te, y_te)

with open(RESULTS / "ablation_metrics.json", "w") as f:
    json.dump(ablations, f, indent=2)

print("\n  ablation results:")
for name, m in ablations.items():
    print(f"    {name:>14}: acc={m['accuracy']:.4f}  ll={m['log_loss']:.4f}  auc={m['auc']:.4f}")


# =================== 3. Bias analysis: market size ===================
print("\n[3/3] Bias analysis (market size)...")
test_with_market = test.merge(
    markets.rename(columns={"TEAM_ABBR": "HOME_TEAM", "MARKET": "HOME_MARKET"}),
    on="HOME_TEAM", how="left"
).merge(
    markets.rename(columns={"TEAM_ABBR": "AWAY_TEAM", "MARKET": "AWAY_MARKET"}),
    on="AWAY_TEAM", how="left"
)
test_with_market = test_with_market.merge(
    preds_classical[["GAME_ID", "xgb_p"]].rename(columns={"xgb_p": "p_xgb"}),
    on="GAME_ID", how="left",
)
test_with_market["pred"] = (test_with_market["p_xgb"] >= 0.5).astype(int)
test_with_market["correct"] = (test_with_market["pred"] == test_with_market["HOME_WIN"]).astype(int)

# accuracy by home-team market size
bias = test_with_market.groupby("HOME_MARKET", observed=True).agg(
    n=("correct", "size"),
    accuracy=("correct", "mean"),
    avg_p_home_win=("p_xgb", "mean"),
    actual_home_win_rate=("HOME_WIN", "mean"),
).reset_index()
bias.to_csv(RESULTS / "bias_analysis.csv", index=False)
print(bias.to_string(index=False))

print("\nDone with analyses.")
