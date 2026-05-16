"""Pretty-printed demo of the project for the video walkthrough.

Loads the trained Random Forest, scores 10 hand-picked 2023-24 games, and
prints predictions + actual outcomes in a readable table.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def main():
    print("=" * 78)
    print("  NBA GAME OUTCOME PREDICTION -- LIVE DEMO")
    print("  Model: Random Forest (best on held-out 2023-24 season)")
    print("=" * 78)

    feat = json.loads((DATA / "feature_names.json").read_text())
    test = pd.read_csv(DATA / "tabular_test.csv").sort_values("GAME_DATE").reset_index(drop=True)
    rf = joblib.load(MODELS / "rf.joblib")

    X = test[feat].to_numpy(dtype=np.float32)
    p_home = rf.predict_proba(X)[:, 1]
    pred = (p_home >= 0.5).astype(int)
    correct = (pred == test["HOME_WIN"].values).astype(int)

    overall_acc = correct.mean()
    print(f"\nOverall test-set accuracy:   {overall_acc:.1%}   ({correct.sum()} / {len(correct)} games)")
    print(f"League home-win rate:        {test['HOME_WIN'].mean():.1%}")
    print(f"Model's mean P(home win):    {p_home.mean():.3f}")

    # 5 high-confidence picks that were correct ("model nailed it")
    confidence = np.abs(p_home - 0.5)
    high_conf = test.copy()
    high_conf["p_home"] = p_home
    high_conf["correct"] = correct
    high_conf["confidence"] = confidence

    print("\n" + "-" * 78)
    print("  HIGH-CONFIDENCE CORRECT PREDICTIONS (model was sure, and right)")
    print("-" * 78)
    print(f"  {'Date':10s} {'Matchup':24s} {'Pred':>14s}  {'P(home)':>8s}  {'Actual':>10s}")
    top = high_conf[high_conf["correct"] == 1].nlargest(5, "confidence")
    for _, r in top.iterrows():
        matchup = f"{r['AWAY_TEAM']} @ {r['HOME_TEAM']}"
        pick = r["HOME_TEAM"] if r["p_home"] >= 0.5 else r["AWAY_TEAM"]
        actual = r["HOME_TEAM"] if r["HOME_WIN"] == 1 else r["AWAY_TEAM"]
        date = pd.to_datetime(r["GAME_DATE"]).strftime("%Y-%m-%d")
        print(f"  {date:10s} {matchup:24s} {pick:>14s}  {r['p_home']:>8.3f}  {actual:>10s}")

    # 3 confident misses ("interesting upsets")
    print("\n" + "-" * 78)
    print("  HIGH-CONFIDENCE WRONG PREDICTIONS (the upsets the model missed)")
    print("-" * 78)
    print(f"  {'Date':10s} {'Matchup':24s} {'Pred':>14s}  {'P(home)':>8s}  {'Actual':>10s}")
    misses = high_conf[high_conf["correct"] == 0].nlargest(3, "confidence")
    for _, r in misses.iterrows():
        matchup = f"{r['AWAY_TEAM']} @ {r['HOME_TEAM']}"
        pick = r["HOME_TEAM"] if r["p_home"] >= 0.5 else r["AWAY_TEAM"]
        actual = r["HOME_TEAM"] if r["HOME_WIN"] == 1 else r["AWAY_TEAM"]
        date = pd.to_datetime(r["GAME_DATE"]).strftime("%Y-%m-%d")
        print(f"  {date:10s} {matchup:24s} {pick:>14s}  {r['p_home']:>8.3f}  {actual:>10s}")

    # ---------------- summary table ----------------
    print("\n" + "=" * 78)
    print("  ALL SIX MODELS -- 2023-24 TEST-SET SUMMARY")
    print("=" * 78)
    classical = json.loads((RESULTS / "classical_metrics.json").read_text())
    deep = json.loads((RESULTS / "deep_metrics.json").read_text())
    all_m = {**classical, **deep}
    labels = {"logreg": "LogReg", "rf": "Random Forest *",
              "xgb": "XGBoost", "lgbm": "LightGBM",
              "lstm": "LSTM", "transformer": "Transformer"}
    print(f"\n  {'Model':18s} {'Accuracy':>10s} {'Log-loss':>10s} {'AUC':>8s}")
    print("  " + "-" * 50)
    for k, name in labels.items():
        m = all_m[k]
        print(f"  {name:18s} {m['accuracy']:>10.3f} {m['log_loss']:>10.3f} {m['auc']:>8.3f}")

    print("\n  * = best by accuracy and AUC")
    print("=" * 78)


if __name__ == "__main__":
    main()
