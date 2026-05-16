"""Train Random Forest, XGBoost, LightGBM with time-aware CV + small grid search.

Outputs:
  models/{rf,xgb,lgbm}.joblib
  results/classical_metrics.json
  results/classical_preds_test.csv  (test predictions for downstream analysis)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def load() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_csv(DATA / "tabular_train.csv")
    test = pd.read_csv(DATA / "tabular_test.csv")
    train["GAME_DATE"] = pd.to_datetime(train["GAME_DATE"])
    test["GAME_DATE"] = pd.to_datetime(test["GAME_DATE"])
    feature_cols = json.loads((DATA / "feature_names.json").read_text())
    return train.sort_values("GAME_DATE").reset_index(drop=True), test.sort_values("GAME_DATE").reset_index(drop=True), feature_cols


def time_cv_score(model_factory, params_list, X, y, n_splits: int = 4):
    """Manual CV: find best params by mean validation log-loss."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best, best_score = None, float("inf")
    for params in params_list:
        scores = []
        for tr_idx, va_idx in tscv.split(X):
            m = model_factory(**params)
            m.fit(X[tr_idx], y[tr_idx])
            p = m.predict_proba(X[va_idx])[:, 1]
            scores.append(log_loss(y[va_idx], p, labels=[0, 1]))
        mean = float(np.mean(scores))
        if mean < best_score:
            best_score = mean
            best = params
    return best, best_score


def evaluate(model, X, y) -> dict:
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
    }


def main():
    print("Loading data...")
    train, test, feat = load()
    X_tr = train[feat].to_numpy(dtype=np.float32)
    y_tr = train["HOME_WIN"].to_numpy(dtype=np.int32)
    X_te = test[feat].to_numpy(dtype=np.float32)
    y_te = test["HOME_WIN"].to_numpy(dtype=np.int32)
    print(f"  train: {X_tr.shape}, test: {X_te.shape}, features: {len(feat)}")

    metrics = {}
    preds = pd.DataFrame({"GAME_ID": test["GAME_ID"].values, "y_true": y_te})

    # ---------- Logistic Regression baseline (sanity) ----------
    t0 = time.time()
    sc = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(sc.transform(X_tr), y_tr)
    p_lr = lr.predict_proba(sc.transform(X_te))[:, 1]
    metrics["logreg"] = {**evaluate(lr, sc.transform(X_te), y_te), "fit_seconds": time.time() - t0}
    preds["logreg_p"] = p_lr
    joblib.dump((sc, lr), MODELS / "logreg.joblib")
    print(f"[logreg] {metrics['logreg']}")

    # ---------- Random Forest ----------
    t0 = time.time()
    rf_grid = [
        {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5, "n_jobs": -1, "random_state": 0},
        {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 3, "n_jobs": -1, "random_state": 0},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 5, "n_jobs": -1, "random_state": 0},
    ]
    best, cv = time_cv_score(RandomForestClassifier, rf_grid, X_tr, y_tr)
    print(f"  RF best params: {best} (cv log-loss {cv:.4f})")
    rf = RandomForestClassifier(**best).fit(X_tr, y_tr)
    p_rf = rf.predict_proba(X_te)[:, 1]
    metrics["rf"] = {**evaluate(rf, X_te, y_te), "best_params": best, "cv_logloss": cv, "fit_seconds": time.time() - t0}
    preds["rf_p"] = p_rf
    joblib.dump(rf, MODELS / "rf.joblib")
    print(f"[rf] {metrics['rf']}")

    # ---------- XGBoost ----------
    t0 = time.time()
    xgb_grid = [
        {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9,
         "colsample_bytree": 0.8, "n_jobs": -1, "eval_metric": "logloss", "random_state": 0},
        {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.85,
         "colsample_bytree": 0.8, "n_jobs": -1, "eval_metric": "logloss", "random_state": 0},
        {"n_estimators": 800, "max_depth": 5, "learning_rate": 0.02, "subsample": 0.9,
         "colsample_bytree": 0.7, "n_jobs": -1, "eval_metric": "logloss", "random_state": 0},
    ]
    best, cv = time_cv_score(XGBClassifier, xgb_grid, X_tr, y_tr)
    print(f"  XGB best params: {best} (cv log-loss {cv:.4f})")
    xgb = XGBClassifier(**best).fit(X_tr, y_tr)
    p_xgb = xgb.predict_proba(X_te)[:, 1]
    metrics["xgb"] = {**evaluate(xgb, X_te, y_te), "best_params": best, "cv_logloss": cv, "fit_seconds": time.time() - t0}
    preds["xgb_p"] = p_xgb
    joblib.dump(xgb, MODELS / "xgb.joblib")
    print(f"[xgb] {metrics['xgb']}")

    # ---------- LightGBM ----------
    t0 = time.time()
    lgb_grid = [
        {"n_estimators": 500, "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20,
         "subsample": 0.9, "colsample_bytree": 0.8, "n_jobs": -1, "random_state": 0, "verbose": -1},
        {"n_estimators": 800, "num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 20,
         "subsample": 0.85, "colsample_bytree": 0.8, "n_jobs": -1, "random_state": 0, "verbose": -1},
        {"n_estimators": 1000, "num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 30,
         "subsample": 0.9, "colsample_bytree": 0.7, "n_jobs": -1, "random_state": 0, "verbose": -1},
    ]
    best, cv = time_cv_score(LGBMClassifier, lgb_grid, X_tr, y_tr)
    print(f"  LGBM best params: {best} (cv log-loss {cv:.4f})")
    lgbm = LGBMClassifier(**best).fit(X_tr, y_tr)
    p_lgbm = lgbm.predict_proba(X_te)[:, 1]
    metrics["lgbm"] = {**evaluate(lgbm, X_te, y_te), "best_params": best, "cv_logloss": cv, "fit_seconds": time.time() - t0}
    preds["lgbm_p"] = p_lgbm
    joblib.dump(lgbm, MODELS / "lgbm.joblib")
    print(f"[lgbm] {metrics['lgbm']}")

    preds.to_csv(RESULTS / "classical_preds_test.csv", index=False)
    with open(RESULTS / "classical_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved metrics and predictions.")


if __name__ == "__main__":
    main()
