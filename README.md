# NBA Game Outcome Prediction — Hybrid Machine Learning Approaches

**Hanad Ali — CSCI 4931 Machine Learning Final Project**

This project compares classical machine learning (Logistic Regression, Random Forest, XGBoost, LightGBM) against deep sequence models (LSTM, Transformer) on the task of predicting NBA regular-season game winners. All six models are trained on the same 2018-19 through 2022-23 data and evaluated on the held-out 2023-24 season.

**Headline result:** Random Forest achieves 64.6% accuracy and 0.709 AUC on the test set — the deep models do **not** beat the classical approaches on team-level rolling box-score features.

## Repository Layout

```
final_project/
├── src/
│   ├── collect_data.py      # Pulls NBA game logs 2018-2024 via nba_api
│   ├── build_features.py    # Builds tabular (108 cols) + sequence (20x11) features
│   ├── train_classical.py   # LogReg, RF, XGBoost, LightGBM with TimeSeriesSplit CV
│   ├── train_deep.py        # LSTM + Transformer in PyTorch with early stopping
│   ├── analyze.py           # SHAP feature importance + ablation studies + bias check
│   ├── make_figures.py      # Generates all PNG figures
│   └── demo.py              # Quick demo: scores 2023-24 test set + prints results
├── data/                    # Cached game-log CSVs + engineered features
├── models/                  # Trained model artifacts (.joblib, .pt)
├── results/                 # Metrics JSONs + per-game test predictions
├── figures/                 # 8 PNG figures: model comparison, ROC, calibration, SHAP, ablation, bias
├── requirements.txt         # Pinned Python dependencies
└── README.md                # this file
```

## Setup

Tested on **Python 3.14** (Windows 11). Should work on Python 3.10+ with the listed versions.

```bash
# (Optional) create a fresh virtual environment first
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

If PyTorch fails to install on your Python version, see https://pytorch.org/get-started/locally/ for the CPU wheel that matches your environment.

## Reproducing the Results

The repo includes all cached intermediate data and trained models. You can either run the full pipeline end-to-end (~1 minute on a modern laptop) or run only the parts you want to inspect.

### Quickest path — see the headline demo

```bash
python src/demo.py
```

This loads the trained Random Forest, scores the 2023-24 test set, and prints:
- Overall accuracy + game count
- Top 5 high-confidence correct predictions (real team matchups)
- Top 3 confident misses (notable upsets the model didn't predict)
- A comparison table across all 6 models

### Full pipeline — re-run everything from scratch

The scripts are designed to run in this order. Each one writes artifacts that the next one reads:

```bash
python src/collect_data.py      # 30-60s   — fetches 6 seasons of game logs via nba_api
python src/build_features.py    # ~10s     — builds tabular + sequence feature tables
python src/train_classical.py   # ~30s     — trains LogReg, RF, XGB, LGBM with CV
python src/train_deep.py        # ~30s     — trains LSTM + Transformer (PyTorch, CPU)
python src/analyze.py           # ~15s     — SHAP, ablations, bias analysis
python src/make_figures.py      # ~5s      — generates all PNG figures
```

`collect_data.py` will skip seasons that are already cached in `data/`, so you can safely re-run it.

## Data

Raw game logs come from `nba_api.stats.endpoints.leaguegamefinder` and cover **2018-19 through 2023-24 regular seasons** (~7,059 games, 14,118 team-game rows). They are cached as CSVs in `data/`. No personal or proprietary data is used.

Train split: 2018-19 through 2022-23 (5,829 games).
Test split: 2023-24 (1,230 games).

## Method Summary

- **Tabular features** (108 columns per game): rolling box-score averages over 5/10/20 games, rest days, back-to-back indicators, season-to-date win percentages, head-to-head records — all computed strictly from games occurring *before* the target game (no leakage).
- **Sequence tensors** (20 × 11 per game): the last 10 games of each team's box-score history, stacked.
- **Classical models** use 4-fold `TimeSeriesSplit` cross-validation for hyperparameter tuning.
- **Deep models** use a chronological 85/15 train/validation split, Adam optimizer with cosine LR schedule, and early stopping on validation log-loss.
- **Interpretability** via SHAP `TreeExplainer` on XGBoost.
- **Ablation studies** retrain XGBoost on feature subsets to quantify each group's contribution.
- **Bias analysis** stratifies test accuracy by home-team market size (large / medium / small).

## Key Results

| Model | Accuracy | Log-loss | AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.635 | 0.626 | 0.701 |
| **Random Forest** | **0.646** | 0.627 | **0.709** |
| XGBoost | 0.637 | 0.646 | 0.674 |
| LightGBM | 0.634 | 0.655 | 0.675 |
| LSTM | 0.639 | 0.638 | 0.688 |
| Transformer | 0.614 | 0.646 | 0.683 |

**Ablation finding:** Dropping the rolling-window features collapses accuracy from 64% to 54%, confirming they carry nearly all the signal. The `DIFF`-only feature set (36 features, just home-minus-away differences) slightly *outperforms* the full 108-column set.

**Bias finding:** Accuracy is within one percentage point across large-, medium-, and small-market home teams. No detectable market-size bias.

## Ethical Note

This project is for educational and research purposes. It is not intended for gambling and does not incorporate betting-line features.

## References

- Hubáček, O., Šourek, G., & Železný, F. (2019). Exploiting sports-betting market using machine learning. *International Journal of Forecasting*, 35(2).
- Pai, P. F., Chang Liao, L. H., & Lin, K. P. (2017). Analyzing basketball games by support vector machines with decision tree models. *Neural Computing and Applications*, 28(12).
- Thabtah, F., Zhang, L., & Abdelhamid, N. (2019). NBA game result prediction using feature analysis and machine learning. *Annals of Data Science*, 6(1).
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
