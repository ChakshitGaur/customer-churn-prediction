# Customer Churn Prediction

Predict which customers will churn, explain **why** with SHAP, and turn predictions into
concrete retention recommendations.

> 🚧 Work in progress. Results, figures, and resume bullets are filled in as phases complete.

## Problem
_To be completed in Phase 1 (dataset, business context, churn definition)._

## Approach
| Phase | What | Status |
|---|---|---|
| 0 | Project setup, environment, tests | ✅ |
| 1 | Data understanding (EDA) | ✅ |
| 2 | Cleaning + leakage-safe train/val/test split | ✅ |
| 3 | Feature engineering in a scikit-learn Pipeline | ✅ |
| 4 | Baseline → RF / XGBoost / LightGBM, tuning, cost-based threshold | ⏳ |
| 5 | Evaluation + SHAP explainability | ⏳ |
| 6 | Ranked risk list + retention recommendations | ⏳ |
| 7 | Streamlit dashboard + monitoring plan (optional) | ⏳ |

**Metrics:** PR-AUC (primary), ROC-AUC, recall in top 10% risk decile, lift.

## Results
_To be completed in Phases 4–6._

## Project structure
```
churn-project/
├── data/{raw,interim,processed}   # raw is read-only; data is never committed
├── notebooks/                     # 01_eda, 02_features, 03_model_analysis
├── src/
│   ├── config.py                  # paths, SEED=42, split sizes, costs
│   ├── data/                      # load, clean, split
│   ├── features/                  # build_features, pipeline
│   ├── models/                    # baseline, train, tune, evaluate, predict
│   ├── explain/                   # shap_analysis
│   └── utils/                     # logging, seeding
├── models/                        # trained models (gitignored)
├── reports/{figures,metrics}      # plots + metric JSON; final_report.md
├── tests/                         # pytest suite
└── app/                           # Streamlit dashboard (optional)
```

## How to run (Windows)
```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest
```
With GNU Make installed you can use `make setup`, `make test`, `make lint`, `make pipeline`.

## Tech stack
Python 3.11+ · pandas · scikit-learn · XGBoost · LightGBM · SHAP · matplotlib/seaborn · pytest · ruff · black
