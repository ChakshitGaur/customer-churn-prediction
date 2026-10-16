# CLAUDE.md — Churn Project Rules

## Goal
Predict which customers will churn, explain why, and turn predictions into retention recommendations.
- **Primary metric:** PR-AUC. Also report ROC-AUC, recall in the top 10% risk decile, and lift.
- **Success:** final model beats the logistic regression baseline on the held-out test set.

## Workflow
- Work in phases (0–7). Stop after each phase, summarize, and wait for the user's "go".
- Ask before: deleting files, installing heavy packages, or assuming the churn definition.

## Data rules
- `data/raw/` is **read-only**. Write intermediate outputs to `data/interim/`, final splits to `data/processed/`.
- Never commit data, models, or secrets (enforced by `.gitignore`).
- **No leakage:** fit every transformer on the training split only; drop columns only known after churn.
- Split by time if a date column exists, otherwise stratified 70/15/15. Document the choice.
- Evaluate on the test set **once**, at the end of Phase 4.

## Code rules
- All config (paths, `SEED=42`, split sizes, costs) lives in `src/config.py`. No hardcoded paths.
- Reusable logic goes in `src/`; notebooks are for exploration only and import from `src/`.
- Type hints + docstrings on public functions. Use `src.utils.logging_utils.get_logger`, never `print`.
- Lint/format: `ruff` + `black` (line length 100). Tests: `pytest`.

## Commands (Windows; `make` not installed — run the commands directly)
| Task | Command |
|---|---|
| Tests | `.venv\Scripts\python -m pytest` |
| Lint | `.venv\Scripts\python -m ruff check src tests` and `... -m black --check src tests` |
| Pipeline | `.venv\Scripts\python -m src.pipeline` (~6 min, includes tuning) |
| App | `.venv\Scripts\python -m streamlit run app/app.py` (Phase 7) |

## Git
- Never push directly to `main`; use a branch + PR. The user approves merges.
