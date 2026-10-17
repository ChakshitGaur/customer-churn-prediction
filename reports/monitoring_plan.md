# Monitoring Plan: Churn Model in Production

**What is monitored:** the calibrated XGBoost churn model (`models/calibrated_model.joblib`), scored monthly on all active customers.
**Who acts on it:** the data science owner (model health) and the retention team lead (campaign results).

The model can fail in three ways. Each needs a different check:

| Failure | Example | When you can detect it |
|---|---|---|
| **Inputs change** | A new pricing plan moves MonthlyCharges | Immediately, at scoring time |
| **Scores stop meaning what they say** | "30% risk" customers start churning at 45% | About 1 month later, when churn outcomes arrive |
| **Actions stop working** | Contract offers no longer change behaviour | After each campaign, using a control group |

---

## 1. Every scoring run (monthly, before the list goes out)

These checks need no outcome labels, so they run before anyone is contacted.

| Check | How | Alert |
|---|---|---|
| **Schema** | `app.helpers.prepare_batch` / `src.data.clean.clean`: required columns, unique IDs, blank `TotalCharges` only at tenure 0 | Any failure **blocks** the run |
| **Input drift** | PSI of each monitored feature vs. the training split (`src.models.monitor.drift_report`): Contract, tenure, MonthlyCharges, InternetService, PaymentMethod, TechSupport, OnlineSecurity | PSI ≥ 0.10 **investigate**; PSI ≥ 0.25 **hold the campaign** until explained |
| **Unseen categories** | Values not seen in training (e.g. a new payment method). These are encoded as all-zero and scored silently | Any new category → **investigate** |
| **Score distribution** | PSI of this month's calibrated scores vs. the test-set scores; mean predicted churn vs. the 26.5% historical rate | PSI ≥ 0.10, or mean predicted churn moves more than 5 points |
| **Tier sizes** | Share of customers in High / Medium / Low vs. the test set (33% / 9% / 58%) | High tier changes by more than 10 points |

PSI reading (also in `src/models/monitor.py`): **< 0.10** stable · **0.10–0.25** moderate shift · **≥ 0.25** major shift.

The Batch scoring tab in the dashboard shows the drift table for any uploaded file.

## 2. One month later (when churn outcomes arrive)

Join last month's scores to who actually churned. **Exclude customers who received an offer**, because the offer changes their outcome. Use the untreated Medium/Low tiers plus the High-tier control group (section 3).

| Metric | Baseline (test set) | Alert |
|---|---|---|
| **PR-AUC** (primary) | 0.671 | Below **0.62** for 2 months in a row (about the baseline model's level) |
| **Top-10% lift** | 2.92× | Below **2.3×** |
| **Calibration error (ECE)** | 0.027 | Above **0.06** |
| **Mean predicted vs. actual churn** | 27.4% vs 26.5% | Gap above **5 points** |
| **Actual churn by tier** | High 58.5%, Medium 29.9%, Low 7.7% | Tiers stop being ordered, or High drops below 45% |
| **Segment PR-AUC** | Contract, tenure band, internet type (`performance_by_segment`) | Any large segment drops more than 0.08 below its test value |

Monthly samples are small (about 1,000 customers), so judge trends over **2–3 months**, not single readings. Report bootstrap intervals where it matters.

## 3. After each retention campaign

The model predicts **who will churn**, not **who an offer will save**. Only a controlled test shows whether the actions work.

- **Hold out a random 10–20% of the High tier** as a control group that gets no offer.
- **Uplift** = churn rate (control) − churn rate (treated). Measure it per action (contract offer, onboarding, fiber check, auto-pay credit, security bundle), using the `primary_action` / `secondary_action` columns.
- **Replace the placeholder economics** in `src/config.py` (`OFFER_SUCCESS_RATE`, `RETENTION_OFFER_COST`, `RETAINED_CUSTOMER_VALUE`) with measured values, then rerun the pipeline. The threshold and tier cut-offs update automatically.
- **Alert:** an action's uplift is not positive over 2 campaigns → stop that action and review the rule in `src/models/recommend.py`.

## 4. When to retrain or recalibrate

| Trigger | Response |
|---|---|
| Calibration alert only (ranking metrics fine) | **Recalibrate**: refit the sigmoid on the latest labelled month. It's quick, and the ranking doesn't change |
| PR-AUC or lift alert for 2 months | **Retrain**: `python -m src.pipeline` on a refreshed dataset |
| Major input drift (PSI ≥ 0.25) on Contract, tenure or MonthlyCharges | Find the cause first (pipeline bug or real change). If real, retrain |
| New product, price plan or category | Retrain once a few months of outcomes exist; until then, flag those customers for manual review |
| Measured campaign economics differ from the placeholders | Update `src/config.py` and rerun; the model itself is unchanged |
| No alerts | **Scheduled retrain every 6 months** |

**Retraining safeguards:**
- The pipeline keeps its rules: train-only preprocessing, validation-only selection and threshold, test scored once.
- **Promote the new model only if** it beats the current one on the same fresh test month (PR-AUC, with a bootstrap CI) and its calibration error is ≤ 0.05.
- Keep the previous `models/` artifacts so you can roll back.

## 5. Known blind spots

- **Snapshot features.** Tenure and TotalCharges in the training data were recorded at churn time. In production, compute features as of the scoring date, or the model will look better in training than in use.
- **No behavioral data.** Usage trends, support tickets and payment failures are the most likely improvements if they become available. They would help most inside the month-to-month segment, where the model is weakest (ROC-AUC about 0.77).
- **Feedback loop.** Customers who receive offers churn less, which makes later training data look safer than reality. Always keep the control group, and train only on untreated customers or record treatment as a feature.
