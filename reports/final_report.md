# Customer Churn: Final Report

**Data:** IBM Telco Customer Churn (7,043 customers, 26.5% churned in the last month)
**Model:** tuned XGBoost with sigmoid calibration
**Evaluation:** all results are on a held-out test set of 1,057 customers that was used once, after every modeling decision was made.

---

## 1. Summary

- **The model finds churners far better than chance.** The 10% of customers it ranks riskiest churn at **77%**, 2.9 times the average. Contacting the top third reaches **72% of all churners**.
- **Targeting pays; blanket campaigns don't.** With the placeholder costs below, contacting the customers the model flags earns **+13,000** on the test customers. Contacting the same number at random **loses 3,521**, and contacting everyone **loses 10,850**.
- **The main drivers of churn** are month-to-month contracts, short tenure and fiber-optic internet, followed by missing security or tech-support add-ons, electronic-check payment and high monthly charges. Each one points to a specific retention action.
- **The model beats the logistic regression baseline,** but only modestly: PR-AUC **0.671 vs 0.641**, 95% CI of the gain [+0.007, +0.052]. Repeated cross-validation puts the gain nearer **+0.011** with an interval that includes zero (section 2). Most of the value comes from targeting at all, not from the choice of algorithm.

---

## 2. How well the model works (test set)

| Metric | Final model | LR baseline |
|---|---|---|
| **PR-AUC** (primary metric) | **0.671** | 0.641 |
| ROC-AUC | 0.855 | 0.849 |
| Recall in top 10% risk | 29.3% | 27.9% |
| Lift in top 10% | **2.92×** | 2.78× |
| Brier score (after calibration) | **0.134** | 0.167 (uncalibrated) |

**Targeting curve:**

| Customers contacted (highest risk first) | Share of all churners reached |
|---|---|
| Top 10% | 29% |
| Top 30% | 68% |
| Top 50% | 88% |
| Bottom 30% (317 customers) | contains only 6 churners |

**Calibration.** The raw model over-predicted, because class weighting during training inflates scores: an average prediction of 41% against an actual rate of 26.5%. Sigmoid calibration fit on the validation set fixed this. On the test set the average prediction is now **27.5%** and the calibration error fell from **0.149 to 0.033**. The ranking is unchanged, so the probabilities in the customer list can be read at face value.

| Risk tier | Rule | Customers | Predicted churn | **Actual churn** | Share of all churners |
|---|---|---|---|---|---|
| **High** | probability ≥ 0.400 (contact) | 343 (32%) | 58.3% | **58.6%** | 71.8% |
| **Medium** | above-average risk (≥ 0.265) | 111 (11%) | 33.3% | **27.0%** | 10.7% |
| **Low** | everything else | 603 (57%) | 8.9% | **8.1%** | 17.5% |

The High and Low tiers are well calibrated. The Medium tier is looser (33.3% predicted vs 27.0% actual) but holds only 111 customers, so a gap of that size is within sampling noise.

**Robustness check: repeated cross-validation.** The test set holds only 1,057 customers, so one split is a noisy yardstick. The comparison was repeated with 5×5 stratified K-fold on train + validation (5,986 customers, test set untouched; `python -m src.models.repeated_cv`):

| Model | PR-AUC (mean ± sd over 25 folds) | ROC-AUC |
|---|---|---|
| Logistic regression baseline | 0.656 ± 0.029 | 0.844 |
| Logistic regression, tuned | 0.656 ± 0.029 | 0.844 |
| **Final model (XGBoost)** | **0.667 ± 0.027** | **0.847** |

The final model's PR-AUC gain over the baseline is **+0.011** (95% CI [−0.006, +0.027], Nadeau-Bengio corrected paired test, p = 0.19); it wins in 72% of folds. This is smaller than the +0.030 seen on the test split, and the interval includes zero. Read it as: the model is at least as good as logistic regression and probably slightly better, but the data does not prove the algorithm matters. Tuning logistic regression gives it no gain, so the small edge comes from the model family, not tuning. The tuned params were chosen on CV over the train split, so these numbers are slightly optimistic for both tuned models.

---

## 3. Why customers churn

SHAP attribution on the test set. The share is how much of the model's reasoning each factor accounts for.

| Driver | Share | What the data shows |
|---|---|---|
| **Contract type** | 27% | Month-to-month churn 42.7% vs one-year 11.3% vs two-year 2.8% |
| **Tenure** | 18% | 53% churn in the first 6 months, falling to 9.5% after 4 years |
| **Internet service** | 13% | Fiber optic 41.9% vs DSL 19.0% |
| **No OnlineSecurity / TechSupport** | 6% + 6% | About 42% churn without each, vs about 15% with |
| **Payment method** | 5% | Electronic check 45.3% vs 15–19% for automatic payments |
| **Monthly charges** | 5% | Churners pay more (mean 74 vs 61 per month) |

**Who is in the High tier:** 99.4% are on month-to-month contracts, 81.9% have fiber internet, 60% are in their first year, and 63% pay by electronic check.

**Not drivers:** gender, partner status and phone service contribute almost nothing. Senior citizens churn at 42%, but the model attributes only about 1% of its reasoning to age. Their risk is explained by their contracts and services, so retention offers should target those, not age.

---

## 4. Recommendations

### For the High tier (343 test customers, 32%)

Every High-risk customer gets a **primary action** from their strongest churn reason and a **secondary action** from their strongest reason that maps to a different action:

| Action | As primary | As secondary | Trigger |
|---|---|---|---|
| **Discount to move to a 1- or 2-year contract** | 232 | 109 | Month-to-month contract |
| **First-90-days onboarding and check-in program** | 110 | 59 | Short tenure |
| **Fiber service-quality check and price review** | 1 | 163 | Fiber optic internet |
| One-time credit for switching to auto-pay | 0 | 8 | Electronic check payment |
| Plan review and right-sizing | 0 | 2 | High monthly charges |
| Free 3-month security + tech support trial | 0 | 2 | No OnlineSecurity / TechSupport |

### For Medium and Low tiers

- **Medium (11%):** low-cost email nudge about their top risk factor. A full offer doesn't pay at this risk level.
- **Low (57%):** no action. Their churn rate is 8.1%, so offers here mostly go to people who would stay anyway.

### Beyond individual offers

1. **Investigate fiber.** Fiber customers pay more and churn at more than twice the rate of DSL customers. This looks like a product, price or service-quality problem that retention offers can only patch.
2. **Fix the first year.** More than half of customers leave within 6 months of joining. Better onboarding helps every new customer, not only the ones flagged.
3. **Push auto-pay and protection bundles** at sign-up, when they are cheapest to add.
4. **Measure, don't assume.** Run the first campaign with a randomly held-out control group from the High tier. The model predicts *who will churn*, not *who an offer will save*. Only a controlled test shows the real effect of each action, and the results can feed an uplift model later.

---

## 5. Business impact (test set, placeholder costs)

Assumptions, all placeholders in `src/config.py`: each contact costs **50**; a saved customer is worth **500**; **30%** of contacted churners stay because of the offer. Break-even churn probability: 50 / (500 × 0.30) = **0.33**. The profit-maximizing threshold chosen on validation (0.400 calibrated) sits close to this, which is a good sign the calibration is sound.

| Strategy | Contacts | Churners reached | Expected saved | Cost | **Profit** | Per 1,000 customers |
|---|---|---|---|---|---|---|
| Contact nobody | 0 | 0 | 0 | 0 | **0** | 0 |
| Contact everyone | 1,057 | 280 | 84 | 52,850 | **−10,850** | −10,265 |
| Random (same number as model) | 343 | 90.9 | 27.3 | 17,150 | **−3,521** | −3,331 |
| **Model targeting** | 343 | 201 | 60.3 | 17,150 | **+13,000** | **+12,299** |

The ranking is what creates value. Spending the same budget at random loses money, while spending it on the model's High tier earns about **12,300 per 1,000 customers per month** under these assumptions. With real costs, rerun the pipeline: the threshold and the numbers above update automatically.

### How much do the placeholder costs matter?

The profit case rests on three assumed numbers, so they were varied (`python -m src.models.sensitivity`, out-of-fold calibrated scores on train + validation, test set untouched; the threshold is re-optimized for each scenario). Best profit per 1,000 customers per month, with customer value fixed at 500:

| Offer success rate | Cost 25 | Cost 50 (placeholder) | Cost 75 | Cost 100 |
|---|---|---|---|---|
| 10% | 1,783 | 0 | 0 | 0 |
| 20% | 11,155 | 3,567 | 530 | 0 |
| **30% (placeholder)** | 22,586 | **11,995** | 5,350 | 1,520 |
| 40% | 34,635 | 22,310 | 13,264 | 7,133 |
| 50% | 46,993 | 33,495 | 23,012 | 14,901 |

- **The success rate matters most.** At the placeholder cost and value, targeting stops paying below roughly **10%**, and going from 20% to 30% more than triples the profit. The 30% figure is an assumption, so the controlled offer test in section 4 is the most valuable thing to run.
- **Targeting is robust, blanket campaigns are not.** Targeting earned a profit in 79% of the 80 scenarios tried (all three inputs varied); contacting everyone did in 39%.
- **The threshold must follow the costs.** At 30% success, the best rule contacts about 53% of customers at cost 25 but only about 6% at cost 100. Keeping today's threshold while costs change loses money in several scenarios (for example −3,324 per 1,000 at cost 100 and 30% success, where re-tuning earns 1,520).
- The out-of-fold estimate at the placeholders (11,995 per 1,000) is close to the test-set figure above (12,299), which is a good sign.

Full grid: `reports/metrics/profit_sensitivity.csv`.

### How stable is the threshold?

The threshold (0.400 calibrated) was chosen on 1,057 validation customers, so it is noisy. It was bootstrapped (`python -m src.models.threshold_bootstrap`): 1,000 validation-sized resamples of the out-of-fold calibrated scores, re-picking the profit-maximizing threshold each time (placeholder costs, test set untouched).

| | Result |
|---|---|
| Best threshold on all development data | 0.32 (11,995 profit per 1,000 customers) |
| Threshold across resamples | median 0.33, 90% interval **[0.25, 0.47]**, sd 0.068 |
| Profit lost by a resampled threshold | mean **258** per 1,000 (2.2%), 95th percentile 702 (5.9%) |
| Resamples within 500 / 1,000 of the best | 85% / 99% |
| Thresholds within 95% of the best profit | **0.26 to 0.47** |

- **The exact value is not important.** Profit is flat over a wide range, so a noisy threshold costs little. Today's 0.400 sits inside the range, and the theoretical break-even (0.33, section 5) is close to the best.
- **Do not read the threshold, or the tier sizes that follow from it, as precise.** Anywhere between roughly 0.26 and 0.47 is nearly as good, so use that room to match the team's capacity: a lower threshold contacts more customers, a higher one fewer.
- Regret is a lower bound, because the best threshold is estimated from data that includes the resampled rows. And the range moves with the costs (see the sensitivity table above).

Full results: `reports/metrics/threshold_bootstrap.json`, `threshold_profit_curve.csv`.

### Does the model treat groups comparably?

Out-of-fold calibrated scores on train + validation (test set untouched), today's threshold (`python -m src.models.fairness`):

| Group | Customers | Actual churn | Contacted | Churners reached (recall) | Predicted − actual churn |
|---|---|---|---|---|---|
| Female | 2,959 | 26.9% | 30.5% | 67.6% | −0.002 |
| Male | 3,027 | 26.2% | 29.7% | 66.8% | +0.001 |
| Not senior | 5,013 | 23.7% | 26.0% | 63.4% | +0.003 |
| **Senior** | 973 | 41.2% | 51.4% | **78.6%** | −0.017 |
| Partner: no | 3,085 | 32.9% | 39.5% | 72.3% | −0.002 |
| Partner: yes | 2,901 | 19.8% | 20.1% | 58.2% | +0.002 |
| Dependents: no | 4,209 | 31.3% | 36.9% | 70.6% | −0.004 |
| **Dependents: yes** | 1,777 | 15.4% | 14.0% | **50.9%** | +0.009 |

- **Gender: no gap.** Contact rates, recall and calibration are within one point.
- **Calibration holds in every group** (all within 0.02), so a predicted 40% means about 40% in each group.
- **Recall differs by group, and two gaps are flagged (more than 10 points from the overall 67.2%).** Seniors are contacted nearly twice as often (51.4% vs 26.0%) and 78.6% of their churners are reached, because they really do churn more (41% vs 24%). Customers with dependents are the other side: they churn least (15%) and only 51% of their churners are reached.
- **Why:** one global threshold favors groups with a higher base rate. The model does not use age or family status as a driver (section 3), and offers here are a benefit, so the disadvantaged group is the one *under*-contacted. Whether to accept that, or use group-specific thresholds to equalize recall at some cost to profit, is a business and policy decision. Check it with legal or compliance before any group-specific rule, since using age or family status directly can be restricted.

Full table: `reports/metrics/subgroup_report.csv`.

---

## 6. Limitations

- **Placeholder economics.** Offer cost, customer value and success rate are assumptions. The threshold, tier sizes and profit figures depend on them.
- **Snapshot data.** For churned customers, tenure and total charges are measured at the moment they left, while active customers are measured today. So a short tenure partly means "already gone", and the model can use that in training in a way it cannot when scoring live customers. A production system should take features at a fixed date *before* the prediction window. The size of the effect cannot be measured without dated data, but a sensitivity check bounds how much the model leans on these columns (repeated 5-fold CV on train + validation, 3 repeats, columns zeroed out so they carry no information; PR-AUC; reproduce with `python -m src.models.ablation`, results in `reports/metrics/feature_ablation.csv`):

  | Columns removed | Baseline | Final model |
  |---|---|---|
  | None | 0.657 | 0.668 |
  | TotalCharges | 0.655 | 0.667 |
  | tenure and TotalCharges | 0.606 | 0.602 |

  `TotalCharges` is almost exactly `tenure × MonthlyCharges` (ratio 1.00 ± 0.05), so removing it costs nothing. Removing `tenure` as well costs about 0.07 PR-AUC, and the final model then falls to the baseline's level. This is an upper bound on the exposure, not a measured leak: tenure is also a legitimate risk signal for active customers. Treat the reported figures as an upper estimate until the model is re-evaluated on features taken before the prediction window.
- **No behavioral signals.** The data has no usage trends, support tickets or payment failures. That is why the model separates risky groups from safe ones well, but is weaker *within* the riskiest groups (ROC-AUC about 0.77 inside month-to-month and new-customer segments, vs 0.855 overall).
- **Correlation, not causation.** Two-year customers churn less, but converting someone to a two-year contract won't necessarily give them the same risk. The control-group test in section 4 is how to find out.
- **Run-to-run variation.** An earlier version of this pipeline gave different numbers on different machines (XGBoost validation PR-AUC 0.6368 vs 0.6410 with identical hyperparameters), which moved the profit threshold, the High tier size and the test profit by about 10%. The cause was XGBoost's row and column subsampling, which draws random numbers from platform-specific generators. XGBoost now runs without subsampling, which cost nothing (CV PR-AUC 0.6715 vs 0.6717 to 0.6729) and makes the fit identical for every seed and platform. The figures in this report come from a run after that change. They were reproduced exactly on Linux with Python 3.13, on Linux with Python 3.14 (the weekly real-data check on GitHub), and with one thread versus all cores. Windows has not been confirmed: after pulling, run `python -m src.pipeline`; if `git status` then shows no changed files under `reports/`, it reproduces.
- **Calibration data size.** Calibration was fit on 1,057 validation customers. It should be checked again as new outcomes arrive.

**Interpretable alternative: Explainable Boosting Machine.** An EBM is an additive model with one learned shape per feature, so every prediction splits exactly into per-feature contributions, with no SHAP approximation. It was compared with the same 5×5 repeated CV on train + validation (test set untouched; no interaction terms, raw features; `python -m src.models.ebm`):

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Logistic regression baseline | 0.656 | 0.844 |
| Final model (XGBoost) | 0.667 | 0.847 |
| **EBM** | **0.668** | **0.847** |

The EBM matches the final model: PR-AUC difference **+0.001** (95% CI [−0.006, +0.007], p = 0.82), and it is **+0.011** over the baseline (95% CI [−0.001, +0.024], p = 0.07). Its feature importances agree closely with SHAP (rank correlation 0.85; Contract first in both). So the same accuracy is available from a model whose explanations are exact. The shipped model is unchanged: switching would also mean redoing calibration, the threshold and the dashboard's explanations, and a single-split test evaluation, so it is a decision for you, not an automatic swap. Details in `reports/metrics/ebm_comparison.json` and `ebm_importance.csv`.

---

## 7. How this was built

| Step | Decision | Why |
|---|---|---|
| Split | Stratified 70 / 15 / 15 | No date column for a time-based split |
| Features | 41 inputs; engineered features tested, raw chosen | Engineered features gave no accuracy gain in any model |
| Models | Logistic regression, Random Forest, XGBoost, LightGBM with class weights | Cross-validated on train; the two that the explanations support (logistic regression, XGBoost) tuned with 40-configuration searches. Random forest and LightGBM stay in the untuned comparison only |
| Selection | Best **validation** PR-AUC among models the explanations support (logistic regression, XGBoost); a simpler model wins if it is within one standard error → XGBoost | Test set never used for a decision. The tuned models are statistically tied, so the rule avoids letting tiny score differences pick the shipped model |
| Threshold | Maximize expected profit on **validation** | Ties the model to the business goal |
| Calibration | Sigmoid on validation, threshold mapped exactly | Honest probabilities; same customers contacted |
| Test | Scored once at the end | Unbiased estimate of real performance |

Reproduce everything with one command: `python -m src.pipeline` (about 2 minutes on a 4-core machine).
