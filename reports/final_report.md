# Customer Churn: Final Report

**Data:** IBM Telco Customer Churn (7,043 customers, 26.5% churned in the last month)
**Model:** tuned XGBoost with sigmoid calibration
**Evaluation:** all results are on a held-out test set of 1,057 customers that was used once, after every modeling decision was made.

---

## 1. Summary

- **The model finds churners far better than chance.** The 10% of customers it ranks riskiest churn at **77%**, 2.9 times the average. Contacting the top third reaches **73% of all churners**.
- **Targeting pays; blanket campaigns don't.** With the placeholder costs below, contacting the customers the model flags earns **+13,150** on the test customers. Contacting the same number at random **loses 3,582**, and contacting everyone **loses 10,850**.
- **Five factors drive churn:** month-to-month contracts, short tenure, fiber-optic internet, high monthly charges, and no security or tech-support add-ons. Each one points to a specific retention action.
- **The model beats the logistic regression baseline,** but only modestly: PR-AUC **0.671 vs 0.641**, 95% CI of the gain [+0.003, +0.055]. Most of the value comes from targeting at all, not from the choice of algorithm.

---

## 2. How well the model works (test set)

| Metric | Final model | LR baseline |
|---|---|---|
| **PR-AUC** (primary metric) | **0.671** | 0.641 |
| ROC-AUC | 0.856 | 0.849 |
| Recall in top 10% risk | 29.3% | 27.9% |
| Lift in top 10% | **2.92×** | 2.78× |
| Brier score (after calibration) | **0.133** | 0.167 (uncalibrated) |

**Targeting curve:**

| Customers contacted (highest risk first) | Share of all churners reached |
|---|---|
| Top 10% | 29% |
| Top 30% | 67% |
| Top 50% | 89% |
| Bottom 30% | contains only 7 of 317 customers who churned |

**Calibration.** The raw model over-predicted, because class weighting during training inflates scores: an average prediction of 42% against an actual rate of 26.5%. Sigmoid calibration fit on the validation set fixed this. On the test set the average prediction is now **27.4%** and the calibration error fell from **0.153 to 0.027**. The ranking is unchanged, so the probabilities in the customer list can be read at face value.

| Risk tier | Rule | Customers | Predicted churn | **Actual churn** | Share of all churners |
|---|---|---|---|---|---|
| **High** | probability ≥ 0.377 (contact) | 349 (33%) | 58.3% | **58.5%** | 73% |
| **Medium** | above-average risk (≥ 0.265) | 97 (9%) | 31.7% | **29.9%** | 10% |
| **Low** | everything else | 611 (58%) | 9.1% | **7.7%** | 17% |

---

## 3. Why customers churn

SHAP attribution on the test set. The share is how much of the model's reasoning each factor accounts for.

| Driver | Share | What the data shows |
|---|---|---|
| **Contract type** | 34% | Month-to-month churn 42.7% vs one-year 11.3% vs two-year 2.8% |
| **Tenure** | 12% | 53% churn in the first 6 months, falling to 9.5% after 4 years |
| **Internet service** | 10% | Fiber optic 41.9% vs DSL 19.0% |
| **Monthly charges** | 8% | Churners pay more (mean 74 vs 61 per month) |
| **No OnlineSecurity / TechSupport** | 7% + 7% | About 42% churn without each, vs about 15% with |
| **Payment method** | 5% | Electronic check 45.3% vs 15–19% for automatic payments |

**Who is in the High tier:** 99.7% are on month-to-month contracts, 78.5% have fiber internet, 62% are in their first year, and 62% pay by electronic check.

**Not drivers:** gender, partner status and phone service contribute almost nothing. Senior citizens churn at 42%, but the model attributes only about 0.5% of its reasoning to age. Their risk is explained by their contracts and services, so retention offers should target those, not age.

---

## 4. Recommendations

### For the High tier (349 test customers, 33%)

Every High-risk customer gets a **primary action** (contract conversion) and a **secondary action** chosen from their own next-strongest reason:

| Action | Customers | Trigger |
|---|---|---|
| **Discount to move to a 1- or 2-year contract** | 348 (primary) | Month-to-month contract |
| **First-90-days onboarding and check-in program** | 155 (secondary) | Short tenure |
| **Fiber service-quality check and price review** | 150 (secondary) | Fiber optic internet |
| Plan review and right-sizing | 29 (+1 primary) | High monthly charges |
| One-time credit for switching to auto-pay | 11 (secondary) | Electronic check payment |
| Free 3-month security + tech support trial | 4 (secondary) | No OnlineSecurity / TechSupport |

### For Medium and Low tiers

- **Medium (9%):** low-cost email nudge about their top risk factor. A full offer doesn't pay at this risk level.
- **Low (58%):** no action. Their churn rate is 7.7%, so offers here mostly go to people who would stay anyway.

### Beyond individual offers

1. **Investigate fiber.** Fiber customers pay more and churn at more than twice the rate of DSL customers. This looks like a product, price or service-quality problem that retention offers can only patch.
2. **Fix the first year.** More than half of customers leave within 6 months of joining. Better onboarding helps every new customer, not only the ones flagged.
3. **Push auto-pay and protection bundles** at sign-up, when they are cheapest to add.
4. **Measure, don't assume.** Run the first campaign with a randomly held-out control group from the High tier. The model predicts *who will churn*, not *who an offer will save*. Only a controlled test shows the real effect of each action, and the results can feed an uplift model later.

---

## 5. Business impact (test set, placeholder costs)

Assumptions, all placeholders in `src/config.py`: each contact costs **50**; a saved customer is worth **500**; **30%** of contacted churners stay because of the offer. Break-even churn probability: 50 / (500 × 0.30) = **0.33**. The profit-maximizing threshold chosen on validation (0.377 calibrated) sits close to this, which is a good sign the calibration is sound.

| Strategy | Contacts | Churners reached | Expected saved | Cost | **Profit** | Per 1,000 customers |
|---|---|---|---|---|---|---|
| Contact nobody | 0 | 0 | 0 | 0 | **0** | 0 |
| Contact everyone | 1,057 | 280 | 84 | 52,850 | **−10,850** | −10,265 |
| Random (same number as model) | 349 | 92.5 | 27.7 | 17,450 | **−3,582** | −3,389 |
| **Model targeting** | 349 | 204 | 61.2 | 17,450 | **+13,150** | **+12,441** |

The ranking is what creates value. Spending the same budget at random loses money, while spending it on the model's High tier earns about **12,400 per 1,000 customers per month** under these assumptions. With real costs, rerun the pipeline: the threshold and the numbers above update automatically.

---

## 6. Limitations

- **Placeholder economics.** Offer cost, customer value and success rate are assumptions. The threshold, tier sizes and profit figures depend on them.
- **Snapshot data.** For churned customers, tenure and total charges are measured at the moment they left. A production system should take features at a fixed date *before* the prediction window.
- **No behavioral signals.** The data has no usage trends, support tickets or payment failures. That is why the model separates risky groups from safe ones well, but is weaker *within* the riskiest groups (ROC-AUC about 0.77 inside month-to-month and new-customer segments, vs 0.86 overall).
- **Correlation, not causation.** Two-year customers churn less, but converting someone to a two-year contract won't necessarily give them the same risk. The control-group test in section 4 is how to find out.
- **Calibration data size.** Calibration was fit on 1,057 validation customers. It should be checked again as new outcomes arrive.

---

## 7. How this was built

| Step | Decision | Why |
|---|---|---|
| Split | Stratified 70 / 15 / 15 | No date column for a time-based split |
| Features | 41 inputs; engineered features tested, raw chosen | Engineered features gave no accuracy gain in any model |
| Models | Logistic regression, Random Forest, XGBoost, LightGBM with class weights | Cross-validated on train; top 3 tuned with 40-configuration searches |
| Selection | Best **validation** PR-AUC → XGBoost | Test set never used for a decision |
| Threshold | Maximize expected profit on **validation** | Ties the model to the business goal |
| Calibration | Sigmoid on validation, threshold mapped exactly | Honest probabilities; same customers contacted |
| Test | Scored once at the end | Unbiased estimate of real performance |

Reproduce everything with one command: `.venv\Scripts\python -m src.pipeline` (about 6 minutes).
