# Addressing Reviewer Feedback: Additional Baselines and Statistical Significance

> Reviewer: “While the ensemble approach is evaluated, the study could be strengthened by including additional baselines such as XGBoost, CatBoost, or simple logistic regression to highlight the performance gain attributable to the proposed method. Including statistical significance tests or confidence intervals for accuracy differences would improve the credibility of the reported improvements.”

This document details the additions we are making in response to the feedback: new baseline models, a rigorous evaluation protocol aligned with the existing pipeline, and statistical tests with confidence intervals to quantify the reliability of observed differences.

## Added Baselines

We will evaluate three widely used baselines alongside our Decision Tree Ensemble approach, using the same feature sets and data splits as the main study (10-, 20-, and 30-minute windows):

1. Logistic Regression (LR)
   - Binary outcome: Radiant Win (1) vs. Loss (0).
   - Regularization: L2 (ridge) with strength C tuned on a log scale; optionally L1 via saga solver if sparsity benefits.
   - Solver: `liblinear` (dense/smaller), `saga` (large/sparse). Standardization via a `StandardScaler` within a pipeline.
   - Class imbalance (if present): use `class_weight='balanced'` and report with/without balancing.

2. XGBoost (XGBClassifier)
   - Key hyperparameters: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`.
   - Early stopping on a held-out validation fold (patience ≈ 50 rounds), with `eval_metric=['logloss','error','auc']` for monitoring.
   - Reproducibility: fixed `random_state`, deterministic settings where feasible.

3. CatBoost (CatBoostClassifier)
   - Key hyperparameters: `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, `border_count`.
   - Loss: `Logloss`; Metrics: `Accuracy`, `AUC`.
   - Early stopping (patience ≈ 50) and `random_seed` for reproducibility.
   - Note: Even if features are numeric, CatBoost is a strong gradient boosting baseline with robust default handling.

For each model we will use a scikit-learn style pipeline to ensure identical preprocessing across baselines and the proposed method.

## Evaluation Protocol (aligned with existing pipeline)

- Data windows: 10 min, 20 min, 30 min feature snapshots (as in the current study).
- Splits: Stratified K-fold cross-validation (K=5) per window to preserve class balance; optionally repeated 3× for stability.
- Metrics:
  - Primary: Accuracy (to align with the paper’s headline metric).
  - Secondary: F1 (macro), ROC-AUC, and Brier score (optional) for calibration insights.
- Model selection: Hyperparameter tuning via nested cross-validation or a dedicated validation split within each training fold. For boosting models, use early stopping to mitigate overfitting.
- Seeds: Fixed random seeds for comparability; report seed in logs.

## Hyperparameter Ranges (search grids)

These ranges balance thoroughness with compute cost. Final choices will follow nested CV or Bayesian search if available.

- Logistic Regression
  - `C`: [0.01, 0.1, 1.0, 10.0]
  - `penalty`: ['l2'] (optionally ['l1','l2'] with `saga`)
  - `solver`: ['liblinear', 'saga']

- XGBoost
  - `n_estimators`: [300, 600, 1000]
  - `learning_rate`: [0.01, 0.05, 0.1, 0.2]
  - `max_depth`: [3, 5, 7, 10]
  - `subsample`: [0.7, 0.85, 1.0]
  - `colsample_bytree`: [0.6, 0.8, 1.0]
  - `reg_lambda`: [0, 1, 3, 5]
  - `reg_alpha`: [0, 0.5, 1.0]

- CatBoost
  - `iterations`: [500, 1000, 1500]
  - `learning_rate`: [0.01, 0.05, 0.1]
  - `depth`: [4, 6, 8, 10]
  - `l2_leaf_reg`: [1, 3, 5, 7]

We will record the best configuration per fold and report aggregate performance and variability.

## Statistical Significance and Confidence Intervals

To address credibility and quantify uncertainty, we will provide both confidence intervals and formal tests comparing the proposed ensemble with each baseline.

1. Confidence Intervals for Accuracy
   - Binomial/Wilson interval on a single held-out test set: For accuracy p on N samples, the Wilson 95% CI is:  
     p̂ ± z * sqrt(p̂(1−p̂)/N + z²/(4N²)) / (1 + z²/N), with z=1.96.
   - Bootstrap CI (recommended): 1000–5000 stratified bootstrap resamples of the test set; report the 2.5 and 97.5 percentiles of the accuracy distribution.
   - Cross-validation view: Report mean ± std across folds, plus a 95% CI on the mean (using t-distribution) for descriptive context.

2. Pairwise Significance Tests vs. Proposed Ensemble
   - McNemar’s Test (paired classification on the same test set):
     - Build a 2×2 table of disagreements between two models (b = A correct/B wrong, c = A wrong/B correct).
     - With continuity correction: X² = (|b − c| − 1)² / (b + c); report exact p-value.
     - Null: both models have the same error rate. We will test Ensemble vs. LR, Ensemble vs. XGB, Ensemble vs. CatBoost.
   - Corrected Repeated k-Fold t-Test (across folds):
     - Compute per-fold accuracy differences d_k.
     - Apply the Nadeau & Bengio correction for variance inflation due to overlap in train/test sets:  
       Var(d̄) ≈ (1/K + n_test/n_train) * s², where s² is the sample variance of d_k.
     - Report t-statistic and p-value for each pair.

3. Multiple Comparisons
   - Since we compare the ensemble against three baselines, we will control family-wise error using Holm–Bonferroni at α=0.05 across the three pairwise tests per window.

4. Effect Sizes
   - Alongside p-values, report absolute accuracy difference (ΔAcc) and, optionally, Cohen’s d for cross-validated differences to contextualize practical significance.

## Reporting Format in the Revised Paper

- Main text: For each window (10/20/30 min), include a table:

  | Model | Accuracy (95% CI) | F1 (macro) | ROC-AUC | ΔAcc vs Ensemble | p-value (adj.) | Significant? |
  |---|---|---|---|---|---|---|
  | Proposed Ensemble | 0.XXX [L, U] | 0.XXX | 0.XXX | — | — | — |
  | Logistic Regression | 0.XXX [L, U] | 0.XXX | 0.XXX | −0.0XX | 0.0XX | Yes/No |
  | XGBoost | 0.XXX [L, U] | 0.XXX | 0.XXX | −0.0XX | 0.0XX | Yes/No |
  | CatBoost | 0.XXX [L, U] | 0.XXX | 0.XXX | −0.0XX | 0.0XX | Yes/No |

- Appendix/Repo: Provide per-fold scores, tuned hyperparameters per fold, and full statistical outputs (contingency tables, bootstrap distributions, and t-test summaries).

## Reproducibility and Integration Notes

- Dependencies
  - scikit-learn (pip), xgboost, catboost, scipy/statsmodels (for tests), numpy, pandas.
- Implementation sketch
  - Add three new estimators using the existing training pipeline (`Pipeline` with common preprocessing). Ensure consistent StratifiedKFold splits and random_state across all models.
  - Save per-sample predictions for the test sets to enable McNemar’s test and bootstrap CIs.
  - Log all seeds, data hashes, and model parameters for traceability.

- Optional install commands (for documentation):
  - pip install scikit-learn xgboost catboost scipy statsmodels

- Minimal pseudo-code for the testing layer (illustrative):
  - McNemar: from statsmodels.stats.contingency_tables import mcnemar
  - Wilson CI: use statsmodels proportion_confint(method='wilson') or implement formula.
  - Bootstrap CI: resample indices with replacement; recompute accuracy; take percentile CI.

## Expected Outcome

The added baselines (LR, XGBoost, CatBoost) will provide clear reference points relative to the proposed ensemble. Confidence intervals will quantify uncertainty, and pairwise significance tests—corrected for multiple comparisons—will indicate whether observed improvements are statistically meaningful. All artifacts (configurations, metrics, and statistical outputs) will be committed to the repository for verification.
