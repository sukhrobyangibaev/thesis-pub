# Cross-Validation Results and Significance Testing

This document addresses the reviewer feedback by reporting repeated stratified k-fold cross-validation results with standard deviations, 95% confidence intervals, and paired significance tests.

- Protocol: Repeated Stratified K-Fold, 5 folds × 3 repeats (15 resamples)
- Metric: Accuracy
- Datasets: 10-minute, 20-minute, 30-minute snapshots
- Models: CART, C4.5, Extra Trees, Gradient Boosting, Hist Gradient Boosting, Random Forest, AdaBoost
- Significance: Paired t-tests (two-sided) comparing the best model against others on the same resamples

Source files: `part_10_generate_train/cv_results.csv`, `cv_results.txt`, and (if SciPy installed) `cv_significance.csv`.

## Summary (best per dataset)

- 10 min: Hist Gradient Boosting — 0.6533 ± 0.0019 (95% CI [0.6523, 0.6543]); significant vs all others (all p-values < 1e-10)
- 20 min: Extra Trees — 0.7494 ± 0.0015 (95% CI [0.7486, 0.7502]); significant vs all others (all p-values < 1e-17)
- 30 min: Extra Trees — 0.8639 ± 0.0007 (95% CI [0.8636, 0.8643]); significant vs all others (all p-values < 1e-24)

## Full results (mean ± SD, 95% CI)

### 10-minute dataset

| Classifier                | Mean Acc | SD      | 95% CI              |
|---------------------------|----------|---------|---------------------|
| CART                      | 0.5751   | 0.0014  | [0.5744, 0.5758]    |
| C4.5                      | 0.5772   | 0.0010  | [0.5767, 0.5777]    |
| Extra Trees Classifier    | 0.6465   | 0.0015  | [0.6458, 0.6473]    |
| Gradient Boosting         | 0.6483   | 0.0016  | [0.6474, 0.6491]    |
| Hist Gradient Boosting    | 0.6533   | 0.0019  | [0.6523, 0.6543]    |
| Random Forest             | 0.6425   | 0.0016  | [0.6417, 0.6433]    |
| AdaBoost                  | 0.6464   | 0.0015  | [0.6456, 0.6472]    |

### 20-minute dataset

| Classifier                | Mean Acc | SD      | 95% CI              |
|---------------------------|----------|---------|---------------------|
| CART                      | 0.6438   | 0.0018  | [0.6429, 0.6447]    |
| C4.5                      | 0.6460   | 0.0013  | [0.6453, 0.6466]    |
| Extra Trees Classifier    | 0.7494   | 0.0015  | [0.7486, 0.7502]    |
| Gradient Boosting         | 0.7157   | 0.0021  | [0.7146, 0.7167]    |
| Hist Gradient Boosting    | 0.7200   | 0.0016  | [0.7192, 0.7208]    |
| Random Forest             | 0.7334   | 0.0015  | [0.7327, 0.7342]    |
| AdaBoost                  | 0.7147   | 0.0016  | [0.7139, 0.7155]    |

### 30-minute dataset

| Classifier                | Mean Acc | SD      | 95% CI              |
|---------------------------|----------|---------|---------------------|
| CART                      | 0.7195   | 0.0013  | [0.7188, 0.7202]    |
| C4.5                      | 0.7224   | 0.0018  | [0.7215, 0.7233]    |
| Extra Trees Classifier    | 0.8639   | 0.0007  | [0.8636, 0.8643]    |
| Gradient Boosting         | 0.7569   | 0.0013  | [0.7562, 0.7575]    |
| Hist Gradient Boosting    | 0.7628   | 0.0011  | [0.7622, 0.7634]    |
| Random Forest             | 0.8210   | 0.0012  | [0.8204, 0.8216]    |
| AdaBoost                  | 0.7462   | 0.0011  | [0.7456, 0.7467]    |

## Significance testing (paired t-tests)

- 10 min best (Hist Gradient Boosting) vs others: all p-values < 1e-10
- 20 min best (Extra Trees) vs others: all p-values < 1e-17
- 30 min best (Extra Trees) vs others: all p-values < 1e-24

See `part_10_generate_train/cv_significance.csv` for full statistics (t-statistics and exact p-values).

## Reproducibility

Run the cross-validation script from the repository root:

```powershell
python .\part_10_generate_train\train_cv.py
```

Outputs:
- `part_10_generate_train/cv_results.csv` and `cv_results.txt`
- If SciPy is installed: `part_10_generate_train/cv_significance.csv`

## Notes and limitations

- Cross-validation resamples the same dataset; while this stabilizes estimates, it is not a substitute for evaluation on independent data drawn from different distributions (e.g., patches/leagues).
- 95% CIs use a normal approximation with sample SD over resamples; this is standard but approximate.
- Hyperparameters were fixed to match prior experiments; we did not perform nested tuning to keep the protocol simple and compute-light. Rankings may change under tuning.
- P-values are uncorrected for multiple comparisons; future work can apply corrections (e.g., Holm–Bonferroni) or non-parametric alternatives with per-fold predictions.

## Suggested wording for the paper

We evaluated all models using repeated stratified k-fold cross-validation (5 folds × 3 repeats, 15 resamples) and report mean accuracy, standard deviation, and 95% confidence intervals. On the 10-minute dataset, Hist Gradient Boosting achieved 0.6533 ± 0.0019 (95% CI [0.6523, 0.6543]) and was significantly better than all baselines (paired t-tests, all p < 1e-10). On the 20-minute dataset, Extra Trees achieved 0.7494 ± 0.0015 (95% CI [0.7486, 0.7502]) with p < 1e-17 against all alternatives. On the 30-minute dataset, Extra Trees achieved 0.8639 ± 0.0007 (95% CI [0.8636, 0.8643]) with p < 1e-24 versus all alternatives. These results strengthen the reliability of our findings while reflecting the practical constraint of fixed hyperparameters and no nested tuning.
