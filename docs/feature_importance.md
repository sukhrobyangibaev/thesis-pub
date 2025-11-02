# Feature importance visualisations (by match phase)

This short utility produces permutation-importance and impurity-based (Gini) feature importance plots for Random Forest models at 10/20/30-minute phases.

- Input data: the script auto-detects phase CSVs from `part_8/new/`, `part_10_generate_train/dataframes/`, or the root `datasets/` directory (whichever exists first).
- Models reused (if present): `part_8/trained_models/<phase>/rf_classifier.pkl`.
- If no pre-trained model is found, it trains a small Random Forest on the fly for that phase.
- Outputs: `part_8/plots/feature_importance/` as PNG files

## How to run

Run the plotting script:

```bash
python part_8/feature_importance.py
```

This will generate the following files:

- `rf_gini_importance_10min.png`, `rf_gini_importance_20min.png`, `rf_gini_importance_30min.png`
- `rf_permutation_importance_10min.png`, `rf_permutation_importance_20min.png`, `rf_permutation_importance_30min.png`

Each figure shows the top 20 features for the given match phase.

## Notes

- Permutation importance is computed on a held-out test split (same seed/split as used in training scripts) to quantify each feature's contribution.
- Impurity-based (Gini) importance is provided for comparison and to align with conventional tree-based interpretations.
- If you prefer SHAP, install `shap` and use `shap.TreeExplainer` with the same model; permutation importance is dependency-free and should satisfy the reviewer.
