# Visualising Feature Contributions Across Match Phases (Reviewer Response)

This note addresses the reviewer request to complement qualitative discussion with quantitative, visual explanations of model behaviour. We outline and provide code for SHAP- and permutation-based feature importance analyses for the ensemble decision tree models used in our DotA 2 match outcome prediction system, stratified by match phase snapshots (10, 20, 30 minutes).

## Why this matters
- Reinforces interpretability claims with reproducible figures.
- Highlights which in-game metrics drive predictions at different phases.
- Differentiates the work from purely accuracy-driven studies by exposing model reasoning and phase-specific dynamics.

## Scope and datasets
We evaluate the trained tree-based ensemble(s) used in the paper (e.g., Random Forest, Gradient Boosting, XGBoost if applicable) on the phase-specific datasets:
- 10-minute snapshot datasets: `datasets/10min_*.csv`
- 20-minute snapshot datasets: `datasets/20min_*.csv`
- 30-minute snapshot datasets: `datasets/30min_*.csv`

Models are expected under `trained_models/10min`, `trained_models/20min`, and `trained_models/30min` (any of `.pkl`, `.joblib`, or library-native formats). If your pipeline saves elsewhere, adjust the paths accordingly.

## Methods
### 1) Global feature importance via permutation importance
- Method: Randomly permute each feature and measure the drop in validation AUC (or chosen metric). Larger drops imply higher importance.
- Pros: Model-agnostic, robust to certain biases, easy to average across folds.
- Cons: Sensitive to correlated features (shared importance is split).

### 2) Global and local explanations via SHAP values
- Method: Shapley values attribute the prediction difference from a baseline to individual features. For tree ensembles, `shap.TreeExplainer` is efficient.
- Pros: Consistent local explanations; summary plots reveal global importance and directionality; dependence plots show non-linear and interaction effects.
- Cons: With highly correlated features, attributions can be shared; ensure careful interpretation.

## Outputs (figures we will produce)
For each phase (10/20/30 minutes) and each trained model:
- SHAP summary plot (beeswarm): `plots/interpretability/<phase>/<model>_shap_summary.png`
- SHAP bar plot: `plots/interpretability/<phase>/<model>_shap_bar.png`
- Optional SHAP dependence plots for top-k features: `plots/interpretability/<phase>/<model>_dep_<feature>.png`
- Permutation importance bar plot: `plots/interpretability/<phase>/<model>_perm_importance.png`

These figures are suitable for inclusion in the paper’s Results/Interpretability section, with phase-specific commentary.

## Reproducible analysis code (example)
Below is a self-contained example using scikit-learn models and SHAP. Adapt model loading as needed (e.g., XGBoost/LightGBM). This snippet demonstrates both permutation importance and SHAP.

```python
# file: examples/compute_importance_example.py (reference snippet)
import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

# --- CONFIG ---
PHASE = "10min"  # one of: "10min", "20min", "30min"
MODEL_PATH = f"trained_models/{PHASE}/model.pkl"  # adjust to your artifact
DATA_PATH = "datasets/10min_301475x46.csv"        # pick one phase-matched file or your consolidated features
TARGET_COL = "target"  # adjust to your actual target column name (e.g., win/loss label)
TOP_K = 15
OUT_DIR = f"plots/interpretability/{PHASE}"
os.makedirs(OUT_DIR, exist_ok=True)

# --- LOAD DATA & MODEL ---
df = pd.read_csv(DATA_PATH)
assert TARGET_COL in df.columns, f"Missing target column '{TARGET_COL}' in {DATA_PATH}"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].values

model = joblib.load(MODEL_PATH)

# Optional: if you have a validation split saved, use it here instead of in-sample.
# For demonstration, we'll compute metrics in-sample; prefer a held-out set in practice.

# --- METRIC BASELINE ---
y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
baseline_auc = roc_auc_score(y, y_pred_proba)
print(f"Baseline AUC: {baseline_auc:.4f}")

# --- PERMUTATION IMPORTANCE ---
perm = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=10, random_state=42)
importances_mean = perm.importances_mean
importances_std = perm.importances_std

perm_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances_mean,
    "std": importances_std,
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8, max(4, TOP_K * 0.3)))
perm_df.head(TOP_K).iloc[::-1].plot(
    x="feature", y="importance", kind="barh", legend=False, color="#3b82f6"
)
plt.xlabel("Permutation importance (Δ AUC)")
plt.ylabel("Feature")
plt.title(f"Permutation importance — {PHASE}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "perm_importance.png"), dpi=200)
plt.close()

# --- SHAP VALUES (TreeExplainer for tree ensembles) ---
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# If model is binary classifier, shap_values may be list-like. Use the positive class.
if isinstance(shap_values, list):
    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

# Summary beeswarm
plt.figure()
shap.summary_plot(shap_values, X, show=False, max_display=TOP_K)
plt.title(f"SHAP summary — {PHASE}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_summary.png"), dpi=200)
plt.close()

# Summary bar
plt.figure()
shap.summary_plot(shap_values, X, show=False, plot_type="bar", max_display=TOP_K)
plt.title(f"SHAP bar — {PHASE}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_bar.png"), dpi=200)
plt.close()

# Optional: dependence plots for top-k features
# Compute mean |SHAP| per feature to pick top features
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1][:min(TOP_K, X.shape[1])]
top_features = X.columns[top_idx]

for feat in top_features[:5]:  # up to 5 dependence plots
    plt.figure()
    shap.dependence_plot(feat, shap_values, X, show=False)
    plt.title(f"SHAP dependence — {PHASE} — {feat}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_dep_{feat}.png"), dpi=200)
    plt.close()

print(f"Figures saved to: {OUT_DIR}")
```

Notes:
- For XGBoost/LightGBM models, `TreeExplainer` works efficiently; for HistGradientBoosting, use `shap.Explainer(model, X)` as a fallback.
- Prefer evaluation on a held-out validation set (or cross-validation folds) rather than the training data to avoid optimistic importances.

## Phase-specific protocol
To emphasise dynamics across match phases:
1) Run the analysis separately for 10, 20, and 30 minutes using phase-matched models and datasets.
2) For each phase, report:
   - Top 10 features by permutation importance (ΔAUC) with 95% CI from repeats.
   - SHAP summary plot highlighting both magnitude and directionality (color scale matches feature values).
3) Provide a short interpretation per phase notedly answering:
   - Which metrics dominate early (10 min)?
   - Which mid-game signals escalate (20 min)?
   - Which late-game features consolidate predictions (30 min)?

This structured comparison directly addresses the reviewer’s request for phase-wise dominance patterns.

## Recommended reporting (paper integration)
- Main text: 1 combined panel with three SHAP summary plots (10/20/30 min) or a small multiples grid, plus a permutation importance bar chart appendix per phase.
- Caption: State model, data slice, metric for permutation importance, and note any strong correlations among top features.
- Brief discussion: Contrast shifts in importance across phases (e.g., early economy vs. mid-game objectives vs. late-game team composition effects). Avoid over-claiming causality; emphasize predictive contribution.

## Reliability checks
- Stability across folds: compute permutation importance on each validation fold and average; add error bars.
- Seed sensitivity: repeat with different random seeds; report variation for top features.
- Correlation clusters: report when top features are highly correlated; interpret at the group level (e.g., economy-related metrics).
- Sanity tests: verify that permuting obviously irrelevant/ID columns yields near-zero importance.

## Dependencies
Install the following if not already present:
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`, `shap`

On Windows (PowerShell), for reference:
```powershell
pip install pandas numpy scikit-learn matplotlib joblib shap
```

## Optional: integrate into training pipeline
If desired, add a small module (e.g., `part_10_generate_train/interpretability.py`) that:
- Loads the latest trained model artifact per phase.
- Loads the corresponding validation split (or reconstructs it deterministically).
- Produces the figures listed above into `plots/interpretability/<phase>/`.
- Optionally writes CSVs with numeric importances (SHAP mean |value|, permutation ΔAUC, and bootstrap CIs) for archival.

This keeps interpretability artifacts versioned alongside models.

## Limitations and caveats
- Correlated features share importance; rank ordering may be unstable among highly collinear metrics.
- SHAP values explain model predictions, not causal effects; interpret contributions as predictive associations.
- For extremely large datasets, SHAP computation can be slow; consider subsampling the validation set while preserving class balance.

## References
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions (NeurIPS).
- Molnar (2022). Interpretable Machine Learning.
- Fisher, Rudin, Dominici (2019). All Models are Wrong, but Many are Useful: Variable Importance for Black-Box, Proprietary, or Misspecified Prediction Models.
