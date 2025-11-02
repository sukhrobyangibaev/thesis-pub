# Classifier Hyperparameters and Rationale

This note addresses reviewer feedback regarding missing quantitative configuration details for the tree-ensemble models. It documents the hyperparameters actually used in training (see `part_10_generate_train/train.py`) and explains the rationale and interpretability/accuracy trade-offs.

Scope:
- Extra Trees Classifier
- Random Forest Classifier
- Histogram-based Gradient Boosting Classifier

For parameters not listed below, scikit-learn defaults were used for the project’s installed version.

## Extra Trees Classifier

Used in code:
- classifier: `sklearn.ensemble.ExtraTreesClassifier`
- hyperparameters explicitly set:
  - `criterion = 'entropy'`
  - `n_estimators = 150`
  - `random_state = 1`
- other hyperparameters: scikit-learn defaults

Rationale and trade-offs:
- Number of estimators (150): increases variance reduction and typically improves accuracy up to a point; 150 balances accuracy gains with training time and memory.
- Entropy criterion: information-gain splits can be slightly more discriminative than Gini in some datasets, at a modest compute cost.
- Interpretability: as a large ensemble, individual-tree interpretability is low. Global feature importance (impurity- or permutation-based) remains available. Per-sample explanations can be produced with post-hoc methods (e.g., SHAP) if needed.

## Random Forest Classifier

Used in code:
- classifier: `sklearn.ensemble.RandomForestClassifier`
- hyperparameters explicitly set:
  - `n_estimators = 50`
  - `criterion = 'gini'`
  - `random_state = 1`
- other hyperparameters: scikit-learn defaults

Rationale and trade-offs:
- Number of estimators (50): a compact forest that provides strong baseline accuracy while keeping runtime smaller than larger ensembles; suitable for repeated experiments.
- Gini criterion: computationally efficient and commonly comparable to entropy in classification accuracy for tabular data.
- Interpretability: similar to Extra Trees—global importance is available; single-tree paths are readable but not representative of the ensemble as a whole.

## Histogram-based Gradient Boosting Classifier

Used in code:
- classifier: `sklearn.ensemble.HistGradientBoostingClassifier`
- hyperparameters explicitly set:
  - `learning_rate = 0.2`
  - `max_iter = 100`
  - `random_state = 1`
- other hyperparameters: scikit-learn defaults

Rationale and trade-offs:
- Learning rate (0.2) with 100 iterations: a moderate rate to converge faster than 0.1 while mitigating overfitting relative to larger rates; 100 boosting iterations provides a practical cap on training time with strong accuracy.
- Histogram-based trees: scale well on medium-to-large tabular datasets; often yield top accuracy among tree ensembles but reduce direct interpretability.
- Interpretability: boosting further reduces direct transparency. To explain results, prefer permutation feature importance and local explainers (e.g., SHAP TreeExplainer) on representative samples.

## Reproducibility and Defaults
- All three models set `random_state = 1` to stabilize training/testing splits and model stochasticity.
- Hyperparameters not listed here use scikit-learn defaults for the project’s environment. Exact defaults can vary across scikit-learn versions; consult the installed version’s documentation if replicating results in a different environment.

## Why these settings?
- Practical balance: the project prioritizes strong baseline accuracy with reasonable runtime to enable multiple experiments over several time windows (10/20/30 minutes).
- Conventional choices: estimator counts (50–150) and gradient boosting settings (learning rate ≈ 0.1–0.2, iterations ≈ 100) are widely used starting points for tabular classification.
- Interpretability: a single decision tree (CART/C4.5) remains the most interpretable model reported; ensembles provide accuracy gains at the cost of transparency. We mitigate this via feature importance reporting and recommend local explanations for critical decisions.

## Future Enhancements
- Systematic tuning: k-fold cross-validation with grid/random/Bayesian search over key hyperparameters (e.g., `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, `learning_rate`, `max_leaf_nodes`).
- Version pinning: lock scikit-learn version to make defaults unambiguous; record exact package versions in a `requirements.txt`/`pyproject.toml`.
- Explainability appendix: include permutation feature importance plots and a small set of SHAP summaries to quantify feature effects.
