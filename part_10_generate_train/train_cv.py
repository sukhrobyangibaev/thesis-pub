import time
import os
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

# Optional: significance testing if SciPy is available
try:
    from scipy import stats  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# -------------------------------
# Configuration
# -------------------------------
FOLDERNAMES = ['10min', '20min', '30min']
FILENAMES = ['10min_414939x46.csv', '20min_378924x46.csv', '30min_516693x46.csv']
# Resolve paths relative to this file to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Use the repository-level datasets directory
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'datasets'))
OUT_DIR = BASE_DIR  # write results next to this script
N_SPLITS = 5
N_REPEATS = 3
CV_RANDOM_STATE = 42
SCORING = 'accuracy'

# -------------------------------
# Helpers
# -------------------------------

def ci95_from_scores(scores: np.ndarray) -> tuple[float, float]:
    """Return (mean, half_width) for a normal-approx 95% CI.
    Bounds can be clipped to [0,1] by the caller if desired.
    """
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1))  # sample std
    n = len(scores)
    if n <= 1:
        return mean, 0.0
    half_width = 1.96 * (std / math.sqrt(n))
    return mean, half_width


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -------------------------------
# Main
# -------------------------------

def main() -> None:
    start_time = time.time()

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=CV_RANDOM_STATE
    )

    # Define classifiers as close as possible to the existing training script
    classifiers = {
        'CART': DecisionTreeClassifier(criterion='gini', random_state=1),
        'C4.5': DecisionTreeClassifier(criterion='entropy', random_state=1),
        'Extra Trees Classifier': ExtraTreesClassifier(
            criterion='entropy', n_estimators=150, random_state=1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            loss='log_loss', n_estimators=50, learning_rate=1, criterion='friedman_mse', max_depth=4, random_state=1
        ),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(
            learning_rate=0.2, max_iter=100, random_state=1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=50, criterion='gini', random_state=1
        ),
        'Adaboost': AdaBoostClassifier(
            n_estimators=50, learning_rate=0.1, algorithm='SAMME', random_state=0
        ),
    }

    rows = []  # aggregated results across datasets
    significance_rows = []  # optional pairwise tests per dataset

    print(f"Cross-validation: {N_SPLITS} folds x {N_REPEATS} repeats (stratified)")

    for i_f, filename in enumerate(FILENAMES):
        dataset_tag = FOLDERNAMES[i_f]
        csv_path = os.path.join(DATA_DIR, filename)
        print(f"\nDataset: {dataset_tag} -> {csv_path}")

        df = pd.read_csv(csv_path)
        X = df.iloc[:, 0:-1].values
        y_raw = df.iloc[:, -1].values

        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        # Keep per-classifier scores to enable paired tests (same CV splitter)
        clf_scores: dict[str, np.ndarray] = {}

        for clf_name, clf in classifiers.items():
            # cross_val_score returns array of shape (n_splits * n_repeats,)
            scores = cross_val_score(
                clf, X, y, cv=splitter, scoring=SCORING, n_jobs=-1
            )
            clf_scores[clf_name] = scores
            mean, half = ci95_from_scores(scores)
            lower = max(0.0, mean - half)
            upper = min(1.0, mean + half)
            std = float(np.std(scores, ddof=1))
            rows.append({
                'dataset': dataset_tag,
                'classifier': clf_name,
                'n_folds': N_SPLITS,
                'n_repeats': N_REPEATS,
                'n_scores': len(scores),
                'mean_accuracy': mean,
                'std_accuracy': std,
                'ci95_lower': lower,
                'ci95_upper': upper,
            })
            print(f"  {clf_name:24s} mean={mean:.4f} std={std:.4f} 95%CI=[{lower:.4f},{upper:.4f}]")

        # Optional significance testing: compare best vs others (paired t-test)
        if SCIPY_AVAILABLE and len(clf_scores) >= 2:
            # Identify best by mean accuracy
            best_name = max(clf_scores.items(), key=lambda kv: np.mean(kv[1]))[0]
            best_scores = clf_scores[best_name]
            for other_name, other_scores in clf_scores.items():
                if other_name == best_name:
                    continue
                # Paired t-test across same CV folds
                t_stat, p_val = stats.ttest_rel(best_scores, other_scores)
                significance_rows.append({
                    'dataset': dataset_tag,
                    'best_classifier': best_name,
                    'other_classifier': other_name,
                    't_statistic': float(t_stat),
                    'p_value': float(p_val),
                    'n_pairs': len(best_scores)
                })

    # Save outputs
    ensure_dir(OUT_DIR)
    results_csv = os.path.join(OUT_DIR, 'cv_results.csv')
    results_txt = os.path.join(OUT_DIR, 'cv_results.txt')
    pd.DataFrame(rows).to_csv(results_csv, index=False)

    with open(results_txt, 'w', encoding='utf-8') as f:
        f.write(f"Cross-validation: {N_SPLITS} folds x {N_REPEATS} repeats (stratified)\n")
        f.write(f"Scoring: {SCORING}\n\n")
        for r in rows:
            f.write(
                f"{r['dataset']:>5s} | {r['classifier']:<24s} | mean={r['mean_accuracy']:.4f} "
                f"std={r['std_accuracy']:.4f} 95%CI=[{r['ci95_lower']:.4f},{r['ci95_upper']:.4f}]\n"
            )

    if SCIPY_AVAILABLE and significance_rows:
        sig_csv = os.path.join(OUT_DIR, 'cv_significance.csv')
        pd.DataFrame(significance_rows).to_csv(sig_csv, index=False)

    elapsed = time.time() - start_time
    m, s = divmod(elapsed, 60)
    print(f"\nDone. Wrote: {results_csv} and {results_txt}")
    if SCIPY_AVAILABLE and significance_rows:
        print(f"Also wrote: cv_significance.csv (paired t-tests best vs others per dataset)")
    print(f"Execution time: {int(m)} minutes and {s:.1f} seconds")


if __name__ == '__main__':
    main()
