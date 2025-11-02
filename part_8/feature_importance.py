import os
import pickle
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PHASES: List[str] = ["10min", "20min", "30min"]

# We'll search for phase-specific CSVs in these locations (first match wins)
DATA_CANDIDATE_DIRS: List[Path] = [
    Path("part_8/new"),                 # used by part_8/train.py historically
    Path("part_10_generate_train/dataframes"),
    Path("datasets"),                   # large canonical datasets in repo root
]

MODELS_DIR = Path("trained_models")  # optional: if present, we'll reuse models
OUT_DIR = Path("part_8/plots/feature_importance")


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    feature_names = df.columns[:-1].tolist()
    X = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values
    # Note: labels were label-encoded in training; tree models work with any label encoding for importance
    return df, X, y, feature_names


def make_split(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Match training split for fair permutation importance on held-out data
    return train_test_split(X, y, test_size=0.1, random_state=1)


def topk(arr: np.ndarray, names: List[str], k: int = 20) -> Tuple[np.ndarray, List[str]]:
    order = np.argsort(arr)[::-1]
    order = order[: min(k, len(order))]
    return arr[order], [names[i] for i in order]


def plot_bar(values: np.ndarray, labels: List[str], title: str, outfile: Path) -> None:
    plt.figure(figsize=(10, max(4, 0.35 * len(labels))))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, values, color="#4C72B0")
    plt.yticks(y_pos, labels)
    plt.gca().invert_yaxis()  # largest at top
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def find_csv_for_phase(phase: str) -> Path:
    """Find a CSV for the given phase, preferring 46-feature datasets."""
    # Common filename fragments to try (broad to narrow)
    patterns = [
        f"{phase}_*x46*_samples.csv",
        f"{phase}_*x46*.csv",
        f"{phase}_*.csv",
    ]
    for base in DATA_CANDIDATE_DIRS:
        for pat in patterns:
            matches = sorted(base.glob(pat))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"No CSV found for phase {phase} under {DATA_CANDIDATE_DIRS}")


def compute_and_plot_for_phase(phase: str) -> None:
    model_path = MODELS_DIR / phase / "rf_classifier.pkl"
    csv_path = find_csv_for_phase(phase)

    print(f"[load] Data: {csv_path}")
    _, X, y, feature_names = load_data(csv_path)
    X_train, X_test, y_train, y_test = make_split(X, y)

    model = None
    if model_path.exists():
        print(f"[load] Model: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        # Fallback: train a quick RandomForest if a pre-trained model isn't available
        from sklearn.ensemble import RandomForestClassifier
        print(f"[train] No pre-trained model found for {phase}. Training a fresh RandomForest (n=100)...")
        model = RandomForestClassifier(n_estimators=100, criterion="gini", random_state=1, n_jobs=-1)
        model.fit(X_train, y_train)

    ensure_out_dir()

    # 1) Impurity-based (Gini) importances from RandomForest
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_)
        imp_top, names_top = topk(imp, feature_names, k=20)
        out_file = OUT_DIR / f"rf_gini_importance_{phase}.png"
        plot_bar(imp_top, names_top, f"Random Forest (Gini) Feature Importance — {phase}", out_file)
        print(f"[ok] Saved: {out_file}")
    else:
        print(f"[warn] Model has no feature_importances_: {type(model)}")

    # 2) Permutation importance on held-out test set
    # Some pre-trained models were fit on label-encoded y (e.g., 0/1),
    # while current CSVs often contain string labels (e.g., 'dire'/'radiant').
    # Align types to avoid sklearn metric errors.
    y_test_for_perm = y_test
    try:
        if hasattr(model, "classes_"):
            classes_dtype = np.asarray(model.classes_).dtype
            y_dtype = np.asarray(y_test).dtype
            # If model expects numeric classes but y_test is strings/objects, encode y_test
            if np.issubdtype(classes_dtype, np.number) and not np.issubdtype(y_dtype, np.number):
                le = LabelEncoder()
                le.fit(y_train)
                y_test_for_perm = le.transform(y_test)
                print("[align] Encoded y_test to numeric to match model classes for permutation importance.")
    except Exception as e:
        print(f"[warn] Could not auto-align label types for permutation importance: {e}. Proceeding with original labels.")

    print(f"[info] Computing permutation importance (this may take ~seconds)...")
    perm = permutation_importance(model, X_test, y_test_for_perm, n_repeats=10, random_state=1, n_jobs=-1)
    perm_mean = perm.importances_mean
    perm_top, perm_names_top = topk(perm_mean, feature_names, k=20)
    out_file_perm = OUT_DIR / f"rf_permutation_importance_{phase}.png"
    plot_bar(perm_top, perm_names_top, f"Random Forest (Permutation) Feature Importance — {phase}", out_file_perm)
    print(f"[ok] Saved: {out_file_perm}")


def main() -> None:
    for phase in PHASES:
        try:
            compute_and_plot_for_phase(phase)
        except Exception as e:
            print(f"[error] Phase {phase}: {e}")


if __name__ == "__main__":
    main()
