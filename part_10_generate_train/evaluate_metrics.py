import os
import math
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)
from sklearn.calibration import CalibrationDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
)
import matplotlib.pyplot as plt

# -------------------------------
# Configuration (aligned with train_cv.py)
# -------------------------------
FOLDERNAMES = ['10min', '20min', '30min']
FILENAMES = ['10min_414939x46.csv', '20min_378924x46.csv', '30min_516693x46.csv']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'datasets'))
OUT_DIR = os.path.join(BASE_DIR, 'metrics')
N_SPLITS = 5
N_REPEATS = 3
CV_RANDOM_STATE = 42

# Classifiers (same as in train_cv.py)
CLASSIFIERS: Dict[str, object] = {
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
        n_estimators=50, criterion='gini', random_state=1, n_jobs=-1
    ),
    'Adaboost': AdaBoostClassifier(
        n_estimators=50, learning_rate=0.1, algorithm='SAMME', random_state=0
    ),
}

# -------------------------------
# Helpers
# -------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ci95_from_scores(scores: np.ndarray) -> Tuple[float, float, float, float, float]:
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0
    n = scores.size
    half_width = 1.96 * (std / math.sqrt(n)) if n > 1 else 0.0
    return mean, std, max(0.0, mean - half_width), min(1.0, mean + half_width), half_width


def plot_confusion_matrix(cm_norm: np.ndarray, class_names: List[str], title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=150)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks([0, 1], labels=class_names)
    ax.set_yticks([0, 1], labels=class_names)
    for (i, j), v in np.ndenumerate(cm_norm):
        ax.text(j, i, f"{v:.2f}", ha='center', va='center', color='black', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Normalized rate')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc_val: float, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=150)
    ax.plot(fpr, tpr, label=f'AUC = {auc_val:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, y_proba: np.ndarray, title: str, out_path: str, n_bins: int = 10) -> float:
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=150)
    disp = CalibrationDisplay.from_predictions(y_true, y_proba, n_bins=n_bins, strategy='uniform', ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    # Brier score (lower is better)
    # Assume positive class label is 1
    brier = brier_score_loss(y_true, y_proba, pos_label=1)
    return float(brier)


# -------------------------------
# Core evaluation
# -------------------------------

def evaluate_dataset(dataset_tag: str, csv_path: str) -> None:
    # Load data
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = [str(c) for c in le.classes_]

    # Class balance
    counts = np.bincount(y)
    proportions = counts / counts.sum()

    # Output dirs
    ds_dir = os.path.join(OUT_DIR, dataset_tag)
    ensure_dir(ds_dir)
    plots_dir = os.path.join(ds_dir, 'plots')
    ensure_dir(plots_dir)

    # Save class balance
    balance_csv = os.path.join(ds_dir, 'class_balance.csv')
    pd.DataFrame({
        'class': class_names,
        'count': counts,
        'proportion': proportions,
    }).to_csv(balance_csv, index=False)

    # Repeated CV metrics (means/std/CI across folds*repeats)
    rep_cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=CV_RANDOM_STATE)

    agg_rows: List[Dict[str, object]] = []

    for clf_name, clf in CLASSIFIERS.items():
        accs: List[float] = []
        aucs: List[float] = []
        briers: List[float] = []
        # macro precision/recall/f1 per split
        mac_prec: List[float] = []
        mac_rec: List[float] = []
        mac_f1: List[float] = []

        for train_idx, test_idx in rep_cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = clf
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Probability of positive class (assumed class label 1)
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                if y_prob.shape[1] == 2:
                    pos_proba = y_prob[:, 1]
                else:
                    # fallback: one-vs-rest of the max class
                    pos_proba = np.max(y_prob, axis=1)
            elif hasattr(model, 'decision_function'):
                # Convert decision function to [0,1] via logistic-like mapping
                scores = model.decision_function(X_test)
                # Min-max scale as a fallback; not ideal but keeps pipeline running
                s_min, s_max = scores.min(), scores.max()
                pos_proba = (scores - s_min) / (s_max - s_min + 1e-9)
            else:
                # No probability; use predicted labels as pseudo-probability
                pos_proba = (y_pred == 1).astype(float)

            accs.append(accuracy_score(y_test, y_pred))
            # Binary ROC AUC if 2 classes; otherwise macro ovo
            try:
                if len(class_names) == 2:
                    aucs.append(roc_auc_score(y_test, pos_proba))
                else:
                    aucs.append(roc_auc_score(y_test, y_prob, multi_class='ovo', average='macro'))  # type: ignore
            except Exception:
                # AUC may fail if only one class present in a fold
                pass

            try:
                briers.append(brier_score_loss(y_test, pos_proba, pos_label=1))
            except Exception:
                pass

            p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
            mac_prec.append(float(p))
            mac_rec.append(float(r))
            mac_f1.append(float(f1))

        # Aggregate stats
        for metric_name, values in [
            ('accuracy', np.array(accs, dtype=float)),
            ('roc_auc', np.array(aucs, dtype=float) if len(aucs) else np.array([], dtype=float)),
            ('brier', np.array(briers, dtype=float) if len(briers) else np.array([], dtype=float)),
            ('precision_macro', np.array(mac_prec, dtype=float)),
            ('recall_macro', np.array(mac_rec, dtype=float)),
            ('f1_macro', np.array(mac_f1, dtype=float)),
        ]:
            if values.size:
                mean, std, lo, hi, _ = ci95_from_scores(values)
                agg_rows.append({
                    'dataset': dataset_tag,
                    'classifier': clf_name,
                    'metric': metric_name,
                    'n_splits': N_SPLITS,
                    'n_repeats': N_REPEATS,
                    'n_values': int(values.size),
                    'mean': mean,
                    'std': std,
                    'ci95_lower': lo,
                    'ci95_upper': hi,
                })

    agg_csv = os.path.join(ds_dir, 'cv_metrics_summary.csv')
    pd.DataFrame(agg_rows).to_csv(agg_csv, index=False)

    # Single 5-fold OOF predictions for diagnostic plots and per-class metrics
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)

    diag_rows: List[Dict[str, object]] = []

    for clf_name, clf in CLASSIFIERS.items():
        y_oof = np.full_like(y, fill_value=-1)
        proba_oof = np.full(y.shape[0], fill_value=np.nan, dtype=float)

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = clf
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_oof[test_idx] = y_pred

            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                pos_proba = y_prob[:, 1] if y_prob.shape[1] >= 2 else np.max(y_prob, axis=1)
            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(X_test)
                s_min, s_max = scores.min(), scores.max()
                pos_proba = (scores - s_min) / (s_max - s_min + 1e-9)
            else:
                pos_proba = (y_pred == 1).astype(float)
            proba_oof[test_idx] = pos_proba

        # Safety checks
        mask_valid = (y_oof != -1) & (~np.isnan(proba_oof))
        y_valid = y[mask_valid]
        y_pred_valid = y_oof[mask_valid]
        proba_valid = proba_oof[mask_valid]

        # Per-class metrics from OOF predictions
        p_c, r_c, f1_c, support_c = precision_recall_fscore_support(
            y_valid, y_pred_valid, average=None, zero_division=0
        )
        # Confusion matrix normalized by true labels
        cm = confusion_matrix(y_valid, y_pred_valid)
        with np.errstate(all='ignore'):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

        # ROC/AUC (binary assumed)
        auc_val = np.nan
        fpr, tpr = None, None  # type: ignore
        if len(class_names) == 2:
            try:
                auc_val = roc_auc_score(y_valid, proba_valid)
                fpr, tpr, _ = roc_curve(y_valid, proba_valid)
            except Exception:
                pass

        # Calibration plot and Brier score
        brier_val = np.nan
        try:
            brier_val = brier_score_loss(y_valid, proba_valid, pos_label=1)
        except Exception:
            pass

        # Save plots
        base = f"{dataset_tag}__{clf_name.replace(' ', '_').replace('.', '')}"
        # Confusion matrix
        cm_path = os.path.join(plots_dir, f"{base}__confusion_matrix.png")
        plot_confusion_matrix(cm_norm, class_names, f"{dataset_tag} - {clf_name}\nConfusion matrix (normalized)", cm_path)

        # ROC curve
        if fpr is not None and tpr is not None and not np.isnan(auc_val):
            roc_path = os.path.join(plots_dir, f"{base}__roc.png")
            plot_roc(fpr, tpr, float(auc_val), f"{dataset_tag} - {clf_name}\nROC curve", roc_path)

        # Calibration curve
        calib_path = os.path.join(plots_dir, f"{base}__calibration.png")
        brier_from_plot = plot_calibration(y_valid, proba_valid, f"{dataset_tag} - {clf_name}\nCalibration (reliability)", calib_path)
        # Prefer calculated brier if valid
        if not np.isnan(brier_val):
            brier_use = float(brier_val)
        else:
            brier_use = float(brier_from_plot)

        # Persist diagnostics rows
        # Per-class metrics rows
        for idx, cname in enumerate(class_names):
            diag_rows.append({
                'dataset': dataset_tag,
                'classifier': clf_name,
                'type': 'per_class',
                'class': cname,
                'precision': float(p_c[idx]),
                'recall': float(r_c[idx]),
                'f1': float(f1_c[idx]),
                'support': int(support_c[idx]),
            })

        # Overall metrics rows
        diag_rows.append({
            'dataset': dataset_tag,
            'classifier': clf_name,
            'type': 'overall',
            'accuracy': float(accuracy_score(y_valid, y_pred_valid)),
            'roc_auc': float(auc_val) if not np.isnan(auc_val) else np.nan,
            'brier': brier_use,
        })

        # Save confusion matrix values
        cm_csv_path = os.path.join(ds_dir, f"{base}__confusion_matrix.csv")
        pd.DataFrame(cm, index=[f"true_{c}" for c in class_names], columns=[f"pred_{c}" for c in class_names]).to_csv(cm_csv_path)
        cmn_csv_path = os.path.join(ds_dir, f"{base}__confusion_matrix_normalized.csv")
        pd.DataFrame(cm_norm, index=[f"true_{c}" for c in class_names], columns=[f"pred_{c}" for c in class_names]).to_csv(cmn_csv_path)

    diag_csv = os.path.join(ds_dir, 'oof_diagnostics.csv')
    pd.DataFrame(diag_rows).to_csv(diag_csv, index=False)


def main() -> None:
    start = time.time()
    ensure_dir(OUT_DIR)

    for folder, filename in zip(FOLDERNAMES, FILENAMES):
        csv_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(csv_path):
            print(f"Skipping {folder}: {csv_path} not found")
            continue
        print(f"\nEvaluating dataset {folder}: {csv_path}")
        evaluate_dataset(folder, csv_path)

    elapsed = time.time() - start
    m, s = divmod(elapsed, 60)
    print(f"\nDone. Metrics and plots written to: {OUT_DIR}")
    print(f"Execution time: {int(m)} minutes and {s:.1f} seconds")


if __name__ == '__main__':
    main()
