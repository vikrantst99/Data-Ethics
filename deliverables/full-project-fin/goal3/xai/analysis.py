# Model analysis: cross-validation, model comparison, calibration, error slices

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold


def cross_val_recall(model, X, y, cv=5, random_state=42):
    # Red-Flag recall with stratified k-fold CV (more trustworthy than one split)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring="recall")
    return scores


def compare_models(models, X, y, cv=5):
    # Mean and spread of the cross-validated recall for each model
    rows = []
    for name, model in models.items():
        scores = cross_val_recall(model, X, y, cv=cv)
        rows.append({
            "model": name,
            "cv_recall_mean": scores.mean(),
            "cv_recall_std": scores.std(),
        })
    return pd.DataFrame(rows).set_index("model").round(3)


def calibration_bins(y_true, proba, n_bins=10):
    # Reliability table: predicted probability vs. the real Red-Flag rate, per bin
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    edges = np.linspace(0, 1, n_bins + 1)
    # For every probability, find which bin it falls into (0 ... n_bins-1)
    bin_index = np.clip(np.digitize(proba, edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        in_bin = bin_index == b
        if in_bin.any():
            rows.append({
                "bin_mid": (edges[b] + edges[b + 1]) / 2,
                "mean_proba": proba[in_bin].mean(),
                "observed_rate": y_true[in_bin].mean(),
                "n": int(in_bin.sum()),
            })
    return pd.DataFrame(rows)


def error_indices(y_true, y_pred, kind="fn"):
    # Positions of false negatives ('fn') or false positives ('fp')
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if kind == "fn":
        mask = (y_true == 1) & (y_pred == 0)
    elif kind == "fp":
        mask = (y_true == 0) & (y_pred == 1)
    else:
        raise ValueError("kind must be 'fn' or 'fp'")
    return np.where(mask)[0]
