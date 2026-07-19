"""Metrics for the evaluation, the risk tests and the fairness analysis."""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)


def binary_scores(y_true, y_pred, proba=None):
    # Return the common scores. Recall on Red Flags is the most important one
    scores = {}
    scores["accuracy"] = accuracy_score(y_true, y_pred)
    scores["precision"] = precision_score(y_true, y_pred, zero_division=0)
    scores["recall"] = recall_score(y_true, y_pred, zero_division=0)
    scores["f1"] = f1_score(y_true, y_pred, zero_division=0)
    if proba is not None:
        scores["roc_auc"] = roc_auc_score(y_true, proba)
    else:
        scores["roc_auc"] = np.nan
    return scores


def confusion_counts(y_true, y_pred):
    # Return the four confusion-matrix numbers (tn, fp, fn, tp) as plain ints
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def miss_rate(y_true, y_pred):
    # Share of the real violations that we missed (= 1 - recall). Our key error
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    denom = fn + tp
    if denom == 0:
        return float("nan")
    return fn / denom


def threshold_sweep(y_true, proba, thresholds=None):
    # Try many decision thresholds and record miss-rate and false-positive-rate
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 33)

    rows = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)

        if (fn + tp) == 0:
            miss = np.nan
        else:
            miss = fn / (fn + tp)

        if (fp + tn) == 0:
            fpr = np.nan
        else:
            fpr = fp / (fp + tn)

        rows.append({"threshold": float(t), "miss_rate": miss, "fpr": fpr})
    return pd.DataFrame(rows)


# Fairness

def group_metrics(frame, col, y_true="y_true", y_pred="y_pred"):
    # Recall, false-positive-rate, precision and selection rate for each group.

    for c in [col, y_true, y_pred]:
        if c not in frame.columns:
            raise KeyError("Column '" + str(c) + "' missing from frame")

    rows = []
    for group_value, sub in frame.groupby(col):
        yt = sub[y_true].to_numpy()
        yp = sub[y_pred].to_numpy()
        has_pos = (yt == 1).any()
        has_neg = (yt == 0).any()

        if has_pos:
            recall = recall_score(yt, yp, zero_division=np.nan)
        else:
            recall = np.nan

        if has_neg:
            fpr = (yp[yt == 0] == 1).mean()
        else:
            fpr = np.nan

        if (yp == 1).any():
            precision = precision_score(yt, yp, zero_division=np.nan)
        else:
            precision = np.nan

        row = {}
        row[col] = group_value
        row["n"] = len(sub)
        row["n_pos"] = int((yt == 1).sum())
        row["recall_TPR"] = recall
        row["FPR"] = fpr
        row["precision"] = precision
        row["selection_rate"] = yp.mean()
        rows.append(row)

    return pd.DataFrame(rows).set_index(col)


def eo_gaps(frame, col):
    # Biggest minus smallest value per metric across the groups (smaller = fairer)
    m = group_metrics(frame, col)
    gaps = {}
    gaps["TPR gap"] = m["recall_TPR"].max() - m["recall_TPR"].min()
    gaps["FPR gap"] = m["FPR"].max() - m["FPR"].min()
    gaps["selection-rate gap"] = m["selection_rate"].max() - m["selection_rate"].min()
    return pd.Series(gaps)
