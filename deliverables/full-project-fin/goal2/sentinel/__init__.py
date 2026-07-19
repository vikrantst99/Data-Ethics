"""The Autonomous Compliance Sentinel — Goal 2 baseline package.
Reusable, importable and unit-testable building blocks for the baseline model,
the risk tests and the fairness analysis.
"""
from .data import (load_proposals, make_split, get_xy, find_data_path,
                   TEXT_COLUMN, LABEL_COLUMN, TARGET_COLUMN, POSITIVE_LABEL)
from .baselines import rule_predict, majority_predict, RULE_TERMS
from .models import (make_tfidf_logreg, make_tfidf_nb, make_tfidf_xgb,
                     make_model, model_factory,
                     train_models, save_model, load_model, top_terms)
from .metrics import (binary_scores, confusion_counts, miss_rate,
                      threshold_sweep, group_metrics, eo_gaps)
from .pipeline import build_artifacts

__all__ = [
    "load_proposals", "make_split", "get_xy", "find_data_path",
    "TEXT_COLUMN", "LABEL_COLUMN", "TARGET_COLUMN", "POSITIVE_LABEL",
    "rule_predict", "majority_predict", "RULE_TERMS",
    "make_tfidf_logreg", "make_tfidf_nb", "make_tfidf_xgb", "make_model",
    "model_factory", "train_models", "save_model", "load_model", "top_terms",
    "binary_scores", "confusion_counts", "miss_rate", "threshold_sweep",
    "group_metrics", "eo_gaps", "build_artifacts",
]
__version__ = "1.0"
