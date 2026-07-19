"""Goal 3 — model analysis, XAI and automated weakness detection.

Builds on the Goal-2 `sentinel` package and its saved model artifacts.
"""
from .loader import (goal2_dir, load_sentinel, load_data, load_models,
                     load_test_frame)
from .analysis import (cross_val_recall, compare_models, calibration_bins,
                       error_indices)
from .explain import (global_importance_linear, top_terms_by_class,
                      ablation_importance, explain_instance_linear,
                      predict_proba_text)
from .weakness import (mask_terms, obfuscate, evasion_recall,
                       per_policy_recall, worst_policy,
                       negation_gap, negation_blind_count,
                       stopwords_without_negations,
                       NEGATIONS, NEGATION_PAIRS)

__all__ = [
    "goal2_dir", "load_sentinel", "load_data", "load_models", "load_test_frame",
    "cross_val_recall", "compare_models", "calibration_bins", "error_indices",
    "global_importance_linear", "top_terms_by_class",
    "ablation_importance", "explain_instance_linear",
    "predict_proba_text",
    "mask_terms", "obfuscate", "evasion_recall", "per_policy_recall", "worst_policy",
    "negation_gap", "negation_blind_count", "stopwords_without_negations",
    "NEGATIONS", "NEGATION_PAIRS",
]
__version__ = "1.0"
