"""
Certify the model with the Goal-3 weakness tests before the agent ships.

This is the Section-3 option "Goal-3 weakness pre-gate": the agent should only
run on a model that passes the robustness checks. We reuse the Goal-3 `xai`
package directly (the same model, the same test frame) and turn a handful of its
weakness metrics into simple pass and fail gates.

"""

import os
import sys

import pandas as pd

from . import loader


def _import_goal3_xai():
    # Goal 3 sits next to Goal 2 and this agent, at
    here = os.path.dirname(os.path.abspath(__file__))     # .../agent
    version2 = os.path.dirname(os.path.dirname(here))     # .../version2
    goal3 = os.path.join(version2, "goal3")
    if goal3 not in sys.path:
        sys.path.insert(0, goal3)
    import xai
    return xai


def _test_frame():
    path = os.path.join(loader.artifacts_dir(), "test_predictions_logreg.csv")
    return pd.read_csv(path)


def certify(max_blind=1, min_worst_recall=0.5):
    # Run the weakness gates. Returns a dict with per-check values and 'passed'
    try:
        xai = _import_goal3_xai()
    except Exception as error:
        return {"available": False, "reason": str(error)}

    model = loader.load_model("tfidf+logreg")
    frame = _test_frame()

    # Gate 1: negation robustness: how many minimal pairs the model can't tell apart
    blind = xai.negation_blind_count(model)
    # Gate 2: weakest policy recall: the biggest blind spot must clear a floor
    per_policy = xai.per_policy_recall(frame)
    worst_policy = xai.worst_policy(frame)
    worst_recall = float(per_policy.loc[worst_policy, "recall"])

    checks = {
        "negation_blind_pairs": blind,
        "negation_ok": blind <= max_blind,
        "worst_policy": worst_policy,
        "worst_policy_recall": round(worst_recall, 3),
        "worst_policy_ok": worst_recall >= min_worst_recall,
    }
    checks["available"] = True
    checks["passed"] = bool(checks["negation_ok"]
                            and checks["worst_policy_ok"])
    return checks
