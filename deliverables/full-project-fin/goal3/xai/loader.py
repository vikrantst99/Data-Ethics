"""
Find the bundled Goal-2 snapshot and load its data and trained model files.

Goal 3 does not train anything new. It looks at the models that Goal 2 already built.
"""

import os
import sys
import joblib
import pandas as pd


def goal2_dir():
    # Path to the Goal-2 snapshot: the sibling folder 'goal2' one level up from Goal 3.
    here = os.path.dirname(os.path.abspath(__file__))     # .../goal3/xai
    goal3 = os.path.dirname(here)                         # .../goal3
    candidate = os.path.join(os.path.dirname(goal3), "goal2")   # .../Version_2/goal2
    if not os.path.isdir(candidate):
        raise FileNotFoundError("Goal-2 snapshot not found at " + candidate)
    return candidate


def artifacts_dir():
    # Path to the snapshot's "artifacts" folder
    return os.path.join(goal2_dir(), "artifacts")


def load_sentinel():
    # Import and return Goal 2's 'sentinel' package
    g2 = goal2_dir()
    if g2 not in sys.path:
        sys.path.insert(0, g2)
    import sentinel
    return sentinel


def load_data():
    # Load the 1000 proposals (with the target column) using Goal 2's loader
    sentinel = load_sentinel()
    return sentinel.load_proposals()


def load_models():
    # Load the saved models. Returns {'tfidf+logreg': ..., 'tfidf+nb': ...}
    folder = artifacts_dir()
    model_files = {
        "tfidf+logreg": "model_tfidf_logreg.joblib",
        "tfidf+nb": "model_tfidf_nb.joblib",
    }
    models = {}
    for name in model_files:
        path = os.path.join(folder, model_files[name])
        if os.path.exists(path):
            models[name] = joblib.load(path)
    if len(models) == 0:
        raise FileNotFoundError(
            "No model files in " + folder + ". Run Goal 2 build_artifacts() first.")
    return models


def load_test_frame():
    # Load the main model's hold-out predictions that Goal 2 saved
    path = os.path.join(artifacts_dir(), "test_predictions_logreg.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Missing " + path + ". Run Goal 2 first.")
    return pd.read_csv(path)
