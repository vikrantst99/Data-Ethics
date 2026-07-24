"""
Find the sibling Goal-2 snapshot and load its trained model and data.

The agent does not train anything. It reuses the model that Goal 2 already
built and saved to 'goal2/artifacts'. This is the Version-2 (marker-free)
model.
"""

import os
import sys
import joblib


def goal2_dir():
    # The Goal-2 folder is a sibling of this goal-4 folder, one level up.
    here = os.path.dirname(os.path.abspath(__file__))      
    goal4 = os.path.dirname(here)                          
    candidate = os.path.join(os.path.dirname(goal4), "goal2")   
    if not os.path.isdir(candidate):
        raise FileNotFoundError("Goal-2 snapshot not found at " + candidate)
    return candidate


def artifacts_dir():
    # The 'artifacts' folder Goal 2 writes its models and scores to
    return os.path.join(goal2_dir(), "artifacts")


def load_sentinel():
    # Import and return Goal 2's 'sentinel' package (for load_proposals etc.)
    g2 = goal2_dir()
    if g2 not in sys.path:
        sys.path.insert(0, g2)
    import sentinel
    return sentinel


def load_model(name="tfidf+logreg"):
    # Load one saved model pipeline. Default is the explainable main model
    files = {
        "tfidf+logreg": "model_tfidf_logreg.joblib",
        "tfidf+nb": "model_tfidf_nb.joblib",
    }
    if name not in files:
        raise ValueError("Unknown model name: " + str(name))
    path = os.path.join(artifacts_dir(), files[name])
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Missing " + path + ". Run Goal 2 (run_goal2.py) first.")
    return joblib.load(path)


def load_data():
    # Load the 1000 proposals (with the 0/1 target column) via Goal 2's loader
    sentinel = load_sentinel()
    return sentinel.load_proposals()
