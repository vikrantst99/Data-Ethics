"""Start file for Goal 2 - run THIS file (not the files inside 'sentinel/').

It trains both models, evaluates them and saves all artifacts that Goal 3 and
the agent need.
"""

import sentinel

# Train both models, evaluate them and save everything to 'artifacts/'.
result = sentinel.build_artifacts()

print("Goal 2 finished. Artifacts were saved to:", result["artifact_dir"])
print()
print("Scores on the hold-out test set:")
print(result["scores"][["recall", "precision", "f1", "roc_auc"]].round(3))
