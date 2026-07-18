"""Start file for Goal 2 - run THIS file (not the files inside 'sentinel/').

It trains both models, evaluates them and saves all artifacts that Goal 3 and
the agent need.
"""

import sentinel

# Switch the whole pipeline between LogReg and XGBoost here.
# "tfidf+logreg" = our explainable main model, "tfidf+xgb" = the stronger tree model.
PRIMARY_MODEL = "tfidf+logreg"

# Train the chosen model (+ the NB comparison) and save everything to 'artifacts/'.
result = sentinel.build_artifacts(primary_model=PRIMARY_MODEL)

print("Goal 2 finished. Primary model:", result["primary_model"])
print("Artifacts were saved to:", result["artifact_dir"])
print()
print("Scores on the hold-out test set:")
print(result["scores"][["recall", "precision", "f1", "roc_auc"]].round(3))
