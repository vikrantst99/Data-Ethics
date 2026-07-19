"""
Train both models, evaluate them and save all files (artifacts) for Goal 3.
"""

import os
import joblib
import pandas as pd

from . import data
from . import models
from . import metrics

# Default folder where we store the trained models and the result files.
ARTIFACT_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "artifacts")
# Extra columns we keep for the fairness analysis.
META_COLS = ["ai_method", "issue_type",
             "grounded_flag", "n_red_flags", "red_flags"]


def build_artifacts(path=None, artifact_dir=None, random_state=42,
                    primary_model="tfidf+logreg"):
    # Train the models, test them and save everything to the artifacts folder.
    # 'primary_model' is the switch for the whole pipeline: "tfidf+logreg" (default)
    # or "tfidf+xgb". The chosen model becomes the primary; Naive Bayes stays as the
    # fast comparison model. The default keeps the original behaviour unchanged.

    if artifact_dir is None:
        artifact_dir = ARTIFACT_DIR
    artifact_dir = os.path.normpath(artifact_dir)
    os.makedirs(artifact_dir, exist_ok=True)

    # 1) Load the data and split it into train and test.
    df = data.load_proposals(path)
    idx_train, idx_test = data.make_split(df, random_state=random_state)
    X_train, y_train = data.get_xy(df, idx_train)
    X_test, y_test = data.get_xy(df, idx_test)

    # 2) Build the models to train: the chosen primary (LogReg or XGBoost) plus the
    #    Naive-Bayes comparison. make_model is the single switch point.
    factory = {}
    factory[primary_model] = models.make_model(primary_model)
    if primary_model != "tfidf+nb":
        factory["tfidf+nb"] = models.make_tfidf_nb()
    fitted = models.train_models(X_train, y_train, factory)

    # 3) Test every model. Collect one score row and one prediction table each.
    score_rows = []
    fair_frames = {}
    for name in fitted:
        model = fitted[name]
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= 0.45).astype(int)  # Manual fine-tuning

        # Build the score row: start with the model name, then add every score.
        row = {"model": name}
        scores = metrics.binary_scores(y_test, y_pred, proba)
        for key in scores:
            row[key] = scores[key]
        score_rows.append(row)

        # Build the prediction table with the extra columns for fairness.
        fair = df.loc[idx_test, META_COLS].copy()
        fair["y_true"] = y_test.to_numpy()
        fair["y_pred"] = y_pred
        fair["proba"] = proba
        fair_frames[name] = fair

        # Save the trained model to a file.
        file_name = "model_" + name.replace("+", "_") + ".joblib"
        models.save_model(model, os.path.join(artifact_dir, file_name))

    scores = pd.DataFrame(score_rows).set_index("model")

    # 4) Save the split and the PRIMARY model's prediction table for Goal 3.
    #    The file name follows the primary model (e.g. test_predictions_logreg.csv
    #    or test_predictions_xgb.csv), so downstream code can find the right one.
    split_info = {"idx_train": list(idx_train), "idx_test": list(idx_test)}
    joblib.dump(split_info, os.path.join(artifact_dir, "split.joblib"))
    primary_short = primary_model.replace("tfidf+", "")
    primary_csv = "test_predictions_" + primary_short + ".csv"
    fair_frames[primary_model].to_csv(
        os.path.join(artifact_dir, primary_csv), index=False)
    scores.to_csv(os.path.join(artifact_dir, "scores.csv"))

    result = {
        "df": df,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "models": fitted,
        "scores": scores,
        "fair_frames": fair_frames,
        "artifact_dir": artifact_dir,
        "primary_model": primary_model,
    }
    return result
