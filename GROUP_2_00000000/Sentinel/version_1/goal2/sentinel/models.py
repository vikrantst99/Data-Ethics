"""
The machine-learning baseline models + helpers to save and load them.

All are text classifiers with predict_proba (this needed for thresholds and ROC/PR)
and readable weights (this is needed for the XAI part in Goal 3):

- "tfidf+logreg" - TF-IDF + Logistic Regression with class_weight="balanced"
  against the 1/3 class imbalance. This is our main model.
- "tfidf+nb"     - TF-IDF + Naive Bayes, a classic fast text model we compare to.
- "tfidf+xgb"    - TF-IDF + XGBoost, a stronger tree-based comparison model. It is
  optional (needs the extra 'xgboost' package) and is NOT part of the default
  factory, so it never breaks the pipeline if xgboost is missing. Use make_model
  or make_tfidf_xgb to switch to it.
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def make_tfidf():
    # Create the TF-IDF step that turns text into numbers (used by both models)
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        stop_words="english",
        sublinear_tf=True,
    )


def make_tfidf_logreg(random_state=RANDOM_STATE):
    # Model 1: TF-IDF + Logistic Regression (balanced). This is our main model
    model = Pipeline([
        ("tfidf", make_tfidf()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])
    return model


def make_tfidf_nb():
    # Model 2: TF-IDF + Naive Bayes. Used to compare against model 1
    model = Pipeline([
        ("tfidf", make_tfidf()),
        ("clf", MultinomialNB()),
    ])
    return model


def make_tfidf_xgb(scale_pos_weight=2.0):
    # Model 3 (optional): TF-IDF + XGBoost (gradient-boosted trees). A stronger
    # comparison model. scale_pos_weight handles the 1/3 imbalance the same way
    # class_weight="balanced" does for LogReg; the default 2.0 is the 667:333 ratio.
    # We import xgboost inside the function so the whole package still imports even
    # when xgboost is not installed - only this one model needs it.
    import xgboost as xgb
    model = Pipeline([
        ("tfidf", make_tfidf()),
        ("clf", xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
        )),
    ])
    return model


def make_model(name):
    # Switch between the models by name and return a fresh, untrained pipeline.
    # This is the single place a notebook or script picks LogReg vs XGBoost.
    if name == "tfidf+logreg":
        return make_tfidf_logreg()
    if name == "tfidf+nb":
        return make_tfidf_nb()
    if name == "tfidf+xgb":
        return make_tfidf_xgb()
    raise ValueError("Unknown model name: " + str(name))


def model_factory():
    # Return the two default models (not trained yet) in a dictionary.
    # XGBoost is optional and left out here on purpose (see make_model).
    models = {}
    models["tfidf+logreg"] = make_tfidf_logreg()
    models["tfidf+nb"] = make_tfidf_nb()
    return models


def train_models(X_train, y_train, factory=None):
    # Train every model and return a dictionary: name -> trained model
    if factory is None:
        factory = model_factory()
    fitted = {}
    for name in factory:
        model = factory[name]
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def save_model(model, path):
    # Save a trained model to a file with joblib. Returns the path
    joblib.dump(model, path)
    return path


def load_model(path):
    # Load a model that was saved with save_model
    return joblib.load(path)


def top_terms(pipeline, n=15):
    # Return the strongest terms of a linear model
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        raise AttributeError("Model has no coef_ (it is not a linear model).")
    weights = classifier.coef_[0]
    feature_names = vectorizer.get_feature_names_out()
    coefs = pd.Series(weights, index=feature_names)
    return coefs.nlargest(n), coefs.nsmallest(n)
