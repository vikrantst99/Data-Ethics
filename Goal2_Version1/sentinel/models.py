"""
The two machine-learning baseline models + helpers to save and load them.

Both are text classifiers with predict_proba (this needed for thresholds and ROC/PR)
and readable weights (this is needed for the XAI part in Goal 3):

- "tfidf+logreg" - TF-IDF + Logistic Regression with class_weight="balanced"
  against the 1/3 class imbalance. This is our main model.
- "tfidf+nb"     - TF-IDF + Naive Bayes, a classic fast text model we compare to.
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


def model_factory():
    # Return the two models (not trained yet) in a dictionary
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
