"""
Explainable AI (XAI) for the Sentinel baseline

Two views:
--- Global: which terms matter overall.
--- Local: why THIS proposal was flagged.
"""

import re
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score


def global_importance_linear(pipeline, n=20):
    # Top-n terms by the size of their Logistic-Regression weight (signed).
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        raise AttributeError("Model is not linear (no coef_).")
    coefs = pd.Series(classifier.coef_[0],
                      index=vectorizer.get_feature_names_out())
    # Sort by absolute weight (strongest terms first), then keep the top n.
    order = coefs.abs().sort_values(ascending=False).index
    return coefs.reindex(order).head(n)


def top_terms_by_class(pipeline, n=10):
    # The n strongest Red-Flag terms (positive) and n strongest Compliant terms (negative).
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        raise AttributeError("Model is not linear (no coef_).")
    coefs = pd.Series(classifier.coef_[0],
                      index=vectorizer.get_feature_names_out())
    # kind="stable" so tied weights keep the same order on every machine.
    ordered = coefs.sort_values(ascending=False, kind="stable")
    red_flag = ordered.head(n)
    compliant = ordered.tail(n).sort_values(kind="stable")
    return red_flag, compliant


def ablation_importance(pipeline, texts, y_true, terms):
    # Model-independent importance: how much recall drops when a term is removed
    texts = list(texts)
    base_recall = recall_score(y_true, pipeline.predict(texts))

    drops = {}
    for term in terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        # Copy of the texts with this term replaced by a space
        ablated = []
        for x in texts:
            ablated.append(pattern.sub(" ", str(x)))
        new_recall = recall_score(y_true, pipeline.predict(ablated))
        drops[term] = base_recall - new_recall

    return pd.Series(drops).sort_values(ascending=False)


def explain_instance_linear(pipeline, text, n=10):
    # Local explanation for a linear model: contribution = tfidf_value * coefficient
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        raise AttributeError(
            "Local linear explanation needs a model with coef_.")

    x = vectorizer.transform([text])                 # one row of tfidf values
    coef = classifier.coef_[0]
    contributions = x.multiply(
        coef).toarray().ravel()   # tfidf * coef per word
    feature_names = vectorizer.get_feature_names_out()

    # Keep only the words that actually appear in this text (value not zero)
    used = np.nonzero(contributions)[0]
    df = pd.DataFrame({
        "term": feature_names[used],
        "contribution": contributions[used],
    })
    df["abs"] = df["contribution"].abs()
    df = df.sort_values("abs", ascending=False)
    df = df.drop(columns="abs").head(n).reset_index(drop=True)
    return df


def predict_proba_text(pipeline, text):
    # Red-Flag probability for a single proposal text
    proba = pipeline.predict_proba([text])[0, 1]
    return float(proba)
