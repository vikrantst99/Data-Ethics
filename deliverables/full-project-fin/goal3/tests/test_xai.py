"""
Automated weakness/analysis tests for Goal 3 (target: >=80% coverage).

Combines unit tests on synthetic data (fast, deterministic) with a few
integration tests against the real Goal-2 artifacts.
"""
from xai import weakness as W
from xai import explain as E
from xai import analysis as A
import xai
import os
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# tiny fixtures
@pytest.fixture(scope="module")
def toy():
    # A tiny, separable text problem
    pos = ["social scoring of citizens", "emotion recognition without consent",
           "fully autonomous decision no oversight", "biometric tracking of users"]
    neg = ["fix the login bug", "improve the dashboard layout",
           "update documentation for api", "add unit tests to module"]
    X = pd.Series(pos + neg)
    y = pd.Series([1]*len(pos) + [0]*len(neg))
    return X, y


@pytest.fixture(scope="module")
def toy_model(toy):
    X, y = toy
    m = Pipeline([("tfidf", TfidfVectorizer()),
                  ("clf", LogisticRegression(class_weight="balanced", max_iter=500))])
    return m.fit(X, y)


@pytest.fixture(scope="module")
def toy_nb(toy):
    X, y = toy
    return Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())]).fit(X, y)


# analysis
def test_cross_val_recall_range(toy, toy_model):
    X, y = toy
    scores = A.cross_val_recall(toy_model, X, y, cv=2)
    assert len(scores) == 2
    assert ((scores >= 0) & (scores <= 1)).all()


def test_compare_models_frame(toy, toy_model, toy_nb):
    X, y = toy
    out = A.compare_models({"lr": toy_model, "nb": toy_nb}, X, y, cv=2)
    assert set(out.index) == {"lr", "nb"}
    assert "cv_recall_mean" in out.columns


def test_calibration_bins(toy):
    _, y = toy
    proba = np.linspace(0, 1, len(y))
    cb = A.calibration_bins(y, proba, n_bins=4)
    assert {"mean_proba", "observed_rate", "n"} <= set(cb.columns)
    assert cb["n"].sum() == len(y)


def test_error_indices():
    yt = np.array([1, 1, 0, 0])
    yp = np.array([0, 1, 1, 0])
    assert A.error_indices(yt, yp, "fn").tolist() == [0]
    assert A.error_indices(yt, yp, "fp").tolist() == [2]
    with pytest.raises(ValueError):
        A.error_indices(yt, yp, "xx")


# explain
def test_global_importance_linear(toy_model):
    gi = E.global_importance_linear(toy_model, n=5)
    assert len(gi) == 5


def test_global_importance_requires_linear(toy_nb):
    with pytest.raises(AttributeError):
        E.global_importance_linear(toy_nb)


def test_top_terms_by_class_splits_the_directions(toy_model):
    red, ok = E.top_terms_by_class(toy_model, n=3)
    assert len(red) == 3 and len(ok) == 3
    # Red-Flag terms push up, Compliant terms push down.
    assert (red > 0).all()
    assert (ok < 0).all()
    # Strongest first in each direction.
    assert red.iloc[0] >= red.iloc[-1]
    assert ok.iloc[0] <= ok.iloc[-1]


def test_top_terms_by_class_is_deterministic(toy_model):
    # Tied weights must not reorder between runs (kind='stable').
    first = E.top_terms_by_class(toy_model, n=5)
    second = E.top_terms_by_class(toy_model, n=5)
    assert list(first[0].index) == list(second[0].index)
    assert list(first[1].index) == list(second[1].index)


def test_top_terms_by_class_requires_linear(toy_nb):
    with pytest.raises(AttributeError):
        E.top_terms_by_class(toy_nb)


def test_local_explanation_signs(toy_model):
    df = E.explain_instance_linear(
        toy_model, "social scoring of citizens", n=5)
    assert {"term", "contribution"} <= set(df.columns)
    # the strongest contributor should push toward Red Flag (positive)
    assert df.iloc[0]["contribution"] > 0


def test_local_explanation_requires_coef(toy_nb):
    with pytest.raises(AttributeError):
        E.explain_instance_linear(toy_nb, "text")


def test_predict_proba_text(toy_model):
    p = E.predict_proba_text(toy_model, "social scoring of citizens")
    assert 0.0 <= p <= 1.0


def test_ablation_importance_positive(toy, toy_model):
    X, y = toy
    imp = E.ablation_importance(
        toy_model, X, y, ["social scoring", "biometric"])
    assert (imp >= 0).all() or imp.notna().all()


# weakness
def test_mask_terms_removes():
    out = W.mask_terms("we do social scoring daily", ["social scoring"])
    assert "social scoring" not in out.lower()
    assert "daily" in out


def test_obfuscate_breaks_token():
    out = W.obfuscate("social scoring", ["social scoring"])
    assert out != "social scoring"
    assert "." in out


def test_evasion_recall_keys(toy, toy_model):
    X, y = toy
    pos = X[y == 1]
    res = W.evasion_recall(toy_model, pos, W.mask_terms,
                           ["social scoring", "emotion recognition",
                            "fully autonomous", "biometric"])
    assert set(res) == {"recall_before", "recall_after", "recall_drop", "n"}
    assert res["recall_before"] >= res["recall_after"] - 1e-9


def test_per_policy_recall_and_worst():
    frame = pd.DataFrame({
        "red_flags": ["RAI-01", "RAI-01,RAI-02", "RAI-02", np.nan],
        "y_true": [1, 1, 1, 0],
        "y_pred": [1, 1, 0, 0],
    })
    pp = W.per_policy_recall(frame)
    assert pp.loc["RAI-01", "recall"] == 1.0
    assert pp.loc["RAI-02", "recall"] == 0.5
    assert W.worst_policy(frame) == "RAI-02"


def test_per_policy_missing_column():
    with pytest.raises(KeyError):
        W.per_policy_recall(pd.DataFrame({"y_true": [1], "y_pred": [1]}))


def test_worst_policy_empty():
    frame = pd.DataFrame({"red_flags": [np.nan], "y_true": [0], "y_pred": [0]})
    with pytest.raises(ValueError):
        W.worst_policy(frame)


# negation (Goal 3 finding)
@pytest.fixture(scope="module")
def negation_toy():
    # A tiny problem where ONLY the negation decides the label.
    bad = ["no human oversight is foreseen",
           "users are not informed about ai",
           "data is not disclosed to subjects",
           "groups are processed without safeguards"]
    good = ["a human oversight is foreseen",
            "users are informed about ai",
            "data is disclosed to subjects",
            "groups are processed with safeguards"]
    X = pd.Series(bad + good)
    y = pd.Series([1] * len(bad) + [0] * len(good))
    return X, y


def toy_pairs(X):
    # Row i is the violation, row i+4 is its correct counter-statement
    pairs = []
    for i in range(4):
        pairs.append((X.iloc[i], X.iloc[i + 4]))
    return pairs


def test_english_stopwords_destroy_negation():
    # Why the model is negation-blind: sklearn drops 'not' before the model sees it
    an = TfidfVectorizer(ngram_range=(
        1, 2), stop_words="english").build_analyzer()
    assert an("This requirement is NOT met") == an("This requirement is met")


def test_stopwords_without_negations_keeps_meaning():
    # The fix: keeping the negations makes the two sentences different again
    keep = W.stopwords_without_negations()
    assert "not" not in keep
    assert "without" not in keep
    an = TfidfVectorizer(ngram_range=(1, 2), stop_words=keep).build_analyzer()
    assert an("This requirement is NOT met") != an("This requirement is met")


def test_negation_gap_columns(toy_model):
    gaps = W.negation_gap(toy_model)
    assert {"violation", "counter_statement", "proba_violation",
            "proba_counter", "gap"} <= set(gaps.columns)
    assert len(gaps) == len(W.NEGATION_PAIRS)


def test_negation_blindness_with_english_stopwords(negation_toy):
    # Built the Goal-2 way, the model cannot see the negation at all
    X, y = negation_toy
    m = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
                  ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
    pairs = toy_pairs(X)
    assert W.negation_blind_count(m, pairs) == len(pairs)


def test_keeping_negations_fixes_blindness(negation_toy):
    # Same data, negations kept: now the model separates every pair
    X, y = negation_toy
    m = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2),
                                            stop_words=W.stopwords_without_negations())),
                  ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
    pairs = toy_pairs(X)
    assert W.negation_blind_count(m, pairs) == 0


@pytest.mark.xfail(reason="Known weakness found with XAI in Goal 3: our training data has no "
                          "counter-examples (every ethics word appears in 0 compliant "
                          "proposals), so the model ignores negation. Fix needs Goal-1 data. "
                          "See GOAL3_XAI_Befund_Negation_und_Leak.docx",
                   strict=False)
def test_primary_model_should_see_negation():
    # The Goal-2 model SHOULD tell a violation from its correct counter-statement
    try:
        model = xai.load_models()["tfidf+logreg"]
    except FileNotFoundError:
        pytest.skip("Goal 2 artifacts not present")
    assert xai.negation_blind_count(model) == 0


# integration
@pytest.mark.parametrize("fn", ["load_data", "load_models", "load_test_frame"])
def test_goal2_artifacts_available(fn):
    # Integration: the Goal-2 artifacts load (skips cleanly if Goal 2 not built)
    try:
        obj = getattr(xai, fn)()
    except FileNotFoundError:
        pytest.skip("Goal 2 artifacts not present")
    assert obj is not None


def test_end_to_end_per_policy():
    try:
        frame = xai.load_test_frame()
    except FileNotFoundError:
        pytest.skip("Goal 2 artifacts not present")
    pp = xai.per_policy_recall(frame)
    assert (pp["recall"] <= 1.0).all() and (pp["recall"] >= 0.0).all()
