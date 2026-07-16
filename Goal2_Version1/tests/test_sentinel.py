"""Starter test suite for the Sentinel baseline package (Goal 2).
Goal 3 extends this into a weakness-detection suite with >=80% coverage; this
file already exercises every module so the coverage target is within reach.
"""
import os
import numpy as np
import pandas as pd
import pytest

import sentinel as S
from sentinel import data as D
from sentinel import baselines as B
from sentinel import models as M
from sentinel import metrics as MET


# fixtures
@pytest.fixture(scope="session")
def df():
    return S.load_proposals()


@pytest.fixture(scope="session")
def split(df):
    idx_tr, idx_te = S.make_split(df, random_state=42)
    return idx_tr, idx_te


@pytest.fixture(scope="session")
def trained(df, split):
    idx_tr, _ = split
    X, y = S.get_xy(df, idx_tr)
    return S.train_models(X, y)


# data
def test_load_has_target_and_balance(df):
    assert S.TARGET_COLUMN in df.columns
    assert set(df[S.TARGET_COLUMN].unique()) == {0, 1}
    # Goal-1 design: about one third are red flags
    assert 0.28 <= df[S.TARGET_COLUMN].mean() <= 0.38


def test_find_data_path_explicit_ok(df):
    p = S.find_data_path()
    assert os.path.exists(p)
    assert S.find_data_path(p) == p


def test_find_data_path_missing_raises():
    with pytest.raises(FileNotFoundError):
        S.find_data_path("/no/such/file_xyz.csv")


def test_make_split_is_stratified_and_disjoint(df, split):
    idx_tr, idx_te = split
    assert set(idx_tr).isdisjoint(set(idx_te))
    assert len(idx_tr) + len(idx_te) == len(df)
    r_tr = df.loc[idx_tr, S.TARGET_COLUMN].mean()
    r_te = df.loc[idx_te, S.TARGET_COLUMN].mean()
    assert abs(r_tr - r_te) < 0.05


def test_make_split_bad_test_size(df):
    with pytest.raises(ValueError):
        S.make_split(df, test_size=1.5)


def test_get_xy_shapes(df, split):
    idx_tr, _ = split
    X, y = S.get_xy(df, idx_tr)
    assert len(X) == len(y) == len(idx_tr)
    assert X.isna().sum() == 0


def test_load_missing_label_column(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"description": ["x"]}).to_csv(p, index=False)
    with pytest.raises(KeyError):
        D.load_proposals(str(p))


# baselines
def test_rule_predict_detects_terms():
    texts = ["We use social scoring on citizens.", "A perfectly normal bug fix."]
    pred = B.rule_predict(texts)
    assert pred.tolist() == [1, 0]


def test_rule_predict_is_case_insensitive():
    assert B.rule_predict(["FULLY AUTONOMOUS decisions"]).tolist() == [1]


def test_majority_predict():
    out = B.majority_predict(5)
    assert out.tolist() == [0, 0, 0, 0, 0]
    assert B.majority_predict(3, majority_class=1).tolist() == [1, 1, 1]


# models
def test_model_factory_keys():
    f = M.model_factory()
    assert set(f) == {"tfidf+logreg", "tfidf+nb"}


def test_train_and_predict(trained, df, split):
    _, idx_te = split
    X, y = S.get_xy(df, idx_te)
    for name, model in trained.items():
        pred = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        assert len(pred) == len(y)
        assert ((proba >= 0) & (proba <= 1)).all()


def test_logreg_beats_majority_on_recall(trained, df, split):
    _, idx_te = split
    X, y = S.get_xy(df, idx_te)
    pred = trained["tfidf+logreg"].predict(X)
    assert MET.binary_scores(y, pred)["recall"] > 0.5


def test_save_and_load_roundtrip(trained, df, split, tmp_path):
    _, idx_te = split
    X, _ = S.get_xy(df, idx_te)
    model = trained["tfidf+logreg"]
    p = str(tmp_path / "m.joblib")
    S.save_model(model, p)
    loaded = S.load_model(p)
    assert np.array_equal(loaded.predict(X), model.predict(X))


def test_top_terms_linear(trained):
    pos, neg = S.top_terms(trained["tfidf+logreg"], n=5)
    assert len(pos) == 5 and len(neg) == 5
    assert (pos.values > 0).all()


def test_top_terms_rejects_non_linear(trained):
    with pytest.raises(AttributeError):
        S.top_terms(trained["tfidf+nb"])


# metrics
def test_binary_scores_perfect():
    y = [0, 1, 1, 0]
    s = MET.binary_scores(y, y, proba=[0.1, 0.9, 0.8, 0.2])
    assert s["accuracy"] == 1.0 and s["recall"] == 1.0 and s["roc_auc"] == 1.0


def test_confusion_counts_and_miss_rate():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]           # one missed violation
    tn, fp, fn, tp = MET.confusion_counts(y_true, y_pred)
    assert (tn, fp, fn, tp) == (2, 0, 1, 1)
    assert MET.miss_rate(y_true, y_pred) == 0.5


def test_threshold_sweep_monotone_missrate():
    proba = np.array([0.1, 0.4, 0.6, 0.9])
    y = np.array([0, 0, 1, 1])
    sweep = MET.threshold_sweep(y, proba, thresholds=[0.2, 0.5, 0.8])
    assert list(sweep.columns) == ["threshold", "miss_rate", "fpr"]
    # higher threshold -> miss rate cannot decrease
    assert sweep["miss_rate"].is_monotonic_increasing


def test_group_metrics_and_gaps():
    frame = pd.DataFrame({
        "grp": ["a", "a", "b", "b"],
        "y_true": [1, 0, 1, 0],
        "y_pred": [1, 0, 0, 0],
    })
    gm = MET.group_metrics(frame, "grp")
    assert gm.loc["a", "recall_TPR"] == 1.0
    assert gm.loc["b", "recall_TPR"] == 0.0
    gaps = MET.eo_gaps(frame, "grp")
    assert gaps["TPR gap"] == 1.0


def test_group_metrics_missing_column():
    frame = pd.DataFrame({"grp": ["a"], "y_true": [1]})
    with pytest.raises(KeyError):
        MET.group_metrics(frame, "grp")


# pipeline
def test_build_artifacts_writes_files(tmp_path):
    out = S.build_artifacts(artifact_dir=str(tmp_path))
    assert set(out["scores"].index) == {"tfidf+logreg", "tfidf+nb"}
    for f in ["model_tfidf_logreg.joblib", "model_tfidf_nb.joblib",
              "split.joblib", "scores.csv", "test_predictions_logreg.csv"]:
        assert (tmp_path / f).exists()
    # both real models clear a sane recall floor
    assert (out["scores"]["recall"] > 0.5).all()
