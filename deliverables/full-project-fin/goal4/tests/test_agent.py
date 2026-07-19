"""
Offline tests for the Goal-4 agent.

Cover the core flow plus the Section-3 add-ons: configurable severity, borderline
routing, open-ended verification, writeback, batch and the weakness pre-gate.
All run without a Groq key and without a network download.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent import (Classifier, PolicyRetriever, get_severity, worst_severity,
                   policy_docs, load_severity_overrides, build_graph, run_once,
                   run_batch, certify, writeback, extract_text)
from agent import loader
from agent.graph import select_matched


RED_FLAG = ("Our chatbot will handle customer complaints automatically "
            "without telling users it is an AI.")
COMPLIANT = ("The system shall support a send and receive operation and "
             "display action items in one location.")


@pytest.fixture(scope="session")
def classifier():
    return Classifier(loader.load_model("tfidf+logreg"))


@pytest.fixture(scope="session")
def retriever():
    return PolicyRetriever()


@pytest.fixture(scope="session")
def graph(classifier, retriever):
    return build_graph(classifier=classifier, retriever=retriever)


# core (same guarantees as basic)
def test_severity_and_docs():
    assert get_severity("RAI-02") == "High"
    assert get_severity("RAI-07") == "Medium"
    assert len(policy_docs()) == 9


def test_graph_flags_red_flag(graph):
    result = run_once(RED_FLAG, graph=graph)
    assert result["y_pred"] == 1
    assert result["policy_id"].startswith("RAI-")
    assert "is_violation" in result
    # Multi-policy: the verdict lists the matched policies and a governing severity.
    assert len(result["matched_policies"]) >= 1
    assert result["severity"] in ("High", "Medium")
    # When the gate is High, the primary (reason) policy must itself be High.
    if result["severity"] == "High":
        assert get_severity(result["policy_id"]) == "High"
    # Transparency: the verdict must carry the provenance note.
    assert "trained TF-IDF model" in result["verdict"]["provenance"]


def test_findings_one_per_policy(graph):
    result = run_once(RED_FLAG, graph=graph)
    findings = result["findings"]
    assert len(findings) == len(result["matched_policies"])
    assert [f["policy_id"] for f in findings] == \
        [m["policy_id"] for m in result["matched_policies"]]
    for f in findings:
        assert f["reason"] and f["recommended_fix"]
    assert findings[0]["policy_id"] == result["policy_id"]


# Multi-policy governing severity (the Aurora fix)
def test_worst_severity_and_matching():
    assert worst_severity(["RAI-08", "RAI-04"]) == "High"
    assert worst_severity(["RAI-08", "RAI-06"]) == "Medium"
    retrieved = [{"policy_id": "RAI-07", "score": 0.45},
                 {"policy_id": "RAI-03", "score": 0.44},
                 {"policy_id": "RAI-06", "score": 0.10}]
    matched = select_matched(retrieved, margin=0.06)
    ids = [m["policy_id"] for m in matched]
    assert ids == ["RAI-07", "RAI-03"]              # RAI-06 too far below
    assert worst_severity(ids) == "High"           # a close High governs the gate


def test_retriever_no_duplicate_policies():
    # Rebuilding the retriever must not accumulate duplicate policy docs
    query = "automated decision without consent public applicants"
    for _ in range(3):
        ids = [h["policy_id"] for h in PolicyRetriever().search(query, k=9)]
        assert len(ids) == len(set(ids))


# Section 3 - configurable severity from CSV
def test_severity_overrides_loaded():
    overrides = load_severity_overrides()
    # The shipped config/policies.csv defines all nine policies.
    assert len(overrides) == 9
    assert overrides["RAI-01"] == "High"


# Section 3 - open-ended verification can clear a false alarm
def test_verification_can_clear(classifier, retriever):
    # verify=True lets the offline heuristic mark weak signals as non-violations.
    g = build_graph(classifier=classifier, retriever=retriever, verify=True)
    result = run_once(COMPLIANT, graph=g)
    assert "is_violation" in result
    assert result["final_decision"]


# Section 3 - borderline LLM routing does not break the flow
def test_borderline_band_runs(classifier, retriever):
    g = build_graph(classifier=classifier, retriever=retriever,
                    llm_band=(0.35, 0.65))
    result = run_once(RED_FLAG, graph=g)
    assert result["final_decision"]


# Section 3 - writeback produces files
def test_writeback_writes_files(classifier, retriever, tmp_path):
    g = build_graph(classifier=classifier, retriever=retriever,
                    writeback=True, outbox=str(tmp_path))
    result = run_once(RED_FLAG, graph=g, proposal_id="test_prop")
    files = result["written_files"]
    assert any("confluence_comment" in f for f in files)
    assert any("agent_log.jsonl" in f for f in files)
    assert os.path.exists(os.path.join(str(tmp_path), "agent_log.jsonl"))


def test_writeback_module_direct(tmp_path):
    fake = {"y_pred": 1, "proba": 0.8, "policy_id": "RAI-02", "severity": "High",
            "final_decision": "x", "verdict": {"reason": "r", "recommended_fix": "f",
            "is_violation": True}, "audit": {"predicted": 1}}
    files = writeback.write_all("PROP-9", fake, outbox=str(tmp_path))
    assert len(files) == 3   # confluence + jira (High violation) + audit log


# Section 3 - batch over a folder
def test_batch_over_inbox(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "a.txt").write_text(RED_FLAG, encoding="utf-8")
    (inbox / "b.txt").write_text(COMPLIANT, encoding="utf-8")
    summary = run_batch(inbox=str(inbox), writeback=True,
                        outbox=str(tmp_path / "out"))
    assert len(summary) == 2
    assert {r["proposal_id"] for r in summary} == {"a", "b"}


# Section 3 - PDF upload (extract text, then feed the agent)
def test_pdf_extract_roundtrip(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    proposal = ("The loan decision is made fully automatically without human "
                "intervention and no human-in-the-loop is foreseen.")
    pdf_path = tmp_path / "proposal.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        fig = plt.figure(figsize=(8, 4))
        fig.text(0.05, 0.9, proposal, wrap=True, fontsize=11, va="top")
        pdf.savefig(fig)
        plt.close(fig)

    from_path = extract_text(str(pdf_path))
    from_bytes = extract_text(pdf_path.read_bytes())
    assert "automatically" in from_path
    assert from_path.strip() == from_bytes.strip()


# Streamlit analysis charts (A1-A3)
def test_charts_build(graph):
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import charts
    result = run_once(RED_FLAG, graph=graph)
    severities = {h["policy_id"]: get_severity(h["policy_id"])
                  for h in result["retrieved"]}
    matched_ids = [m["policy_id"] for m in result["matched_policies"]]
    # Each builder must return a matplotlib Figure without raising.
    f1 = charts.fig_probability(result["proba"])
    f2 = charts.fig_trigger_words(result["trigger_words"])
    f3 = charts.fig_policy_scores(result["retrieved"], matched_ids, severities)
    for fig in (f1, f2, f3):
        assert fig.get_axes()
        charts.plt.close(fig)


# Section 3 - weakness pre-gate
def test_certify_runs():
    result = certify()
    # Goal-3 xai should be importable in the version2 tree.
    assert "available" in result
    if result["available"]:
        assert "passed" in result
        assert result["worst_policy"].startswith("RAI-")
