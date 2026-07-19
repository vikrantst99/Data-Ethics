"""
The LangGraph

On top of the basic flow it adds three of the Section-3 options:

- borderline LLM routing : the LLM is only used when the probability sits in an
  uncertain band (default 0.35-0.65). Confident cases use the offline template,
  which saves LLM calls (the "Spar-Trick" from the concept).
- open-ended verification : after gathering evidence the agent may conclude the
  proposal is a false alarm (is_violation = False) instead of always assuming a
  violation.
- writeback : an optional final node writes a Confluence comment, a Jira ticket
  and an audit-log line to the outbox.

classify -> assess -> verify? -> human_gate -> writeback -> END
not Red Flag        false alarm
    v                    v
compliant             cleared
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from .classify import Classifier, DEFAULT_THRESHOLD
from .rag import PolicyRetriever
from .policies import get_severity, worst_severity, ALL_POLICY_IDS
from .llm import compose_verdict, per_policy_finding
from . import loader
from . import writeback as wb

DEFAULT_LLM_BAND = (0.35, 0.65)


def select_matched(retrieved, margin=0.06, cap=5):
    # Keep every retrieved policy whose score is within 'margin' of the best match, not just the top hit
    if not retrieved:
        return []
    top = retrieved[0]["score"]
    matched = []
    for hit in retrieved:
        if hit["score"] >= top - margin:
            matched.append(hit)
    return matched[:cap]


def _policy_label(primary, matched_ids):
    # "RAI-03 (+RAI-07, RAI-04)" when several policies are involved
    others = [pid for pid in matched_ids if pid != primary]
    if others:
        return primary + " (+" + ", ".join(others) + ")"
    return primary


def _build_findings(matched_policies, trigger_words, verdict):
    # One reason and fix per matched policy: the primary reuses the main verdict, the extra ones get an offline finding
    findings = []
    for m in matched_policies:
        if m["policy_id"] == verdict["policy_id"]:
            findings.append({
                "policy_id": verdict["policy_id"],
                "severity": m["severity"],
                "reason": verdict["reason"],
                "recommended_fix": verdict["recommended_fix"],
                "source": verdict["source"],
            })
        else:
            findings.append(per_policy_finding(
                m["policy_id"], m["severity"], trigger_words))
    return findings


class AgentState(TypedDict, total=False):
    proposal_id: str
    proposal_text: str
    threshold: float
    proba: float
    y_pred: int
    trigger_words: list
    retrieved: list
    matched_policies: list
    findings: list
    policy_id: str
    severity: str
    verdict: dict
    is_violation: bool
    human_required: bool
    human_approved: bool
    final_decision: str
    audit: dict
    written_files: list


def _auto_approve(state):
    return True


def build_graph(classifier=None, retriever=None, approver=None,
                threshold=DEFAULT_THRESHOLD, llm_band=DEFAULT_LLM_BAND,
                verify=True, writeback=False, outbox=None,
                policy_margin=0.06, max_policies=5):
    if classifier is None:
        classifier = Classifier(loader.load_model("tfidf+logreg"), threshold)
    if retriever is None:
        retriever = PolicyRetriever()
    if approver is None:
        approver = _auto_approve

    def classify_node(state):
        proba = classifier.proba(state["proposal_text"])
        thr = state.get("threshold", threshold)
        return {"proba": round(proba, 4), "y_pred": int(proba >= thr)}

    def assess_node(state):
        text = state["proposal_text"]
        triggers = classifier.trigger_words(text, n=10)
        # Ask for all nine policies so we see every one the document matches
        query = " ".join(word for word, _ in triggers[:8]) or text
        retrieved = retriever.search(query, k=len(ALL_POLICY_IDS))
        # Which policies really apply? Everything close to the best match
        matched = select_matched(retrieved, policy_margin, max_policies)
        # Put High-severity policies first so the verdict is phrased around the most severe one
        high = []
        medium = []
        for hit in matched:
            if get_severity(hit["policy_id"]) == "High":
                high.append(hit)
            else:
                medium.append(hit)
        matched = high + medium
        matched_policies = []
        for hit in matched:
            matched_policies.append({"policy_id": hit["policy_id"],
                                     "severity": get_severity(hit["policy_id"]),
                                     "score": hit["score"]})
        matched_ids = [m["policy_id"] for m in matched_policies]
        # Governing severity is the worst among the matches, so a Medium top hit cannot hide a High violation
        governing = worst_severity(matched_ids) if matched_ids else "Unknown"
        # Borderline routing: only spend an LLM call inside the uncertain band; confident cases use the offline template
        proba = state["proba"]
        in_band = llm_band is None or (llm_band[0] <= proba <= llm_band[1])
        verdict = compose_verdict(text, triggers, matched, proba, verify=verify,
                                  use_llm=in_band)
        if not in_band:
            # Outside the band we trust the model: keep the template wording and treat it as a violation
            verdict = dict(verdict)
            verdict["is_violation"] = True
        # One finding per matched policy (primary keeps main verdict, rest offline)
        findings = _build_findings(matched_policies, triggers, verdict)
        return {
            "trigger_words": triggers,
            "retrieved": retrieved,
            "matched_policies": matched_policies,
            "findings": findings,
            "policy_id": verdict["policy_id"],
            "severity": governing,
            "verdict": verdict,
            "is_violation": verdict.get("is_violation", True),
            "human_required": governing == "High",
        }

    def human_gate_node(state):
        matched_ids = [m["policy_id"] for m in state.get("matched_policies", [])]
        label = _policy_label(state["policy_id"], matched_ids)
        audit = _base_audit(state)
        if state["severity"] == "High":
            approved = bool(approver(state))
            decision = ("High severity (" + label + "): human "
                        "reviewer " + ("approved remediation." if approved
                                       else "rejected. No changes made."))
        else:
            approved = True
            decision = ("Medium severity (" + label +
                        "): automatic remediation approved.")
        audit["human_approved"] = approved
        return {"human_approved": approved, "final_decision": decision,
                "audit": audit}

    def cleared_node(state):
        audit = _base_audit(state)
        audit["human_approved"] = True
        return {"human_approved": True, "human_required": False,
                "final_decision": ("Reviewed as a false alarm - no RAI "
                                   "violation, no action required."),
                "audit": audit}

    def compliant_node(state):
        return {"is_violation": False, "human_required": False,
                "human_approved": True,
                "final_decision": "Compliant - no action required.",
                "audit": {"predicted": 0, "probability": state["proba"]}}

    def writeback_node(state):
        files = wb.write_all(state.get("proposal_id", "PROP"), state, outbox)
        return {"written_files": files}

    def _base_audit(state):
        return {
            "predicted": state["y_pred"],
            "probability": state["proba"],
            "policy_id": state["policy_id"],
            "matched_policies": [m["policy_id"]
                                 for m in state.get("matched_policies", [])],
            "governing_severity": state["severity"],
            "is_violation": state.get("is_violation", True),
            "trigger_words": [w for w, _ in state.get("trigger_words", [])[:10]],
            "verdict_source": state["verdict"].get("source"),
        }

    def route_after_classify(state):
        return "assess" if state["y_pred"] == 1 else "compliant"

    def route_after_assess(state):
        return "human_gate" if state.get("is_violation", True) else "cleared"

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("assess", assess_node)
    graph.add_node("human_gate", human_gate_node)
    graph.add_node("cleared", cleared_node)
    graph.add_node("compliant", compliant_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify,
                                {"assess": "assess", "compliant": "compliant"})
    graph.add_conditional_edges("assess", route_after_assess,
                                {"human_gate": "human_gate", "cleared": "cleared"})

    if writeback:
        graph.add_node("writeback", writeback_node)
        graph.add_edge("human_gate", "writeback")
        graph.add_edge("cleared", "writeback")
        graph.add_edge("compliant", "writeback")
        graph.add_edge("writeback", END)
    else:
        graph.add_edge("human_gate", END)
        graph.add_edge("cleared", END)
        graph.add_edge("compliant", END)
    return graph.compile()


def run_once(text, graph=None, proposal_id="PROP", threshold=DEFAULT_THRESHOLD,
             **kwargs):
    if graph is None:
        graph = build_graph(threshold=threshold, **kwargs)
    state = {"proposal_text": text, "proposal_id": proposal_id,
             "threshold": threshold}
    return graph.invoke(state)
