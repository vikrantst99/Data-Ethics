"""
Batch mode: run the agent over a whole folder of proposals.

Each *.txt file in the inbox is one proposal. For every file the agent decides,
explains and (if writeback is on) writes the Confluence/Jira/audit files to the
outbox. And returns a small summary table.
"""

import os

from .graph import build_graph, run_once


def _inbox_default():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "inbox")


def run_batch(inbox=None, graph=None, writeback=True, outbox=None, **kwargs):
    inbox = inbox or _inbox_default()
    if not os.path.isdir(inbox):
        raise FileNotFoundError("Inbox folder not found: " + inbox)
    if graph is None:
        graph = build_graph(writeback=writeback, outbox=outbox, **kwargs)

    files = sorted(f for f in os.listdir(inbox) if f.endswith(".txt"))
    summary = []
    for name in files:
        with open(os.path.join(inbox, name), encoding="utf-8") as handle:
            text = handle.read().strip()
        proposal_id = os.path.splitext(name)[0]
        result = run_once(text, graph=graph, proposal_id=proposal_id)
        summary.append({
            "proposal_id": proposal_id,
            "prediction": "Red Flag" if result["y_pred"] == 1 else "Compliant",
            "proba": result["proba"],
            "policy_id": result.get("policy_id", "-"),
            "severity": result.get("severity", "-"),
            "is_violation": result.get("is_violation", "-"),
            "decision": result.get("final_decision", "-"),
        })
    return summary
