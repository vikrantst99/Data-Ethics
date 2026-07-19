"""Writeback: the last step of the concept flow (Kommentar + Ticket + Audit-Log).

The basic agent only printed an audit entry. Here we actually write files to an
'outbox' folder: a Confluence-style comment, a Jira-style ticket (only when a
human is needed) and one line per decision in an audit log (JSONL). These are
local mock files - stand-ins for real Confluence/Jira API calls.
"""

import os
import json
import datetime


def _outbox(outbox=None):
    # Use the given folder or the default 'outbox'; always make sure it exists
    if outbox is None:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        outbox = os.path.join(root, "outbox")
    os.makedirs(outbox, exist_ok=True)
    return outbox


def _slug(name):
    keep = ""
    for char in str(name).lower():
        if char.isalnum():
            keep += char
        else:
            keep += "_"
    return keep.strip("_")[:40] or "proposal"


def write_confluence_comment(proposal_id, result, outbox=None):
    # A short markdown comment summarising the verdict for the proposal page.
    outbox = _outbox(outbox)
    verdict = result.get("verdict", {})
    lines = [
        "# Compliance Sentinel - " + str(proposal_id),
        "",
        "- **Prediction:** " + ("Red Flag" if result["y_pred"] else "Compliant"),
        "- **Probability:** " + str(result["proba"]),
        "- **Policies matched:** " +
        (", ".join(m["policy_id"] + " (" + str(m["severity"]) + ")"
                   for m in result.get("matched_policies", []))
         or str(result.get("policy_id", "-"))),
        "- **Governing severity:** " + str(result.get("severity", "-")),
        "- **Decision:** " + result.get("final_decision", "-"),
        "",
        "## Findings",
    ]
    for f in result.get("findings", []) or [verdict]:
        lines += [
            "### " + str(f.get("policy_id", "-")) +
            " (" + str(f.get("severity", "-")) + ")",
            "- **Reason:** " + f.get("reason", "-"),
            "- **Recommended fix:** " + f.get("recommended_fix", "-"),
        ]
    lines += [
        "",
        "> " + verdict.get("provenance",
                           "Red-Flag decision by the trained model; "
                           "reason/fix may be machine-generated."),
    ]
    path = os.path.join(outbox, _slug(proposal_id) + "_confluence_comment.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def write_jira_ticket(proposal_id, result, outbox=None):
    # A ticket, only for High-severity findings that need remediation.
    outbox = _outbox(outbox)
    verdict = result.get("verdict", {})
    matched = ", ".join(m["policy_id"] for m in result.get("matched_policies", [])) \
        or str(result.get("policy_id", "-"))
    lines = [
        "TICKET: Remediate " + matched + " in " + str(proposal_id),
        "Type    : Compliance",
        "Priority: " + ("High" if result.get("severity") == "High" else "Medium"),
        "Policies: " + matched,
        "Decision: " + result.get("final_decision", "-"),
        "",
    ]
    for f in result.get("findings", []) or [verdict]:
        lines += [
            "- " + str(f.get("policy_id", "-")) +
            " (" + str(f.get("severity", "-")) + ")",
            "    Reason: " + f.get("reason", "-"),
            "    Action: " + f.get("recommended_fix", "-"),
        ]
    lines += [
        "",
        "Note    : " + verdict.get("provenance",
                                   "Red-Flag decision by the trained model; "
                                   "reason/fix may be machine-generated."),
    ]
    path = os.path.join(outbox, _slug(proposal_id) + "_jira_ticket.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def append_audit_log(proposal_id, result, outbox=None):
    # One JSON line per decision - the machine-readable Art.-12 trail
    outbox = _outbox(outbox)
    entry = dict(result.get("audit", {}))
    entry["proposal_id"] = str(proposal_id)
    entry["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    path = os.path.join(outbox, "agent_log.jsonl")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return path


def write_all(proposal_id, result, outbox=None):
    # Write everything that applies and return the list of files created
    outbox = _outbox(outbox)
    written = [write_confluence_comment(proposal_id, result, outbox)]
    # Only open a ticket when a real violation needs human-approved remediation
    verdict = result.get("verdict", {})
    if result.get("y_pred") == 1 and verdict.get("is_violation", True):
        written.append(write_jira_ticket(proposal_id, result, outbox))
    written.append(append_audit_log(proposal_id, result, outbox))
    return written
