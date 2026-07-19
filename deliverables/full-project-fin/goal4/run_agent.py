"""
Start file for Goal 4 (extended) - run THIS file for a single proposal.

Optionally runs the Goal-3 weakness pre-gate first, then assesses one proposal
and writes the Confluence/Jira/audit files to the outbox.

Usage:
    python run_agent.py
    python run_agent.py "my proposal text ..."
"""

import sys

from agent import build_graph, run_once, groq_available, certify


SAMPLE = ("Our chatbot will handle customer complaints automatically "
          "without telling users it is an AI.")


def ask_human(state):
    print("\n--- HIGH SEVERITY: human approval required ---")
    print("Proposal:", state["proposal_text"])
    print("Verdict :", state["verdict"]["reason"])
    return input("Type APPROVE or REJECT: ").strip().upper() == "APPROVE"


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    print("LLM backend:", "Groq" if groq_available() else "offline fallback")

    # Section-3 option: certify the model before using it.
    cert = certify()
    if cert.get("available"):
        print("Weakness pre-gate:", "PASSED" if cert["passed"] else "FAILED",
              "(worst policy", cert["worst_policy"], "recall",
              cert["worst_policy_recall"], "| negation-blind pairs",
              cert["negation_blind_pairs"], ")")
    else:
        print("Weakness pre-gate: skipped (Goal-3 xai not importable)")

    graph = build_graph(approver=ask_human, writeback=True)
    result = run_once(text, graph=graph, proposal_id="manual_run")

    print("\n--- RESULT ---")
    print("Prediction :", "Red Flag" if result["y_pred"] == 1 else "Compliant")
    print("Probability:", result["proba"])
    if result["y_pred"] == 1:
        verdict = result["verdict"]
        matched = ", ".join(m["policy_id"] + "(" + m["severity"] + ")"
                            for m in result["matched_policies"])
        print("Policies   :", matched, "-> governing:", result["severity"])
        print("Violation  :", verdict.get("is_violation", True))
        print("Findings   :")
        for f in result["findings"]:
            print("  [" + f["policy_id"] + " / " + f["severity"] + "]")
            print("    Reason:", f["reason"])
            print("    Fix   :", f["recommended_fix"])
        print("Note       :", verdict["provenance"])
    print("Decision   :", result["final_decision"])
    print("Wrote      :", result.get("written_files", []))


if __name__ == "__main__":
    main()
