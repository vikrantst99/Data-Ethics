"""
Batch runner for Goal 4 (extended) - process the whole inbox.

Reads every *.txt in 'inbox', assesses it and writes the Confluence/Jira/audit
files to 'outbox'. Prints a summary table at the end.

Usage:
    python run_batch.py
"""

from agent import run_batch, groq_available


def main():
    print("LLM backend:", "Groq" if groq_available() else "offline fallback")
    summary = run_batch(writeback=True)

    print("\n--- BATCH SUMMARY ---")
    header = "{:<28} {:<9} {:>6} {:<8} {:<7} {}".format(
        "proposal", "pred", "proba", "policy", "sev", "violation")
    print(header)
    for row in summary:
        print("{:<28} {:<9} {:>6} {:<8} {:<7} {}".format(
            row["proposal_id"][:28], row["prediction"], row["proba"],
            str(row["policy_id"]), str(row["severity"]), str(row["is_violation"])))
    print("\nWrote Confluence/Jira/audit files to the 'outbox' folder.")


if __name__ == "__main__":
    main()
