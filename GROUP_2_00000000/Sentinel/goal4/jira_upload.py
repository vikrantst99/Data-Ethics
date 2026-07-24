"""
Populate the Jira project with proposals from our test dataset.

Reads rows from goal1/proposals_holdout_realistic_EN.csv and creates one Jira
issue per proposal, through the same adapter the agent uses (agent.jira_client).
Only the proposal text is uploaded - the true label stays local in a mapping
file so the agent's later decision can be scored against it.

    python jira_upload.py --dry-run       # show what would be created, no upload
    python jira_upload.py --limit 6       # create 6 issues (default)
    python jira_upload.py --all           # create all 250

Credentials come from goal4/.env (JIRA_SERVER / JIRA_EMAIL / JIRA_API_TOKEN /
JIRA_PROJECT_KEY), loaded automatically by the agent package.
"""

import os
import csv
import time
import argparse

import pandas as pd

from agent import jira_client as jc


def _dataset_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "goal1",
                        "proposals_holdout_realistic_EN.csv")


def select_rows(df, limit, take_all):
    if take_all:
        return df.reset_index(drop=True)
    df = df.sort_values("proposal_id")
    reds = df[df["label"] == "Red Flag"]
    oks = df[df["label"] == "Compliant"]
    n_red = (limit + 1) // 2
    picked = pd.concat([reds.head(n_red), oks.head(limit - n_red)])
    return picked.sort_values("proposal_id").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument("--out", default="jira_upload_map.csv")
    args = parser.parse_args()

    df = pd.read_csv(_dataset_path())
    rows = select_rows(df, args.limit, args.all)
    print("Jira mode: " + jc.mode())
    print("Selected " + str(len(rows)) + " proposals ("
          + str((rows["label"] == "Red Flag").sum()) + " Red Flag, "
          + str((rows["label"] == "Compliant").sum()) + " Compliant).")

    if jc.mode() == "offline" and not args.dry_run:
        raise SystemExit("Jira is offline (no credentials in .env). "
                         "Nothing uploaded. Use --dry-run to preview.")

    if args.dry_run:
        for _, r in rows.iterrows():
            print("\n--- " + r["proposal_id"] + " (true label: " + r["label"]
                  + ", kept local) ---")
            print("  " + str(r["ai_method"]) + " in " + str(r["project"]))
            print("  " + str(r["description"])[:110] + "...")
        print("\nDry run only. No issues created.")
        return

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["proposal_id", "jira_key", "true_label", "red_flags"])
        for _, r in rows.iterrows():
            summary = r["proposal_id"] + ": " + str(r["ai_method"]) + \
                " in " + str(r["project"])
            key = jc.create_proposal(summary, r["description"])
            writer.writerow([r["proposal_id"], key, r["label"], r["red_flags"]])
            print(r["proposal_id"] + "  ->  " + str(key))
            time.sleep(args.sleep)

    print("\nDone. Local map (with true labels, kept out of Jira): " + out_path)


if __name__ == "__main__":
    main()
