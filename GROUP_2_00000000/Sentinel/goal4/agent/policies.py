"""
The RAI policy catalogue (fixed rules) and severity lookup.

Extended version: the severity of each policy can be overridden from a config
file (config/policies.csv with columns policy_id,severity). If that file is
absent, the built-in defaults are used - so the agent still runs out of the box.

Defaults source: Goal 1, cell 4 (project/RAI_Saetze.txt). RAI-01..05 High,
RAI-06..09 Medium.
"""

import csv
import os

RAI_POLICIES = {
    "RAI-01": {"name": "Data Protection", "severity": "High", "examples": [
        "No legal basis for the data processing is documented.",
        "Personal data is used without consent."]},
    "RAI-02": {"name": "Transparency", "severity": "High", "examples": [
        "The automated decision is not disclosed to data subjects.",
        "Users are not informed about the use of AI."]},
    "RAI-03": {"name": "Fairness", "severity": "High", "examples": [
        "The target group is selected by gender.",
        "Applicants are pre-filtered by origin and age."]},
    "RAI-04": {"name": "Human Dignity", "severity": "High", "examples": [
        "The system deliberately exploits vulnerabilities of protected groups.",
        "Vulnerable groups are processed without special safeguards."]},
    "RAI-05": {"name": "Prohibited Purpose", "severity": "High", "examples": [
        "The system rates people based on their social behaviour (social scoring).",
        "Subliminal techniques are used to influence behaviour.",
        "Employees' emotions are automatically recognised in the workplace."]},
    "RAI-06": {"name": "Security", "severity": "Medium", "examples": [
        "No security or access control is foreseen.",
        "The system goes to production without a security assessment."]},
    "RAI-07": {"name": "Human Oversight", "severity": "Medium", "examples": [
        "The decision is made fully automatically without human intervention.",
        "No human-in-the-loop is foreseen."]},
    "RAI-08": {"name": "Data Minimization", "severity": "Medium", "examples": [
        "More data fields are collected than needed for the purpose.",
        "Data is stored on stock for unknown purposes."]},
    "RAI-09": {"name": "Explainability", "severity": "Medium", "examples": [
        "The model is a black box without any means of justification.",
        "Decisions cannot be traced or explained."]},
}

ALL_POLICY_IDS = list(RAI_POLICIES.keys())


def _config_path():
    here = os.path.dirname(os.path.abspath(__file__))   
    root = os.path.dirname(here)                          
    return os.path.join(root, "config", "policies.csv")


def load_severity_overrides():
    # Read config/policies.csv if present. Returns {policy_id: severity}
    path = _config_path()
    overrides = {}
    if not os.path.exists(path):
        return overrides
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pid = (row.get("policy_id") or "").strip().upper()
            sev = (row.get("severity") or "").strip().capitalize()
            if pid and sev:
                overrides[pid] = sev
    return overrides


# Apply the overrides once at import time
for _pid, _sev in load_severity_overrides().items():
    if _pid in RAI_POLICIES:
        RAI_POLICIES[_pid]["severity"] = _sev


def get_severity(policy_id):
    policy = RAI_POLICIES.get(str(policy_id).upper())
    return policy["severity"] if policy else None


def worst_severity(policy_ids):
    # The governing severity of a set of policies: High wins over Medium.
    severities = [get_severity(pid) for pid in policy_ids]
    if "High" in severities:
        return "High"
    if "Medium" in severities:
        return "Medium"
    return "Unknown"


def policy_text(policy_id):
    pid = str(policy_id).upper()
    policy = RAI_POLICIES.get(pid)
    if policy is None:
        return pid + " is not a recognised RAI policy (valid: RAI-01..RAI-09)."
    lines = [pid + " " + policy["name"] + " [" + policy["severity"] + "]"]
    for example in policy["examples"]:
        lines.append("- " + example)
    return "\n".join(lines)


def policy_docs():
    return [(pid, policy_text(pid)) for pid in ALL_POLICY_IDS]
