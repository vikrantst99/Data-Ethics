"""
The single seam between the agent and Jira.

Everything Jira-specific lives here. The rest of the pipeline (graph, streamlit)
only ever calls these three functions:

    list_proposals(limit)      -> [{"key","summary","text","status"}, ...]
    get_proposal(key)          -> "the proposal text"
    write_back(key, result, approved) -> {"mode","actions",...}

Two backends:

  * LIVE   - real Jira REST (API v2), used when a .env with credentials is present.
  * OFFLINE- reads the local hold-out CSV and writes the outbox mock files, so the
             app and the tests still run with no network and no credentials.

--------------------------------------------------------------------------------
Plugging in a colleague's Jira client
--------------------------------------------------------------------------------
If you receive a ready-made Jira module/notebook, you do NOT have to touch the
pipeline. Wrap it in a small object with these four methods and register it:

    class MyBackend:
        def list_issues(self, jql, limit): ...   # -> [{"key","summary","text","status"}]
        def get_issue(self, key): ...            # -> "text"
        def add_comment(self, key, text): ...    # -> url or id
        def create_issue(self, fields): ...      # -> new key

    import agent.jira_client as jc
    jc.set_backend(MyBackend())

From then on list_proposals / write_back use your colleague's code, and the
offline fallback is bypassed.
"""

import os
import re

from . import writeback as _wb


def _looks_like_issue_key(key):
    # A real Jira key is PROJECT-NUMBER, e.g. DE2-12. Manual proposals
    # ("streamlit_run", "pdf_upload") are not, and cannot be commented on.
    return bool(re.match(r"^[A-Z][A-Z0-9]+-\d+$", str(key)))


# --------------------------------------------------------------- credentials

def _load_env():
    # Read a local .env into os.environ, no external dependency.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # goal4/
    path = os.path.join(here, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def _config():
    _load_env()
    # Accept either JIRA_BASE_URL or JIRA_SERVER for the site address.
    base = os.environ.get("JIRA_BASE_URL") or os.environ.get("JIRA_SERVER", "")
    return {
        "base":    base.strip().rstrip("/"),
        "email":   os.environ.get("JIRA_EMAIL", ""),
        "token":   os.environ.get("JIRA_API_TOKEN", ""),
        "project": os.environ.get("JIRA_PROJECT_KEY", ""),
        "issuetype": os.environ.get("JIRA_ISSUE_TYPE", "Task"),
        "jql":     os.environ.get("JIRA_JQL", ""),
    }


def _have_credentials(cfg):
    return all(cfg[k] for k in ("base", "email", "token", "project"))


# --------------------------------------------------------------- live backend

class _RestBackend:
    """Jira Cloud REST v2. Description/comment bodies are plain strings."""

    def __init__(self, cfg):
        import requests
        from requests.auth import HTTPBasicAuth
        self._requests = requests
        self._cfg = cfg
        self._auth = HTTPBasicAuth(cfg["email"], cfg["token"])
        self._headers = {"Accept": "application/json",
                         "Content-Type": "application/json"}

    def _url(self, path):
        return self._cfg["base"] + path

    def list_issues(self, jql, limit):
        # New JQL search endpoint; the old /rest/api/2/search was removed in 2025.
        r = self._requests.post(
            self._url("/rest/api/2/search/jql"), auth=self._auth,
            headers=self._headers,
            json={"jql": jql, "fields": ["summary", "description", "status"],
                  "maxResults": limit}, timeout=30)
        r.raise_for_status()
        out = []
        for issue in r.json().get("issues", []):
            f = issue.get("fields", {})
            status = (f.get("status") or {}).get("name", "-")
            out.append({"key": issue["key"],
                        "summary": f.get("summary", ""),
                        "text": f.get("description") or "",
                        "status": status})
        return out

    def get_issue(self, key):
        r = self._requests.get(
            self._url("/rest/api/2/issue/" + key), auth=self._auth,
            headers=self._headers, params={"fields": "description"}, timeout=30)
        r.raise_for_status()
        return r.json().get("fields", {}).get("description") or ""

    def add_comment(self, key, text):
        r = self._requests.post(
            self._url("/rest/api/2/issue/" + key + "/comment"), auth=self._auth,
            headers=self._headers, json={"body": text}, timeout=30)
        r.raise_for_status()
        return self._cfg["base"] + "/browse/" + key

    def create_issue(self, fields):
        r = self._requests.post(
            self._url("/rest/api/2/issue"), auth=self._auth,
            headers=self._headers, json={"fields": fields}, timeout=30)
        r.raise_for_status()
        return r.json()["key"]


# --------------------------------------------------------------- offline backend

class _OfflineBackend:
    """No network: proposals come from the hold-out CSV, write-back is the outbox."""

    def __init__(self):
        self._rows = None

    def _load(self):
        if self._rows is None:
            import pandas as pd
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(os.path.dirname(here), "goal1",
                                "proposals_holdout_realistic_EN.csv")
            df = pd.read_csv(path).sort_values("proposal_id")
            self._rows = df
        return self._rows

    def list_issues(self, jql, limit):
        df = self._load().head(limit)
        return [{"key": r["proposal_id"],
                 "summary": str(r["ai_method"]) + " in " + str(r["project"]),
                 "text": str(r["description"]),
                 "status": "To Do"} for _, r in df.iterrows()]

    def get_issue(self, key):
        df = self._load()
        hit = df[df["proposal_id"] == key]
        return str(hit.iloc[0]["description"]) if len(hit) else ""

    def add_comment(self, key, text):
        return None      # handled by write_back via the outbox

    def create_issue(self, fields):
        return None


# --------------------------------------------------------------- backend choice

_BACKEND = None          # optional override set by a colleague's module


def set_backend(backend):
    # Register a custom Jira client (see the module docstring).
    global _BACKEND
    _BACKEND = backend


def _backend():
    if _BACKEND is not None:
        return _BACKEND, "custom"
    cfg = _config()
    if _have_credentials(cfg):
        try:
            return _RestBackend(cfg), "live"
        except Exception:
            pass
    return _OfflineBackend(), "offline"


def mode():
    return _backend()[1]


# --------------------------------------------------------------- public API

def _default_jql(cfg):
    if cfg["jql"]:
        return cfg["jql"]
    return ('project = ' + cfg["project"] + ' AND statusCategory != Done '
            'ORDER BY created DESC') if cfg["project"] else ""


def list_proposals(limit=20):
    # The proposals to choose from in Streamlit (Jira issues, or the CSV offline).
    backend, kind = _backend()
    jql = _default_jql(_config()) if kind in ("live", "custom") else ""
    return backend.list_issues(jql, limit)


def get_proposal(key):
    return _backend()[0].get_issue(key)


def create_proposal(summary, description):
    # Push one proposal into Jira as an issue (used to populate the project).
    # Only the text is sent - never the label.
    backend, _ = _backend()
    cfg = _config()
    return backend.create_issue({
        "project":   {"key": cfg["project"]},
        "issuetype": {"name": cfg["issuetype"]},
        "summary":   str(summary)[:250],
        "description": str(description),
        "labels":    ["compliance-sentinel", "proposal"],
    })


def _decision_comment(result, approved):
    lines = ["Compliance Sentinel decision",
             "Prediction: " + ("Red Flag" if result.get("y_pred") else "Compliant"),
             "Probability: " + str(result.get("proba")),
             "Governing severity: " + str(result.get("severity", "-")),
             "Human decision: " + ("APPROVED - remediation to apply"
                                    if approved else "REJECTED - no changes"),
             ""]
    for f in result.get("findings", []):
        lines.append("- " + str(f.get("policy_id")) + " (" +
                     str(f.get("severity")) + "): " + f.get("reason", ""))
        lines.append("  Fix: " + f.get("recommended_fix", ""))
    verdict = result.get("verdict", {})
    if verdict.get("provenance"):
        lines += ["", verdict["provenance"]]
    # Second tool: an optional LLM hint for the human. Advisory only - it did not
    # affect the decision above and is shown only when the LLM raised something.
    note = result.get("advisory_note")
    if note:
        lines += ["", "LLM hint for the reviewer (not part of the decision - "
                  "verify independently): " + note]
    return "\n".join(lines)


def write_back(key, result, approved):
    """Post the human-gated decision back to Jira.

    Live: a comment on the issue, plus a remediation ticket when an approved
    High-severity violation needs follow-up. Offline: the outbox mock files.
    Returns a dict describing what happened.
    """
    backend, kind = _backend()

    if kind == "offline":
        files = _wb.write_all(key, result)
        return {"mode": "offline", "actions": ["outbox files"], "files": files}

    if not _looks_like_issue_key(key):
        # Live Jira, but this proposal was typed/uploaded, not loaded from an
        # issue - there is nothing to comment on. Write to the outbox instead.
        files = _wb.write_all(key, result)
        return {"mode": "local", "files": files,
                "actions": ["outbox files (proposal is not a Jira issue - "
                            "load one from the picker to write back to Jira)"]}

    actions, urls = [], []
    comment_url = backend.add_comment(key, _decision_comment(result, approved))
    actions.append("comment on " + key)
    if comment_url:
        urls.append(comment_url)

    verdict = result.get("verdict", {})
    needs_ticket = (approved and result.get("y_pred") == 1
                    and verdict.get("is_violation", True)
                    and result.get("severity") == "High")
    if needs_ticket:
        cfg = _config()
        matched = ", ".join(m["policy_id"]
                            for m in result.get("matched_policies", [])) \
            or str(result.get("policy_id", "-"))
        summary = "Remediate " + matched + " in " + str(key)
        new_key = backend.create_issue({
            "project": {"key": cfg["project"]},
            "issuetype": {"name": cfg["issuetype"]},
            "summary": summary[:250],
            "description": _decision_comment(result, approved),
        })
        if new_key:
            actions.append("created ticket " + str(new_key))
            urls.append(cfg["base"] + "/browse/" + str(new_key))

    return {"mode": kind, "actions": actions, "urls": urls}
