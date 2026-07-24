"""
Goal 4 (extended): The Autonomous Compliance Sentinel Agent

Same core as the basic agent (classify -> assess -> human_gate) plus the
optional Section-3 add-ons:

- writeback: Confluence comment + Jira ticket + audit log to the outbox
- batch: run over a whole inbox folder
- certify: Goal-3 weakness pre-gate before the agent is allowed to ship
- configurable severity via config/policies.csv
- borderline LLM routing and open-ended verification (in graph/llm)
- ui: ipywidgets front-end (with a graceful fallback)
- Streamlit app: see streamlit_app.py

Everything still runs offline.
"""
import os as _os


def _load_dotenv():
    # Load a local .env (GROQ_API_KEY, JIRA_* ...) into the environment at import
    # time, so the LLM and Jira pick up their credentials. No external dependency.
    here = _os.path.dirname(_os.path.abspath(__file__))        # .../agent
    for base in (here, _os.path.dirname(here)):                # agent/ and goal4/
        path = _os.path.join(base, ".env")
        if not _os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    _os.environ.setdefault(key.strip(),
                                           value.strip().strip('"').strip("'"))


_load_dotenv()

from .classify import Classifier, DEFAULT_THRESHOLD
from .policies import (RAI_POLICIES, get_severity, worst_severity, policy_text,
                       policy_docs, ALL_POLICY_IDS, load_severity_overrides)
from .rag import PolicyRetriever
from .llm import (compose_verdict, groq_available, provenance_note,
                  per_policy_finding)
from .advisor import policy_advice
from .graph import build_graph, run_once, DEFAULT_LLM_BAND
from . import writeback
from .pdf import extract_text
from .batch import run_batch
from .certify import certify
from . import ui
from . import jira_client

__all__ = [
    "Classifier", "DEFAULT_THRESHOLD",
    "RAI_POLICIES", "get_severity", "worst_severity", "policy_text",
    "policy_docs", "ALL_POLICY_IDS", "load_severity_overrides",
    "PolicyRetriever",
    "compose_verdict", "groq_available", "provenance_note", "per_policy_finding",
    "policy_advice",
    "build_graph", "run_once", "DEFAULT_LLM_BAND",
    "writeback", "extract_text", "run_batch", "certify", "ui", "jira_client",
]
__version__ = "1.0"
