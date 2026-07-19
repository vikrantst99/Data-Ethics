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
from .classify import Classifier, DEFAULT_THRESHOLD
from .policies import (RAI_POLICIES, get_severity, worst_severity, policy_text,
                       policy_docs, ALL_POLICY_IDS, load_severity_overrides)
from .rag import PolicyRetriever
from .llm import (compose_verdict, groq_available, provenance_note,
                  per_policy_finding)
from .graph import build_graph, run_once, DEFAULT_LLM_BAND
from . import writeback
from .pdf import extract_text
from .batch import run_batch
from .certify import certify
from . import ui

__all__ = [
    "Classifier", "DEFAULT_THRESHOLD",
    "RAI_POLICIES", "get_severity", "worst_severity", "policy_text",
    "policy_docs", "ALL_POLICY_IDS", "load_severity_overrides",
    "PolicyRetriever",
    "compose_verdict", "groq_available", "provenance_note", "per_policy_finding",
    "build_graph", "run_once", "DEFAULT_LLM_BAND",
    "writeback", "extract_text", "run_batch", "certify", "ui",
]
__version__ = "1.0"
