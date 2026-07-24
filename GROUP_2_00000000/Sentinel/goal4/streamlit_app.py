"""Streamlit web front-end for the extended agent (Section-3 UI option).

Run it from THIS folder (not with `python`, but with the streamlit runner):

    pip install streamlit
    streamlit run streamlit_app.py

Streamlit re-runs this script top to bottom on every interaction, so the model
and the graph are cached with @st.cache_resource. The High-severity human gate
becomes real APPROVE / REJECT buttons instead of a terminal input().
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

from matplotlib import pyplot as plt

from agent import (Classifier, PolicyRetriever, build_graph, run_once,
                   groq_available, get_severity, writeback, jira_client)
from agent import loader
from agent.pdf import extract_text
import charts

DEFAULT_TEXT = ("Our chatbot will handle customer complaints automatically "
                "without telling users it is an AI.")

# Severity -> Streamlit badge colour (kept in sync with the chart palette).
SEV_BADGE = {"High": "red", "Medium": "orange", "Low": "gray"}


def reset_state():
    # Clear the last assessment so the UI is ready for a fresh request. Runs as
    # an on_click callback, i.e. before the widgets are re-instantiated, so it is
    # allowed to reset the text_area's session_state key.
    for _key in ("result", "writeback_outcome"):
        st.session_state.pop(_key, None)
    st.session_state["proposal_text"] = DEFAULT_TEXT
    st.session_state["jira_key"] = "streamlit_run"


# Built once and reused across reruns (loading the model is the slow part).
@st.cache_resource(show_spinner="Loading Goal-2 model and RAG index ...")
def get_engine():
    classifier = Classifier(loader.load_model("tfidf+logreg"))
    retriever = PolicyRetriever()
    # Auto-approve inside the graph; the real APPROVE/REJECT decision is the buttons below.
    # llm_band=None -> the LLM phrases every Red Flag (full deployment), not only
    # the borderline band.
    graph = build_graph(classifier=classifier, retriever=retriever,
                        approver=lambda state: True, writeback=False, verify=True,
                        llm_band=None)
    return classifier, retriever, graph


def theme_ink():
    # Pick chart ink for the viewer's Streamlit theme (light or dark).
    try:
        base = st.get_option("theme.base") or "light"
    except Exception:
        base = "light"
    return charts.dark_ink() if base == "dark" else charts.light_ink()


def show(fig):
    # Render a matplotlib figure and free it (Streamlit reruns often).
    st.pyplot(fig)
    plt.close(fig)


def inject_css():
    # A small, self-contained modern skin on top of the theme in config.toml.
    st.markdown(
        """
        <style>
        html, body, .stApp, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                         "Helvetica Neue", Arial, sans-serif;
        }
        .block-container { padding-top: 3.6rem; padding-bottom: 3rem;
                           max-width: 1760px;
                           padding-left: 2.4rem; padding-right: 2.4rem; }

        /* Header */
        .app-header { margin-bottom: 4px; padding-top: 4px; }
        .app-title { font-size: 1.95rem; font-weight: 700; letter-spacing: -.02em;
                     color: #14181f; line-height: 1.3; padding-top: 2px; }
        .app-sub { color: #5b6472; font-size: .95rem; margin-top: 3px; }

        /* Status pills */
        .statusbar { display: flex; gap: 8px; flex-wrap: wrap;
                     margin: 14px 0 8px; }
        .pill { display: inline-flex; align-items: center; gap: 7px;
                padding: 4px 11px; border-radius: 999px; background: #eef1f5;
                border: 1px solid #e5e8ec; }
        .pill-k { color: #5b6472; font-weight: 700; text-transform: uppercase;
                  letter-spacing: .05em; font-size: 10.5px; }
        .pill-v { color: #14181f; font-weight: 600; font-size: 12.5px; }
        .pill-ok { background: #e8f5ee; border-color: #c4e6d3; }
        .pill-ok .pill-v { color: #0a7d43; }
        .pill-warn { background: #fff4e0; border-color: #ffe0a8; }
        .pill-warn .pill-v { color: #a8650a; }

        /* Page background is white; only the frame interiors are tinted. */
        .stApp { background: #ffffff; }

        /* Bordered containers -> tinted cards against the white page */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #f4f6f8; border: 1px solid #e5e8ec !important;
            border-radius: 14px;
            box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.05);
        }

        /* Metric cards */
        [data-testid="stMetric"] { background: #ffffff; border: 1px solid #e5e8ec;
            border-radius: 12px; padding: 12px 16px; }
        [data-testid="stMetricLabel"] p { color: #5b6472; font-weight: 600;
            font-size: .78rem; text-transform: uppercase; letter-spacing: .03em; }

        /* Buttons */
        .stButton > button { border-radius: 9px; border: 1px solid #d5dae1;
            font-weight: 600; transition: all .15s ease; padding: .45rem 1rem; }
        .stButton > button:hover { border-color: #00699A; color: #00699A; }
        .stButton > button[kind="primary"] { background: #00699A;
            border-color: #00699A; color: #fff; }
        .stButton > button[kind="primary"]:hover { background: #005a86;
            border-color: #005a86; color: #fff; }

        /* Inputs */
        .stTextArea textarea { border-radius: 10px; }
        [data-baseweb="select"] > div { border-radius: 10px; }

        /* Section labels */
        .sec-label { color: #5b6472; font-weight: 700; font-size: .74rem;
            text-transform: uppercase; letter-spacing: .06em; margin: 2px 0 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_bar(llm_value, rag_value, jira_value):
    def pill(key, value, tone="neutral"):
        cls = "pill" + ("" if tone == "neutral" else " pill-" + tone)
        return ('<span class="' + cls + '"><span class="pill-k">' + key +
                '</span><span class="pill-v">' + value + '</span></span>')
    html = ('<div class="statusbar">' +
            pill("LLM", llm_value, "ok" if groq_available() else "warn") +
            pill("RAG", rag_value) +
            pill("Jira", jira_value, "ok" if jira_value == "live" else "neutral") +
            "</div>")
    st.markdown(html, unsafe_allow_html=True)


st.set_page_config(page_title="Autonomous Compliance Sentinel",
                   page_icon="🛡️", layout="wide")
inject_css()

# Reuse the cached engine - building a new PolicyRetriever here would rebuild the vector store on every rerun.
_, retriever, graph = get_engine()

st.markdown(
    '<div class="app-header">'
    '<div class="app-title">Autonomous Compliance Sentinel</div>'
    '<div class="app-sub">RAI policy screening for AI project proposals '
    '&mdash; human-in-the-loop review</div></div>',
    unsafe_allow_html=True)
status_bar("Groq" if groq_available() else "offline fallback",
           retriever.mode, jira_client.mode())


@st.cache_data(show_spinner="Loading proposals from Jira ...")
def get_proposals(limit=20):
    # Cached list of Jira issues (or the local hold-out when offline).
    try:
        return jira_client.list_proposals(limit)
    except Exception as error:
        st.warning("Could not reach Jira (" + str(error) + "). "
                   "Falling back to manual entry.")
        return []

# The proposal text lives in session_state so a picker or PDF upload can fill it.
st.session_state.setdefault("proposal_text", DEFAULT_TEXT)
st.session_state.setdefault("jira_key", "streamlit_run")

# Left = interface (wider), right = analysis charts.
col_ui, col_charts = st.columns([3, 2], gap="large")

with col_ui:
    with st.container(border=True):
        st.markdown('<div class="sec-label">Proposal</div>',
                    unsafe_allow_html=True)
        # 1) Pick a proposal straight from Jira (falls back to the local hold-out).
        proposals = get_proposals()
        if proposals:
            labels = {p["key"] + " — " + p["summary"]: p for p in proposals}
            choice = st.selectbox("Select a proposal from Jira",
                                  list(labels.keys()))
            if st.button("Load selected proposal"):
                picked = labels[choice]
                st.session_state["proposal_text"] = picked["text"]
                st.session_state["jira_key"] = picked["key"]
                st.session_state.pop("result", None)
                st.session_state.pop("writeback_outcome", None)
                st.success("Loaded " + picked["key"] +
                           ". Review the text below, then click Assess.")
        else:
            st.info("No proposals to choose from. Jira is **" +
                    jira_client.mode() + "** and the project has no issues yet "
                    "— paste a proposal below or upload a PDF to assess one now.")

        uploaded = st.file_uploader("...or upload a proposal as PDF (optional)",
                                    type=["pdf"])
        if uploaded is not None:
            try:
                st.session_state["proposal_text"] = extract_text(uploaded.getvalue())
                st.session_state["jira_key"] = "pdf_upload"
                st.success("Read " + str(len(st.session_state["proposal_text"])) +
                           " characters from " + uploaded.name +
                           ". Review the text below, then click Assess.")
            except Exception as error:
                st.error("Could not read the PDF: " + str(error))

        text = st.text_area("Proposal text", height=160, key="proposal_text")
        jira_key = st.session_state["jira_key"]
        st.caption("Source: **" + jira_key + "**")

        assess = st.button("Assess", type="primary")

    if assess:
        st.session_state.pop("writeback_outcome", None)
        st.session_state["result"] = run_once(text.strip(), graph=graph,
                                               proposal_id=jira_key)

    result = st.session_state.get("result")
    if result:
        with st.container(border=True):
            st.markdown('<div class="sec-label">Assessment</div>',
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction",
                      "Red Flag" if result["y_pred"] == 1 else "Compliant",
                      border=True)
            c2.metric("Probability", result["proba"], border=True)
            c3.metric("Policy", result.get("policy_id", "-"), border=True)

            if result["y_pred"] == 1:
                verdict = result["verdict"]
                st.markdown("**Governing severity**")
                st.badge(result["severity"],
                         color=SEV_BADGE.get(result["severity"], "gray"))

                st.markdown("**Policies matched**")
                cols = st.columns(max(len(result["matched_policies"]), 1))
                for col, m in zip(cols, result["matched_policies"]):
                    col.badge(m["policy_id"] + " · " + str(m["severity"]),
                              color=SEV_BADGE.get(m["severity"], "gray"))

                high_count = 0
                for m in result["matched_policies"]:
                    if m["severity"] == "High":
                        high_count += 1
                if high_count >= 1 and len(result["matched_policies"]) > 1:
                    st.warning("Multiple / severe violations — mandatory human review.")

                findings = result.get("findings", [])
                st.markdown("#### Findings (" + str(len(findings)) + ")")
                for f in findings:
                    with st.expander(f["policy_id"] + " — " + str(f["severity"]),
                                     expanded=(len(findings) == 1)):
                        st.write("**Reason:** " + f["reason"])
                        st.write("**Recommended fix:** " + f["recommended_fix"])
                st.caption("ℹ️ " + verdict["provenance"])

                # Second tool: an optional LLM hint for the reviewer. Shown only
                # when the LLM raised something relevant; it did not affect the
                # decision above.
                advisory_note = result.get("advisory_note")
                if advisory_note:
                    st.info("💡 **LLM hint for the reviewer** (not part of the "
                            "decision — verify independently): " + advisory_note)

                # The human gate as buttons. High severity must be decided by a
                # human; either way the decision is written back to Jira.
                def _report_writeback(outcome):
                    if outcome.get("mode") in ("offline", "local"):
                        st.caption("Wrote outbox files: " +
                                   ", ".join(outcome.get("files", [])))
                        for note in outcome.get("actions", []):
                            st.caption(note)
                    else:
                        st.caption("Jira (" + outcome.get("mode", "-") + "): " +
                                   "; ".join(outcome.get("actions", [])))
                        for url in outcome.get("urls", []):
                            st.markdown("→ [" + url + "](" + url + ")")

                def do_writeback(approved, ok_msg):
                    # Never let a Jira API error tear down the page. Persist the
                    # outcome so the message and the reset button survive reruns.
                    try:
                        outcome = jira_client.write_back(jira_key, result,
                                                         approved=approved)
                    except Exception as error:
                        st.session_state["writeback_outcome"] = {
                            "error": str(error)}
                        return
                    st.session_state["writeback_outcome"] = {
                        "approved": approved, "msg": ok_msg, "outcome": outcome}

                st.divider()
                if result["severity"] == "High":
                    st.markdown('<div class="sec-label">Human gate '
                                '(High severity)</div>', unsafe_allow_html=True)
                    st.warning("High severity — human approval required before "
                               "write-back.")
                    approve_col, reject_col = st.columns(2)
                    if approve_col.button("APPROVE", type="primary",
                                          use_container_width=True):
                        do_writeback(True, "Approved and written back to " +
                                     jira_key + ".")
                    if reject_col.button("REJECT", use_container_width=True):
                        do_writeback(False, "Rejected — a 'no changes' note was "
                                     "written for " + jira_key + ".")
                else:
                    st.info(result["final_decision"])
                    if st.button("Write decision back to Jira", type="primary"):
                        do_writeback(True, "Written back for " + jira_key + ".")

                # After a decision: show the persisted result + a reset button
                # to clear the screen for the next request.
                wb = st.session_state.get("writeback_outcome")
                if wb:
                    if wb.get("error"):
                        st.error("Write-back to Jira failed: " + wb["error"])
                    else:
                        (st.success if wb["approved"] else st.error)(wb["msg"])
                        _report_writeback(wb["outcome"])
                    st.button("↻ Start new assessment", on_click=reset_state)
            else:
                st.success(result["final_decision"])

with col_charts:
    # Analysis charts (A1-A3), built only from this proposal's result.
    if result and result["y_pred"] == 1:
        st.markdown('<div class="sec-label">Analysis</div>',
                    unsafe_allow_html=True)
        ink = theme_ink()
        with st.container(border=True):
            show(charts.fig_probability(result["proba"], threshold=0.40, ink=ink))
        with st.container(border=True):
            show(charts.fig_trigger_words(result["trigger_words"], ink=ink))
        severities = {h["policy_id"]: get_severity(h["policy_id"])
                      for h in result["retrieved"]}
        matched_ids = [m["policy_id"] for m in result["matched_policies"]]
        with st.container(border=True):
            show(charts.fig_policy_scores(result["retrieved"], matched_ids,
                                          severities, ink=ink))
    elif result:
        st.info("Compliant — no analysis charts.")
    else:
        st.caption("Charts appear here after you assess a proposal.")
