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
                   groq_available, get_severity, writeback)
from agent import loader
from agent.pdf import extract_text
import charts

DEFAULT_TEXT = ("Our chatbot will handle customer complaints automatically "
                "without telling users it is an AI.")


# Built once and reused across reruns (loading the model is the slow part).
@st.cache_resource(show_spinner="Loading Goal-2 model and RAG index ...")
def get_engine():
    classifier = Classifier(loader.load_model("tfidf+logreg"))
    retriever = PolicyRetriever()
    # Auto-approve inside the graph; the real APPROVE/REJECT decision is the buttons below.
    graph = build_graph(classifier=classifier, retriever=retriever,
                        approver=lambda state: True, writeback=False, verify=True)
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


st.set_page_config(page_title="Autonomous Compliance Sentinel", layout="wide")
st.title("Autonomous Compliance Sentinel")

# Reuse the cached engine - building a new PolicyRetriever here would rebuild the vector store on every rerun.
_, retriever, graph = get_engine()
st.caption("LLM backend: **" + ("Groq" if groq_available() else "offline fallback") +
           "**  ·  RAG mode: **" + retriever.mode + "**")

# The proposal text lives in session_state so a PDF upload can fill it in.
st.session_state.setdefault("proposal_text", DEFAULT_TEXT)

# Left = interface (wider), right = analysis charts.
col_ui, col_charts = st.columns([3, 2], gap="large")

with col_ui:
    uploaded = st.file_uploader("Upload a proposal as PDF (optional)",
                                type=["pdf"])
    if uploaded is not None:
        try:
            st.session_state["proposal_text"] = extract_text(uploaded.getvalue())
            st.success("Read " + str(len(st.session_state["proposal_text"])) +
                       " characters from " + uploaded.name +
                       ". Review the text below, then click Assess.")
        except Exception as error:
            st.error("Could not read the PDF: " + str(error))

    text = st.text_area("Proposal text", height=160, key="proposal_text")

    if st.button("Assess", type="primary"):
        st.session_state["result"] = run_once(text.strip(), graph=graph,
                                               proposal_id="streamlit_run")

    result = st.session_state.get("result")
    if result:
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction",
                  "Red Flag" if result["y_pred"] == 1 else "Compliant")
        c2.metric("Probability", result["proba"])
        c3.metric("Policy", result.get("policy_id", "-"))

        if result["y_pred"] == 1:
            verdict = result["verdict"]
            matched = ", ".join(m["policy_id"] + " (" + str(m["severity"]) + ")"
                                for m in result["matched_policies"])
            st.subheader("Governing severity: " + result["severity"])
            st.write("**Policies matched:** " + matched)
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

            # The human gate as buttons (only for High severity).
            if result["severity"] == "High":
                st.warning("High severity — human approval required.")
                approve_col, reject_col = st.columns(2)
                if approve_col.button("APPROVE"):
                    files = writeback.write_all("streamlit_run", result)
                    st.success("Approved. Wrote: " + ", ".join(files))
                if reject_col.button("REJECT"):
                    st.error("Rejected. No changes made.")
            else:
                st.info(result["final_decision"])
        else:
            st.success(result["final_decision"])

with col_charts:
    # Analysis charts (A1-A3), built only from this proposal's result.
    if result and result["y_pred"] == 1:
        st.markdown("#### Analysis")
        ink = theme_ink()
        show(charts.fig_probability(result["proba"], threshold=0.40, ink=ink))
        show(charts.fig_trigger_words(result["trigger_words"], ink=ink))
        severities = {h["policy_id"]: get_severity(h["policy_id"])
                      for h in result["retrieved"]}
        matched_ids = [m["policy_id"] for m in result["matched_policies"]]
        show(charts.fig_policy_scores(result["retrieved"], matched_ids,
                                      severities, ink=ink))
    elif result:
        st.info("Compliant — no analysis charts.")
    else:
        st.caption("Charts appear here after you assess a proposal.")
