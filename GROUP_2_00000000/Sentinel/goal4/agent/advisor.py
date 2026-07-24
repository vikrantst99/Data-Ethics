"""
Second tool: a policy second-opinion advisor (LLM consultation).

This is the agent's SECOND tool, next to the Jira tool. It deliberately takes
NO part in the compliance decision: classification, governing severity and the
findings are produced by the deterministic pipeline (classifier + RAG + verdict)
and are left completely untouched.

The advisor asks the LLM one single question - "is there a policy angle a HUMAN
reviewer should personally double-check?" - and returns a short hint only when,
and only when, the LLM raises something relevant. If nothing stands out (or no
LLM is available) it returns an empty note and the agent shows nothing.

The note is advisory context for the human, never an instruction the agent acts
on. It is surfaced clearly marked as LLM-generated and to be verified
independently.
"""

from .policies import RAI_POLICIES
from .llm import groq_available

# The single word the LLM must return when it has nothing worth a human's time.
_NONE = "NONE"


def _catalogue_lines():
    # Short catalogue so the LLM can spot an angle the automated match may miss.
    return "\n".join(pid + " " + pol["name"] + " [" + pol["severity"] + "]"
                     for pid, pol in RAI_POLICIES.items())


def policy_advice(proposal_text, matched_policies, use_llm=True):
    """Consult the LLM for a human-only hint. Returns {"note", "source", ...}.

    note == "" means: show nothing (nothing relevant, or no LLM available).
    The compliance decision is never derived from this value.
    """
    if not (use_llm and groq_available()):
        return {"note": "", "source": "offline", "consulted": False}

    matched_ids = [m.get("policy_id") for m in (matched_policies or [])]
    note = _advise_with_groq(proposal_text, matched_ids)
    if note is None:                       # the LLM call failed -> stay silent
        return {"note": "", "source": "error", "consulted": True}
    return {"note": note, "source": "groq", "consulted": True}


def _advise_with_groq(proposal_text, matched_ids):
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage

        system = SystemMessage(content=(
            "You support a HUMAN compliance reviewer. An automated system has "
            "ALREADY decided this proposal and matched its policies - you must "
            "NOT repeat, restate or overturn that decision. Your only job is to "
            "point out a Responsible-AI policy angle the reviewer should "
            "personally double-check that is NOT already in the matched list: "
            "for example a policy the automated match may have under-weighted, "
            "or a nuance in the proposal. Be conservative - only speak up if it "
            "is genuinely worth a human's time, and never invent facts that are "
            "not in the proposal. Reply with EXACTLY one short sentence, or the "
            "single word " + _NONE + " if nothing stands out."))
        human = HumanMessage(content=(
            "Proposal:\n" + proposal_text + "\n\n"
            "Policies the automated system already matched: " +
            (", ".join(matched_ids) if matched_ids else "(none)") + "\n\n"
            "Full RAI policy catalogue:\n" + _catalogue_lines()))

        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
        answer = (llm.invoke([system, human]).content or "").strip()
        return _clean(answer)
    except Exception:
        return None


def _clean(answer):
    # Return "" (show nothing) for an empty answer or the NONE sentinel.
    if not answer:
        return ""
    first = answer.splitlines()[0].strip()
    if first.rstrip(".").strip().upper().startswith(_NONE):
        return ""
    return first[:300]        # defensive cap so a runaway answer can't dominate
