"""
Turn the collected evidence into a grounded compliance verdict.

Extended version: the verdict is *open-ended* - it also carries an
``is_violation`` flag. Instead of assuming that everything past the gate is a
violation (the Version-1 agent hard-coded y_true=1), the agent may conclude the
proposal is actually fine after looking at the evidence. With Groq the LLM
decides; offline a conservative heuristic does.

If a Groq API key is set (GROQ_API_KEY) and langchain-groq is installed, the LLM
phrases reason and fix. Otherwise a deterministic template does.
"""

import os

from .policies import get_severity


def groq_available():
    if not os.environ.get("GROQ_API_KEY"):
        return False
    try:
        import langchain_groq  # noqa: F401
    except Exception:
        return False
    return True


def _redflag_terms(trigger_words):
    return [word for word, score in trigger_words if score > 0]


def compose_verdict(proposal_text, trigger_words, retrieved, proba, verify=True,
                    use_llm=True):
    # use_llm=False forces the offline template even when a Groq key is set
    # (used for confident cases outside the borderline band).
    top = retrieved[0] if retrieved else {"policy_id": "RAI-09", "score": 0.0}
    policy_id = top["policy_id"]
    severity = get_severity(policy_id) or "Unknown"

    if use_llm and groq_available():
        verdict = _compose_with_groq(
            proposal_text, trigger_words, retrieved, policy_id, severity, verify)
        if verdict is not None:
            verdict["provenance"] = provenance_note(verdict["source"])
            return verdict

    verdict = _compose_offline(trigger_words, retrieved, policy_id, severity,
                               proba, verify)
    verdict["provenance"] = provenance_note(verdict["source"])
    return verdict


def per_policy_finding(policy_id, severity, trigger_words):
    # A short offline reason + fix for one extra matched policy of a multi-violation document
    terms = _redflag_terms(trigger_words)[:4]
    if terms:
        reason = ("The proposal is missing the safeguards for " + policy_id +
                  "; the wording around '" + ", ".join(terms) +
                  "' drove the Red-Flag decision.")
    else:
        reason = "The proposal is missing the safeguards for " + policy_id + "."
    fix = ("Add the control required by " + policy_id +
           " and state it explicitly in the proposal.")
    return {"policy_id": policy_id, "severity": severity, "reason": reason,
            "recommended_fix": fix, "source": "offline"}


def provenance_note(source):
    # Transparency line: the model decides, the text only phrases the decision.
    base = "The Red-Flag decision comes from the trained TF-IDF model. "
    if source == "groq":
        return (base + "The reasons and fixes are written by an LLM (primary "
                "finding) or a fixed template (additional policies) and may be "
                "inaccurate - please verify before acting.")
    return (base + "The reasons and fixes are generated from a fixed template "
            "(no LLM).")


def _compose_offline(trigger_words, retrieved, policy_id, severity, proba, verify):
    terms = _redflag_terms(trigger_words)[:4]
    # With verification on, weak evidence (no Red-Flag words and a low match score) counts as a false alarm
    top_score = retrieved[0].get("score", 0.0) if retrieved else 0.0
    is_violation = True
    if verify and not terms and top_score < 0.05:
        is_violation = False

    if not is_violation:
        return {
            "policy_id": policy_id, "severity": severity,
            "is_violation": False,
            "reason": "On review the flagged wording is not an actual RAI "
                      "violation; the signal is weak and unspecific.",
            "recommended_fix": "No fix required - clear as a false alarm.",
            "source": "offline",
        }

    if terms:
        joined = ", ".join(terms)
        reason = ("The proposal is missing the safeguards for " + policy_id +
                  "; the wording around '" + joined +
                  "' drove the Red-Flag decision.")
    else:
        reason = ("The proposal is missing the safeguards for " + policy_id +
                  " (probability " + str(round(proba, 3)) + ").")
    fix = ("Add the control required by " + policy_id +
           " and state it explicitly in the proposal.")
    return {
        "policy_id": policy_id, "severity": severity, "is_violation": True,
        "reason": reason, "recommended_fix": fix, "source": "offline",
    }


def _compose_with_groq(proposal_text, trigger_words, retrieved, policy_id,
                       severity, verify):
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage

        terms = ", ".join(_redflag_terms(trigger_words)[:8])
        policy_block = "\n\n".join(r["text"] for r in retrieved)
        system = SystemMessage(content=(
            "You are a compliance assessor. Use ONLY the evidence given. "
            "First judge whether this really violates the policy. "
            "Answer in exactly three lines:\n"
            "VIOLATION: <yes or no>\n"
            "REASON: <one sentence, grounded in the trigger words>\n"
            "FIX: <one sentence, grounded in the policy text>"))
        human = HumanMessage(content=(
            "Proposal:\n" + proposal_text + "\n\n"
            "Candidate policy: " + policy_id + " (" + severity + ")\n"
            "Trigger words: " + terms + "\n\n"
            "Policy text:\n" + policy_block))

        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
        answer = llm.invoke([system, human]).content

        is_violation, reason, fix = _parse_answer(answer, verify)
        return {
            "policy_id": policy_id, "severity": severity,
            "is_violation": is_violation, "reason": reason,
            "recommended_fix": fix, "source": "groq",
        }
    except Exception:
        return None


def _parse_answer(answer, verify):
    is_violation, reason, fix = True, "", ""
    for line in answer.splitlines():
        low = line.strip().lower()
        if low.startswith("violation:"):
            value = line.split(":", 1)[1].strip().lower()
            if verify:
                is_violation = value.startswith("y")
        elif low.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
        elif low.startswith("fix:"):
            fix = line.split(":", 1)[1].strip()
    if not reason:
        reason = answer.strip()
    if not fix:
        fix = "Add the missing control and state it explicitly."
    return is_violation, reason, fix
