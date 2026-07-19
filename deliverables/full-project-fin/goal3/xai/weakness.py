"""
Automated weakness detection: adversarial evasion, blind spots, negation.

These checks turn the Goal-2 risks into repeatable tests:

-- Evasion (risk R6): if we hide or disguise the trigger words a real author
   might rephrase, does the detector still catch the violation?
-- Per-policy recall: does the agent catch some RAI policies much better than others?
-- Negation (risk R2): does the model still react when we flip the meaning with
   "not" / "no" / "without"? This weakness was found with XAI in Goal 3.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# --- perturbations ---
def mask_terms(text, terms):
    # Remove the trigger terms from the text (author drops the obvious words)
    out = str(text)
    # Remove the longest terms first, so "without user consent" goes before "consent"
    for term in sorted(terms, key=len, reverse=True):
        term_lower = term.lower()
        # Repeat until every case-insensitive occurrence of the term is gone
        while term_lower in out.lower():
            pos = out.lower().find(term_lower)
            out = out[:pos] + " " + out[pos + len(term):]
    # Turn multiple spaces into one and trim the ends
    out = " ".join(out.split())
    return out


def add_dots(word):
    # Put a dot between neighbouring letters/digits
    result = ""
    for i in range(len(word)):
        ch = word[i]
        result = result + ch
        # Add a dot only if this char and the next char are both letters/digits
        if i + 1 < len(word) and ch.isalnum() and word[i + 1].isalnum():
            result = result + "."
    return result


def obfuscate(text, terms):
    # Break the trigger words with dots ('social scoring' -> 's.o.c...'): readable, but no exact match
    out = str(text)
    for term in sorted(terms, key=len, reverse=True):
        term_lower = term.lower()
        while term_lower in out.lower():
            pos = out.lower().find(term_lower)
            original = out[pos:pos + len(term)]
            out = out[:pos] + add_dots(original) + out[pos + len(term):]
    return out


def evasion_recall(pipeline, texts, perturb, terms):
    # Red-Flag recall before vs. after a perturbation (perturb = mask_terms or obfuscate)
    texts = list(texts)
    if len(texts) == 0:
        return {"recall_before": float("nan"), "recall_after": float("nan"),
                "recall_drop": float("nan"), "n": 0}

    base = pipeline.predict(texts)

    changed = []
    for t in texts:
        changed.append(perturb(t, terms))
    perturbed = pipeline.predict(changed)

    recall_before = float(np.mean(base == 1))
    recall_after = float(np.mean(perturbed == 1))
    return {"recall_before": recall_before, "recall_after": recall_after,
            "recall_drop": recall_before - recall_after, "n": len(texts)}


# --- per-policy ---
def per_policy_recall(frame, code_col="red_flags", y_true="y_true", y_pred="y_pred"):
    # Recall per RAI policy code (e.g. RAI-01 ... RAI-09) on the hold-out set
    for c in [code_col, y_true, y_pred]:
        if c not in frame.columns:
            raise KeyError("Column '" + str(c) + "' missing")

    # For each policy code, collect a 1 (caught) or 0 (missed) for every violation
    per_policy = {}
    positives = frame[frame[y_true] == 1]
    for _, r in positives.iterrows():
        codes = str(r[code_col])
        if codes == "nan" or codes == "":
            continue
        caught = int(r[y_pred] == 1)
        for code in codes.split(","):
            code = code.strip()
            if code == "":
                continue
            if code not in per_policy:
                per_policy[code] = []
            per_policy[code].append(caught)

    rows = []
    for code in per_policy:
        results = per_policy[code]
        rows.append({"policy": code, "n": len(results),
                    "recall": float(np.mean(results))})

    out = pd.DataFrame(rows, columns=["policy", "n", "recall"])
    return out.sort_values("policy").set_index("policy")


def worst_policy(frame, code_col="red_flags", y_true="y_true", y_pred="y_pred"):
    # The RAI policy with the lowest recall (the biggest blind spot).
    pp = per_policy_recall(frame, code_col=code_col,
                           y_true=y_true, y_pred=y_pred)
    if pp.empty:
        raise ValueError("No policy rows found")
    return pp["recall"].idxmin()


# --- negation ---
# Words that flip the meaning of a sentence. sklearn's English stop-word list drops
# all of them, which makes "with consent" and "without consent" the same text
NEGATIONS = {"not", "no", "never", "nor",
             "none", "cannot", "without", "against"}

# Minimal pairs: left is a real violation, right is the correct counter-statement
# They differ only in the negation, so a working model must score them differently
NEGATION_PAIRS = [
    ("No human-in-the-loop is foreseen.",
     "A human-in-the-loop is foreseen."),
    ("The automated decision is not disclosed to data subjects.",
     "The automated decision is disclosed to data subjects."),
    ("Users are not informed about the use of AI.",
     "Users are informed about the use of AI."),
    ("Vulnerable groups are processed without safeguards.",
     "Vulnerable groups are processed with safeguards."),
]


def stopwords_without_negations():
    # The English stop-word list, but with the negation words kept
    return list(ENGLISH_STOP_WORDS - NEGATIONS)


def negation_gap(pipeline, pairs=None):
    # Red-Flag probability gap per pair; a gap near zero means the model is negation-blind
    if pairs is None:
        pairs = NEGATION_PAIRS

    rows = []
    for pair in pairs:
        violation = pair[0]
        counter = pair[1]
        p_violation = float(pipeline.predict_proba([violation])[0, 1])
        p_counter = float(pipeline.predict_proba([counter])[0, 1])
        rows.append({
            "violation": violation,
            "counter_statement": counter,
            "proba_violation": p_violation,
            "proba_counter": p_counter,
            "gap": p_violation - p_counter,
        })
    return pd.DataFrame(rows)


def negation_blind_count(pipeline, pairs=None, tol=0.01):
    # How many pairs the model treats as identical? 0 is good, len(pairs) is bad
    gaps = negation_gap(pipeline, pairs=pairs)
    blind = gaps["gap"].abs() < tol
    return int(blind.sum())
