# THE AUTONOMOUS COMPLIANCE SENTINEL
Group 2: Singh, Vikrant and Müller, Kay
## Report

The 'report.pdf' contains our complete project documentation.

## Version 1 and Version 2: What Each Contains

This project was built in two versions. Version 1 was the first attempt. Version 2 is the
final project. They use the same idea but a different dataset.

---

## Version 1 (folder version_1, old dataset)

Version 1 uses the **old dataset**. It has three parts:

- **Goal 1: the data.** It creates 1000 project proposals. One third of them contain
  ethical red flags.
- **Goal 2: the baseline model.** A TF-IDF + Logistic Regression model. It learns to tell
  red flags from compliant proposals.
- **Goal 3: explainability and weakness tests.** This is where the problem was found.

**The problem with Version 1:** the old dataset had a hidden shortcut. Many red-flag texts
contained the word "met". The model learned to look for this word instead of the real meaning.

Version 1 has **no agent**. It is the first attempt that showed the shortcut problem.

---

## Version 2 (final project, folders `goal1` to `goal4`)

Version 2 uses a **new, clean dataset**. The hidden shortcut was removed. It has four parts:

- **Goal 1: the new data.** Still 1000 proposals, still one third red flags. But the "met"
  marker is gone (0% now). The model must learn the real breach vocabulary.
- **Goal 2: the baseline model, retrained.** The scores are honest now. Precision is lower,
  but the model no longer cheats.
- **Goal 3: explainability and weakness tests again.** It confirms the shortcut is gone. It
  also finds remaining weaknesses. For example, the model is blind to negation (it does not
  see the difference between "with consent" and "without consent").
- **Goal 4: the agent (this is new).** This is the final product, the *Autonomous Compliance
  Sentinel*. It reads a proposal, uses the model to flag it, retrieves the matching policy
  with RAG, writes a reason and a fix, asks a human for high-severity cases, and writes the
  result to Confluence/Jira/audit files. It also has a Streamlit web app and a PDF upload.

---

## The main difference (in one line)

**Version 1** = old dataset with a hidden shortcut, so the model could cheat.
**Version 2** = clean dataset with no shortcut, plus the finished agent.

---

## Comparison

| | Version 1 (`version_1`) | Version 2 (final) |
|---|---|---|
| Dataset | old, has the "met" marker | new, marker-free |
| Goal 1 (data) | yes | yes |
| Goal 2 (model) | yes | yes (retrained) |
| Goal 3 (XAI + weakness tests) | yes | yes |
| Goal 4 (agent) | no | yes |
| Model scores | high | honest |
| Status | first attempt | final project |
