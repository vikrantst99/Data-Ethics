# Compliance Sentinel - streamlit_run

- **Prediction:** Red Flag
- **Probability:** 0.5127
- **Policies matched:** RAI-02 (High), RAI-05 (High), RAI-01 (High), RAI-07 (Medium)
- **Governing severity:** High
- **Decision:** High severity (RAI-02 (+RAI-05, RAI-01, RAI-07)): human reviewer approved remediation.

## Findings
### RAI-02 (High)
- **Reason:** The proposal violates RAI-02 (Transparency) because users are not informed about the use of AI.
- **Recommended fix:** Inform users that the chatbot is an AI and provide transparency about its automated decision-making process.
### RAI-05 (High)
- **Reason:** The proposal is missing the safeguards for RAI-05; the wording around 'users, automatically, ai' drove the Red-Flag decision.
- **Recommended fix:** Add the control required by RAI-05 and state it explicitly in the proposal.
### RAI-01 (High)
- **Reason:** The proposal is missing the safeguards for RAI-01; the wording around 'users, automatically, ai' drove the Red-Flag decision.
- **Recommended fix:** Add the control required by RAI-01 and state it explicitly in the proposal.
### RAI-07 (Medium)
- **Reason:** The proposal is missing the safeguards for RAI-07; the wording around 'users, automatically, ai' drove the Red-Flag decision.
- **Recommended fix:** Add the control required by RAI-07 and state it explicitly in the proposal.

> The Red-Flag decision comes from the trained TF-IDF model; the reason and fix below were written by an LLM and may be inaccurate - please verify before acting.
