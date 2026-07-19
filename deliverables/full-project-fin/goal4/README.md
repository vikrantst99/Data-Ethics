# Goal 4 (extended) — The Autonomous Compliance Sentinel Agent

Everything from the *basic* agent (Sections 1 + 2 of the integration note) **plus
the optional Section-3 add-ons**. Same LangGraph flow, same offline fallbacks.

## Section-3 options included

| Option | Where | Notes |
|---|---|---|
| **ipywidgets UI** | `agent/ui.py` + notebook | text box + Assess button; `ui_available()` guards a graceful fallback |
| **Streamlit web UI** | `streamlit_app.py`, `charts.py` | `streamlit run streamlit_app.py` → PDF upload, findings, charts, APPROVE/REJECT |
| **Real writeback** | `agent/writeback.py`, `outbox/` | Confluence comment + Jira ticket + `agent_log.jsonl` |
| **Batch processing** | `agent/batch.py`, `run_batch.py`, `inbox/` | one `*.txt` per proposal → summary + outbox files |
| **Weakness pre-gate** | `agent/certify.py` | reuses Goal-3 `xai`: negation robustness + weakest-policy recall |
| **Configurable severity** | `config/policies.csv` | overrides the built-in High/Medium map |
| **Borderline LLM routing** | `agent/graph.py` (`llm_band`) | LLM only inside the uncertain band (default 0.35–0.65) |
| **Open-ended verification** | `agent/graph.py` + `agent/llm.py` (`is_violation`) | the agent may clear a false alarm instead of assuming a violation |

## Run it

```bash
pip install -r requirements.txt
export GROQ_API_KEY=...          # optional; offline fallback otherwise

python run_agent.py              # one proposal + weakness pre-gate + writeback
python run_batch.py              # process the whole inbox/ folder
streamlit run streamlit_app.py   # Streamlit web UI
python -m pytest tests/ -q       # tests
```

## The weakness pre-gate

`certify()` turns Goal-3 metrics into pass/fail gates. On the Version-2 model it
reports the known **negation-blindness** (`negation_blind_pairs = 4`), so the gate
returns `passed: False` — that is the pre-gate correctly surfacing a documented
model weakness before the agent ships, not a bug. Adjust the thresholds
(`max_blind`, `min_worst_recall`) to your policy.

## Offline behaviour

Without `langchain-groq` the verdict is phrased by a deterministic template;
without the embedding model the RAG falls back to TF-IDF keyword search; without
`ipywidgets` use `run_once()` / Streamlit directly. Nothing hard-fails.
