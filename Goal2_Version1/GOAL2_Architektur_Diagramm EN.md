# GOAL 2: Architecture & Context

The Autonomous Compliance Sentinel · Module: Ethics & Responsible AI

This diagram shows **GOAL 2 (the detector)** in detail and - colour-coded -
the **next steps GOAL 3** and **GOAL 4** that build on it.

## Colour legend

| Colour | Area | Status |
|:---:|---|---|
| ⬜ Grey | **GOAL 1** - Data & policy catalogue | Foundation (existing) |
| 🟦 Blue | **GOAL 2** - Detector / model | **this directory** |
| 🟩 Green | **GOAL 3** - XAI & tests | next step |
| 🟧 Orange | **GOAL 4_Agent** - Agent | next step |

## Diagram

```mermaid
flowchart TB
    subgraph G1["GOAL 1 · Foundation"]
        direction TB
        DATA["proposals_1000_EN.csv<br/>1000 proposals · 1/3 Red Flag"]
        POL["RAI-01 … RAI-09<br/>Policy catalogue"]
    end

    subgraph G2["GOAL 2 · Detector  (this directory)"]
        direction TB
        D["data.py<br/>load + train/test split"]
        B["baselines.py<br/>rule / majority baseline"]
        M["models.py<br/>TF-IDF + LogReg / Naive Bayes"]
        MET["metrics.py<br/>Recall · Fairness · Risk"]
        PIPE["pipeline.py<br/>build_artifacts()"]
        ART["artifacts/<br/>Models · Split · Predictions"]
        TEST2["tests/<br/>22 tests · 96% coverage"]

        D --> M
        B --> PIPE
        M --> PIPE
        MET --> PIPE
        PIPE --> ART
        TEST2 -.tests.-> PIPE
    end

    subgraph G3["GOAL 3 · XAI & Tests  (next step)"]
        direction TB
        XAI["xai/<br/>Cross-validation · Explainability (XAI)<br/>Weakness tests · 97% coverage"]
    end

    subgraph G4["GOAL 4_Agent · Agent  (next step)"]
        direction TB
        AG["agent/<br/>perceive → decide → explain<br/>→ suggest fix → mock writeback"]
    end

    %% Dependencies between the goals
    DATA --> D
    ART --> XAI
    XAI --> AG
    POL -. RAI catalogue .-> AG

    %% ----- Area colours -----
    style G1 fill:#ECEFF1,stroke:#90A4AE,stroke-width:1px,color:#263238
    style G2 fill:#E3F2FD,stroke:#1E88E5,stroke-width:3px,color:#0D47A1
    style G3 fill:#E8F5E9,stroke:#43A047,stroke-width:2px,color:#1B5E20
    style G4 fill:#FFF3E0,stroke:#FB8C00,stroke-width:2px,color:#E65100

    %% ----- Building-block colours -----
    classDef g1 fill:#F5F7F8,stroke:#B0BEC5,color:#37474F
    classDef g2 fill:#BBDEFB,stroke:#1565C0,color:#0D47A1
    classDef g3 fill:#C8E6C9,stroke:#2E7D32,color:#1B5E20
    classDef g4 fill:#FFE0B2,stroke:#EF6C00,color:#E65100

    class DATA,POL g1
    class D,B,M,MET,PIPE,ART,TEST2 g2
    class XAI g3
    class AG g4
```

## How to read the diagram

1. **GOAL 1 (grey)** provides the data (`proposals_1000_EN.csv`) and the policy catalogue (RAI-01…09).
2. **GOAL 2 (blue)** is the core of this directory: load data -> train models ->
   evaluate -> save as **artifacts**. `build_artifacts()` is the single entry point,
   `tests/` ensures quality.
3. **GOAL 3 (green)** loads the saved GOAL 2 artifacts and adds explainability (XAI)
   and weakness tests - **without retraining**.
4. **GOAL 4_Agent (orange)** builds on GOAL 3 and connects everything into the runnable agent;
   it uses the RAI catalogue from GOAL 1 directly for the fix suggestions (dashed line).

**Chain:** `GOAL 1 -> GOAL 2 -> GOAL 3 -> GOAL 4_Agent`.
GOAL 2 is the model foundation on which GOAL 3 and the agent build.
