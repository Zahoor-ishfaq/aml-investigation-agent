# AML Investigation Agent

An AI-powered anti-money-laundering investigation system that combines deterministic rule-based detection with an LLM-driven ReAct investigation loop. The system ingests synthetic financial data, flags suspicious activity through configurable rules, runs an autonomous agent to investigate and disposition each alert, and enforces compliance guardrails before any decision reaches a human review queue.

---

![Tech Stack](https://img.shields.io/badge/Tech_Stack-0A0A0A?style=for-the-badge)

### Core Infrastructure
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white)

### AI / ML
![Anthropic](https://img.shields.io/badge/Claude_API-191919?style=for-the-badge&logo=anthropic&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F61?style=for-the-badge&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-FF9900?style=for-the-badge&logoColor=white)

### Data & Evaluation
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![AMLSim](https://img.shields.io/badge/AMLSim-054ADA?style=for-the-badge&logoColor=white)

### Dashboard & Monitoring
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

### Configuration
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)

---

![Why This Exists](https://img.shields.io/badge/Why_This_Exists-1a1a2e?style=for-the-badge)

Traditional AML compliance systems generate thousands of alerts daily, the vast majority of which are false positives. Human analysts spend most of their time on repetitive evidence-gathering rather than judgment. This project explores whether a tool-calling LLM agent can handle the investigative legwork — pulling transaction histories, examining customer profiles, reviewing linked accounts — while deterministic guardrails enforce the regulatory constraints that an LLM alone cannot be trusted with.

The architecture reflects a deliberate separation of concerns: the agent reasons, the guardrails constrain, and humans decide. The agent is never permitted to autonomously close a high-severity alert or file a SAR. Every action is logged to an immutable audit store for regulatory reconstruction.

---

![Architecture](https://img.shields.io/badge/Architecture-1a1a2e?style=for-the-badge)

```
Rule Engine → Alert Queue → Investigation Agent (ReAct loop)
    → Guardrail Layer (deterministic, non-LLM)
    → Decision + Case File → Human Review Queue

Every step logged to immutable audit store.
Pseudonymization wraps all data before it reaches the LLM.
```

The system is built in ten phases, each independently testable:

| Phase | Component | Purpose |
|:-----:|-----------|---------|
| 1 | Postgres schema + synthetic data | AMLSim-generated transactions, accounts, alerts |
| 2 | Rule engine | Configurable detection rules (structuring, rapid movement, fan-in/fan-out) |
| 3 | Pseudonymization | Token-based masking so the LLM never sees real identifiers |
| 4 | Tool layer | Four investigation tools the agent can call |
| 5 | RAG knowledge base | AML typology retrieval via ChromaDB + sentence-transformers |
| 6 | Guardrail layer | Deterministic policy enforcement |
| 7 | Agent reasoning core | ReAct-style tool-calling loop with 15-iteration cap |
| 8 | Audit log store | Immutable append-only log with delete/update triggers blocked at DB level |
| 9 | Eval harness | Stratified ground-truth evaluation with checkpoint/resume and offline scoring |
| 10 | Monitoring dashboard | Streamlit dashboard (5 pages) |

---

![Key Design Decisions](https://img.shields.io/badge/Key_Design_Decisions-1a1a2e?style=for-the-badge)

**Safe defaults over autonomous efficiency.** Any failure — parse error, guardrail block, iteration cap — degrades to escalation, never to silent closure. A false positive reaching a human reviewer is recoverable; a false negative is not.

**Pseudonymization at the boundary.** All account and transaction identifiers are tokenized before entering the LLM context. The agent works with opaque tokens (`CEXT_xxx`, `AEXT_xxx`, `TEXT_xxx`) and the guardrail layer blocks narratives containing suspected PII leakage.

**Deterministic guardrails, not LLM-based.** The guardrail layer is pure Python with no model calls. High-severity dismissals are blocked, SAR filings require human review, and PII/LLM-artifact patterns are caught via regex. Fully testable and auditable without model variability.

**Eval before trust.** The eval harness runs the agent against AMLSim's ground-truth labels and scores recall, precision, and F1 before any production deployment consideration. Results are persisted as JSON artifacts for offline re-scoring without re-invoking the LLM.

---

![Eval Results](https://img.shields.io/badge/Eval_Results-1a1a2e?style=for-the-badge)

Evaluated on a 30-alert stratified sample (20 laundering, 10 non-laundering) from AMLSim synthetic data:

| Metric | Value |
|--------|:-----:|
| **Recall** | `1.000` |
| **Precision** | `0.667` |
| **F1** | `0.800` |
| Parse OK rate | `100%` |
| Completion rate | `100%` |
| Error rate | `0%` |

> Recall is the priority metric for AML — no laundering case was missed. The precision gap (over-escalation of non-laundering alerts) is a known limitation of the synthetic data: AMLSim's non-laundering accounts exhibit transaction patterns similar to laundering activity, limiting the signal available to the agent.

---

![Project Structure](https://img.shields.io/badge/Project_Structure-1a1a2e?style=for-the-badge)

```
├── src/aml_agent/
│   ├── config.py                # pydantic-settings configuration
│   ├── agent/
│   │   ├── claude_client.py     # Anthropic SDK adapter (primary)
│   │   ├── groq_client.py       # Groq SDK adapter
│   │   ├── gemini_client.py     # Google genai SDK adapter
│   │   ├── cerebras_client.py   # Cerebras adapter
│   │   ├── session.py           # ReAct loop (tool dispatch, token tracking)
│   │   ├── orchestrator.py      # End-to-end investigation wiring
│   │   ├── prompts.py           # System prompt + initial message builder
│   │   └── case_file.py         # Agent JSON decision parser
│   ├── db/
│   │   ├── models.py            # SQLAlchemy ORM models
│   │   └── session.py           # DB session management
│   ├── rule_engine/
│   │   ├── runner.py            # Rule execution loop
│   │   └── rules/               # structuring, rapid_movement, fan detection
│   ├── pseudonymization/
│   │   ├── tokenizer.py         # Token-based identifier masking
│   │   └── depseudonymize.py    # Reverse mapping for human review
│   ├── tools/
│   │   ├── dispatch.py          # Tool name → handler routing
│   │   ├── transaction_history.py
│   │   ├── customer_profile.py
│   │   ├── linked_accounts.py
│   │   └── alert_history.py
│   ├── guardrails/
│   │   ├── engine.py            # Aggregate evaluator (BLOCK > REVIEW > PASS)
│   │   ├── policies.py          # Lifecycle, severity, SAR guardrails
│   │   └── validation.py        # PII leak + LLM artifact detection
│   ├── rag/
│   │   ├── chunker.py           # PDF → chunks
│   │   ├── embedder.py          # sentence-transformers encoding
│   │   ├── store.py             # ChromaDB persistence
│   │   └── retriever.py         # Similarity search
│   ├── eval/
│   │   └── labels.py            # Ground-truth loading, stratified sampling
│   └── audit/
│       ├── writer.py            # Immutable audit_log append
│       └── query.py             # Audit trail queries
├── scripts/
│   ├── eval_agent.py            # Agent eval runner (checkpoint/resume)
│   ├── score_eval.py            # Offline scorer (JSON → metrics)
│   ├── run_rules.py             # Rule engine batch execution
│   ├── run_ingestion.py         # AMLSim data ingestion
│   ├── build_kb.py              # RAG knowledge base builder
│   └── investigate_one.py       # Single-alert investigation (debug)
├── streamlit_app/
│   ├── app.py                   # Dashboard entry point + navigation
│   ├── db.py                    # Read-only DB connection
│   └── pages/
│       ├── alert_overview.py    # Alert status + severity charts
│       ├── rule_engine.py       # Rule code breakdown + stats
│       ├── agent_eval.py        # P/R/F1, confusion matrix, cost
│       ├── guardrails.py        # Pass/block/review rates
│       └── audit_trail.py       # Searchable audit log viewer
├── knowledge_base/              # ChromaDB + FATF/Wolfsberg PDFs
├── migrations/                  # Alembic schema migrations
├── eval_results/                # Persisted eval artifacts
├── tests/                       # Rule engine unit tests
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

![Setup](https://img.shields.io/badge/Setup-1a1a2e?style=for-the-badge)

### Prerequisites

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Required-2496ED?style=flat-square&logo=docker&logoColor=white)

### Installation

```bash
git clone https://github.com/Zahoor-ishfaq/aml-investigation-agent.git
cd aml-investigation-agent

# Start infrastructure
docker compose up -d

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

```env
# Database
POSTGRES_USER=aml_app
POSTGRES_PASSWORD=your_password
POSTGRES_DB=aml_investigation
POSTGRES_HOST=localhost
POSTGRES_PORT=5434

# LLM Provider
CLAUDE_API_KEY=sk-ant-...
CLAUDE_MODEL_NAME=claude-haiku-4-5-20251001
```

### Run Evaluation

```bash
# Run agent evaluation (checkpoint/resume on rate limits)
python scripts/eval_agent.py --fresh --limit 30

# Score results offline (no API calls)
python scripts/score_eval.py
```

### Run Dashboard

```bash
streamlit run streamlit_app/app.py
```

---

![Limitations](https://img.shields.io/badge/Limitations-1a1a2e?style=for-the-badge)

- **Precision on synthetic data.** AMLSim generates non-laundering transactions with patterns similar to laundering activity. Real-world transaction data would likely improve precision.

- **Rule engine thresholds.** Detection thresholds are set conservatively. Tuning requires a larger labeled dataset than the current 30-alert eval sample provides.

- **Single-model evaluation.** Final metrics reflect Claude Haiku 4.5. Other models tested during development (Llama 3.3 70B via Groq, GPT-OSS 120B, Gemini 2.5 Flash) showed varying tool-calling reliability but were not formally benchmarked.

---

![References](https://img.shields.io/badge/References-1a1a2e?style=for-the-badge)

- **FATF Recommendations** — R20 (Reporting of Suspicious Transactions): basis for guardrail policies on dismissal blocking and SAR review requirements
- **Wolfsberg Group AML Principles** — informed the safe-default escalation policy
- **Yao et al.** — "ReAct: Synergizing Reasoning and Acting in Language Models" ([arXiv:2210.03629](https://arxiv.org/abs/2210.03629)): architecture of the agent reasoning loop
- **AMLSim** — [github.com/IBM/AMLSim](https://github.com/IBM/AMLSim): synthetic transaction data generation