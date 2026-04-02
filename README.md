# Agentic Defect Intelligence Pipeline (ADIP)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-1C3C3C?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.9%2B-DC382D?logo=qdrant&logoColor=white)](https://qdrant.tech)
[![Kafka](https://img.shields.io/badge/Kafka-2.0%2B-231F20?logo=apachekafka&logoColor=white)](https://kafka.apache.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-48%2F48_passing-brightgreen)](adip/tests/)

**A continuously running multi-agent AI pipeline that ingests production defect streams, clusters failure patterns via HDBSCAN-over-UMAP, scores file-level risk with a five-feature Bayesian-adaptive formula, and generates actionable test directives — all orchestrated as a LangGraph state machine with human-in-the-loop gating.**

---

## 1. Architecture

The ADIP architecture is organized into four distinct layers, from ingestion to action, ensuring a robust and scalable pipeline for defect intelligence.

![ADIP Architecture Diagram](https://private-us-east-1.manuscdn.com/sessionFile/NsS5dnmuZ8XqsBXnh2lki0/sandbox/tAUP7e8nDUCyr4XbsrOQuE-images_1775118219923_na1fn_L2hvbWUvdWJ1bnR1L2FkaXBfYXJjaGl0ZWN0dXJl.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvTnNTNWRubXVaOFhxc0JYbmgybGtpMC9zYW5kYm94L3RBVVA3ZThuRFVDeXI0WGJzck9RdUUtaW1hZ2VzXzE3NzUxMTgyMTk5MjNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyRmthWEJmWVhKamFHbDBaV04wZFhKbC5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=NcOAqqWrjKx~ERXVRv2PG4x5OVqgNAa4ovtScKHYujEXa9THc3KGqeAqQ0jS7uqJlLZggBCKrgd7J88dTlc5pcjBv2ewOcSGJtwLBGpT6tyHXHiznBkTnr3jOQqstpNWsXI318IWQmXBzUmqB9~Fcwbmx-zqVvR5NVuVls8THUo-XJnr7sAapjHR88XM4omhvXgpcyRGo2nN2~BhTL5e8gMerSBMAn9BLhl24-2-vR6RhkoYKy4WPOVPzB5JYrGWt6Dxzw1-m6-jRrGukR2wE7qEs6EijXv-Dto5MGpVqvBmw7Gao0LqrNniXBkveewjlSMpbZXtm8FYFWRctKiqwA__)

---

## 2. Three Operation Modes

ADIP supports three distinct operation modes to balance latency and depth of analysis based on the trigger source.

| Mode | Trigger | Latency Target | Clustering | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Streaming** | Kafka event / Sentry alert | < 60 seconds | Skip (use existing clusters) | Real-time production error triage |
| **Batch** | Scheduler (every 6 hours) | ~5 minutes | Full re-cluster + re-label | Periodic risk posture reassessment |
| **On-Demand** | PR webhook / API call | < 3 minutes | Conditional (if >= 5 new events) | Pre-merge risk gate in CI/CD |

---

## 3. Agent Roster

The pipeline is orchestrated by seven specialized agents, each with a clear decision boundary and specific input/output requirements.

| # | Agent | Input | Output | Decision Boundary |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Ingestion Parser** | Kafka streams, Jira tickets, Sentry/Datadog alerts | `List[DefectEvent]` | Auto-detects source type; structure-aware parsing per source |
| 2 | **RAG Indexer** | `List[DefectEvent]` | `IndexingResult` (count + errors) | Content-hash dedup — never indexes duplicate events |
| 3 | **Defect Clusterer** | All embeddings (last 30d) from vector store | `List[ClusterResult]` | HDBSCAN decides cluster count; LLM constrained to 11-category taxonomy |
| 4 | **Risk Scorer** | `List[DefectEvent]` + `List[ClusterResult]` | `List[FileRiskScore]` | Five-feature weighted formula with Bayesian-updated weights |
| 5 | **Report Generator** | `List[FileRiskScore]` + `List[ClusterResult]` | `RiskReport` | Rule engine (deterministic) decides HOLD/CONDITIONAL/PROCEED *before* LLM call |
| 6 | **Test Feedback** | `RiskReport` + high-risk files | `List[TestGenerationDirective]` | RAG retrieval of similar past defects informs test scenarios |
| 7 | **Alert Dispatcher** | `RiskReport` | `List[NotificationResult]` | Only dispatches on HOLD/CONDITIONAL; PROCEED = no alerts |

---

## 4. Installation

### Prerequisites

- Python 3.11+
- (Optional) Qdrant, PostgreSQL, Kafka, Redis — all fall back gracefully to local/in-memory alternatives.

### Setup

```bash
git clone <repository-url>
cd "Agentic Defect Intelligence Pipeline"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r adip/requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root:

```env
# Required — LLM
ADIP_OPENAI_API_KEY=voc-your-key-here
ADIP_OPENAI_BASE_URL=https://openai.vocareum.com/v1

# Optional — External Services
ADIP_KAFKA_BOOTSTRAP_SERVERS=localhost:9092
ADIP_QDRANT_HOST=localhost
ADIP_QDRANT_PORT=6333
ADIP_DATABASE_URL=postgresql://localhost/adip
ADIP_REDIS_URL=redis://localhost:6379

# Optional — Integrations
ADIP_JIRA_BASE_URL=https://your-org.atlassian.net
ADIP_JIRA_API_TOKEN=your-jira-token
ADIP_SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Optional — Observability
ADIP_LANGSMITH_API_KEY=ls-your-key
ADIP_LANGSMITH_PROJECT=adip-pipeline
```

---

## 5. Usage

- **Mock Mode**: `python -m adip.main --mock` - Runs the full pipeline with synthetic data and in-memory services.
- **Streaming Mode**: `python -m adip.main --mode streaming` - Consumes real-time events from Kafka.
- **Batch Mode**: `python -m adip.main --mode batch` - Periodic full re-clustering and risk reassessment.
- **On-Demand Mode**: `python -m adip.main --mode on_demand` - Single-shot execution for CI/CD gates.
- **API Server**: `python -m adip.main --api --port 8000` - Starts a FastAPI server for remote triggers and status checks.

---

## 6. Pipeline Walkthrough

A real scenario demonstrating streaming mode from error to actionable test:

**T+0s** — Sentry captures a `NullReferenceError` in `src/auth/token.py` during token refresh on user logout.

**T+22s** — ADIP processes the trace, retrieves similar past defects from Qdrant, calculates a risk score of 0.88 (HOLD), and generates a test directive for a new regression test covering the logout/refresh race condition.

---

## 7. RAG Pipeline

The RAG pipeline uses a hybrid approach combining dense and sparse embeddings with cross-encoder reranking for maximum retrieval accuracy.

![RAG Pipeline Diagram](https://private-us-east-1.manuscdn.com/sessionFile/NsS5dnmuZ8XqsBXnh2lki0/sandbox/tAUP7e8nDUCyr4XbsrOQuE-images_1775118219923_na1fn_L2hvbWUvdWJ1bnR1L3JhZ19waXBlbGluZQ.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvTnNTNWRubXVaOFhxc0JYbmgybGtpMC9zYW5kYm94L3RBVVA3ZThuRFVDeXI0WGJzck9RdUUtaW1hZ2VzXzE3NzUxMTgyMTk5MjNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmhaMTl3YVhCbGJHbHVaUS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=R-gAo1UeX-KbNllX-NN07n2-DQhld4OFt-8SZ4Zhbpbq2GGYZEiHocxiS9yn8sf1irQ6bt42W4pTbeNUG6jciG2G0vR6-zRu6wmeOpb8uc4m0ulaHEhZZ8YxWMug2i1XYKF1g1g1SsJSR9-xdg3LeCoZ7yaC4ScfFS9d8cLsjAtudh3bLH2awcdMNaX4XZz2za0vBEhY90Ne7LmK1UHFWdv9lDXWG4jmjVLR49bsyXB62-RPcAmUqGXz3REv11asj3npS~AXK8eKiDsTncaT2SWU7GTARQv28qtW1caSHiRf-~AO1jkdWctOiir15yY7keA7sjQhh-TeYQ9DZjePsg__)

---

## 8. Structure-Aware Chunking

ADIP implements specialized chunking strategies for different data sources:

- **Stack Traces**: Atomic chunking to preserve the full call stack context.
- **Jira Bodies**: 800-token recursive character splitting to maintain semantic coherence.
- **CI Logs**: Boundary-aware chunking based on `PASSED`/`FAILED` markers.

---

## 9. HDBSCAN-over-UMAP Clustering

ADIP uses UMAP for dimensionality reduction (768d → 50d) followed by HDBSCAN for density-based clustering.

| Factor | HDBSCAN | K-Means | DBSCAN |
| :--- | :--- | :--- | :--- |
| **Cluster Count** | Discovered automatically | Must specify K | Discovered automatically |
| **Density** | Handles variable density | Assumes uniform density | Assumes uniform density |
| **Noise** | Robust (assigns to -1) | Forces into clusters | Robust |
| **Decision** | **Selected** | Rejected (K unknown) | Rejected (fixed epsilon) |

---

## 10. Five-Feature Risk Formula

The risk score is calculated using a weighted formula across five normalized features:

$$risk = freq \times 0.35 + churn \times 0.25 + gap \times 0.20 + sev \times 0.15 + decay \times 0.05$$

- **Frequency (0.35)**: How often the defect occurs.
- **Churn (0.25)**: Code change frequency in the affected file.
- **Gap (0.20)**: Time since the last test execution.
- **Severity (0.15)**: Impact level of the defect.
- **Decay (0.05)**: Time since the defect was first observed.

---

## 11. Rule Engine

The rule engine provides deterministic gating before any LLM-based reporting.

| Rule | Condition | Action |
| :--- | :--- | :--- |
| **HOLD** | Risk Score > 0.85 | Interrupt for human review; block PR |
| **CONDITIONAL** | 0.60 < Risk Score <= 0.85 | Proceed with warnings; require test directive |
| **PROCEED** | Risk Score <= 0.60 | Log and continue; no alerts |

---

## 12. Bayesian Weight Update

ADIP learns from test outcomes to refine its risk scoring weights over time.

- **Learning Rate**: Controls how quickly weights adapt to new feedback.
- **Normalization**: Ensures weights always sum to 1.0.
- **Persistence**: Updated weights are stored in PostgreSQL/SQLite for future runs.

---

## 13. Feedback Loop

The feedback loop connects ADIP with the Automated Quality Assurance Framework (AQAF) to close the loop on defect intelligence.

![Feedback Loop Diagram](https://private-us-east-1.manuscdn.com/sessionFile/NsS5dnmuZ8XqsBXnh2lki0/sandbox/tAUP7e8nDUCyr4XbsrOQuE-images_1775118219923_na1fn_L2hvbWUvdWJ1bnR1L2ZlZWRiYWNrX2xvb3A.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvTnNTNWRubXVaOFhxc0JYbmgybGtpMC9zYW5kYm94L3RBVVA3ZThuRFVDeXI0WGJzck9RdUUtaW1hZ2VzXzE3NzUxMTgyMTk5MjNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyWmxaV1JpWVdOclgyeHZiM0EucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=Qp37Qpaxw~s2NXw0anoS1paGSHDucMGdAXI6pxESmSCJRG6aJX4tBJWZ5hcKs3o2b~ikv50F2g5wc1gxUcbS5p-Sk1Se8bCUqlqqMOa8w3Q1DdFHjZw15sKqxA4EIrLEVsdIeIwdWin~DpO5z-5gLOIh7jovlWHXW85WKvgWVYHjCDlmzHy~CHPE~4iK4NRMjZo34fw7VJJlkgG1zSf0-4aG34gRa2icuJGcoX~RpTNSE9y36YQ-uBjHpL5D4NjYhr5IMFKrHjxp9WCtbx~uOdIIy68jSRGuc6eOoHgRzOStMuMg2DA5TSy~L60HfKCydO-kRyKA2LtEwuUEi-Z0vw__)

---

## 14. Three-Tier Freshness Monitoring

ADIP monitors the staleness of its data sources to ensure high-fidelity intelligence.

- **Tier 1: Kafka Stream Age**: Threshold 300s (5 minutes).
- **Tier 2: Jira Poll Recency**: Threshold 900s (15 minutes).
- **Tier 3: Vector Store Upsert Lag**: Threshold 120s (2 minutes).

---

## 15. Observability

### Prometheus Metrics
- `adip_events_ingested_total`: Counter for ingested events by source.
- `adip_reports_generated_total`: Counter for reports by recommendation type.
- `adip_kafka_stream_age_seconds`: Gauge for stream staleness.

### LangSmith Traces
Full execution traces are available for every pipeline run, including agent durations and LLM token usage.

---

## 16. Design Decisions

| Factor | Decision | Reasoning |
| :--- | :--- | :--- |
| **Vector Store** | **Qdrant** | Native support for hybrid search and named vectors. |
| **Messaging** | **Kafka** | Offset-based replay and high throughput for event logs. |
| **Clustering** | **HDBSCAN** | Automatic K discovery and noise rejection. |
| **API Framework** | **FastAPI** | Native async support and Pydantic validation. |
| **Orchestration** | **LangGraph** | Support for cycles, state persistence, and HITL. |

---

## 17. Graceful Degradation

ADIP is designed to be resilient to external service failures.

| Service | Primary | Fallback | Impact |
| :--- | :--- | :--- | :--- |
| **Qdrant** | Qdrant Server | In-memory Numpy | No persistence across restarts. |
| **PostgreSQL** | PostgreSQL | SQLite | Same schema, file-based storage. |
| **Kafka** | Kafka Consumer | Mock Generator | Synthetic data instead of live stream. |
| **Redis** | Redis Pub/Sub | In-memory Deque | No cross-process messaging. |

---

## 18. Project Structure

```
adip/
├── graph/               # LangGraph state machine & routing
├── agents/              # Specialized agent implementations
├── rag/                 # Chunking, embedding, & retrieval
├── clustering/          # UMAP & HDBSCAN logic
├── scoring/             # Risk formula & Bayesian updates
├── ingestion/           # Kafka & Jira normalization
├── persistence/         # DB & freshness monitoring
└── api/                 # FastAPI REST endpoints
```

---

## 19. Test Results

**48 passed in 193.50s**

- `test_state.py`: 13 passed (enums, models, validation)
- `test_chunker.py`: 8 passed (stack trace, jira, cicd chunking)
- `test_risk_formula.py`: 5 passed (weights, bounds, tiers)
- `test_pipeline.py`: 3 passed (full e2e, batch mode, report gen)

---

## 20. Roadmap

- [ ] **GPU acceleration**: CUDA support for UMAP and embeddings.
- [ ] **Incremental clustering**: Update clusters without full re-fit.
- [ ] **Multi-repo support**: Federated risk scoring across repositories.
- [ ] **Real git integration**: Live `git log` and coverage parsing.
- [ ] **Grafana dashboards**: Pre-built Prometheus visualizations.

---

## License

MIT
