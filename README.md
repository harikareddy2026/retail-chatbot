# Retail Merchant AI Chatbot

### Production-grade agentic RAG system for multi-tenant retail merchants

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-4B8BBE)]()
[![LangChain](https://img.shields.io/badge/LangChain-latest-1C3C3C)]()
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2?logo=mlflow)]()
[![pgvector](https://img.shields.io/badge/pgvector-HNSW_384dim-4169E1?logo=postgresql)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-3_servers-009688?logo=fastapi)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)]()
[![PySpark](https://img.shields.io/badge/PySpark-3.5.3-E25A1C?logo=apachespark)]()
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)]()
[![Routing](https://img.shields.io/badge/Routing_Accuracy-100%25-brightgreen)]()
[![Relevancy](https://img.shields.io/badge/Answer_Relevancy-1.000-brightgreen)]()
[![Faithfulness](https://img.shields.io/badge/Faithfulness-0.722-yellow)]()

---

## What this is

A production-grade retail merchant AI assistant built on LangGraph agentic
architecture. Merchants log in and ask questions about their products,
inventory, orders, and analytics in natural language. The agent classifies
every query, routes it to the correct worker, calls live data APIs, and
returns grounded answers — with merchant data isolation enforced at the
database layer.

Built in 15 days as a portfolio project demonstrating full AI Engineer skills:
data pipeline, vector search, agentic orchestration, evaluation,
and production deployment.

---

## Live metrics

| Metric | Score | Notes |
|---|---|---|
| Routing accuracy | **100%** (20/20) | All 4 worker types tested |
| Answer relevancy | **1.000** | Embedding cosine similarity |
| Faithfulness | **0.722** | Keyword routing baseline |
| Test questions | **20** | RAG · action · data · escalation · injection |
| MLflow runs | **8+** | Full experiment history tracked |
| Workers | **4** | RAG / action / data / escalation |
| MCP servers | **3** | Inventory · orders · pricing |
| Embedding dims | **384** | all-MiniLM-L6-v2, local, zero API cost |

> **Faithfulness note:** 0.722 baseline reflects keyword routing fallback
> (AWS Bedrock blocked by org SCP). Expected to reach 0.85+ with
> Claude Sonnet synthesising answers across all retrieved chunks.

---

## Architecture

```
CSV files (products · merchants · orders)
         │
         ▼
PySpark ETL — Google Colab
    ├── Bronze Delta (raw, append-only)             → S3
    ├── Silver Delta (cleaned, typed)               → S3
    └── Gold Delta  (1000-char chunks, 200 overlap) → S3
         │
         ▼
all-MiniLM-L6-v2 local embeddings (384-dim, no API cost)
         │
         ▼
Supabase pgvector
    ├── HNSW index (m=16, ef_construction=128, ef_search=100)
    ├── Merchant isolation: WHERE merchant_id = %s
    └── user_merchant_map: server-side auth (never trust client)
         │
         ▼
LangGraph StateGraph
    ├── classifier_node  → blocks injections + off-topic queries
    ├── supervisor_node  → keyword routing to correct worker
    │        │               │               │            │
    │    rag_worker   action_worker    data_worker   escalation
    │    pgvector     MCP servers      SQL direct    fallback
    │    search       (3 FastAPI)      analytics
    └── MemorySaver checkpointer → conversation persistence
         │
         ▼
MLflow autolog
    ├── Traces (tool name · latency · routing decision)
    ├── Metrics (faithfulness · relevancy · routing accuracy)
    └── Model Registry (retail-merchant-chatbot v1)
         │
         ▼
Streamlit UI
    ├── Server-side merchant auth (user_merchant_map lookup)
    ├── st.spinner (agent thinking indicator)
    └── Agent reasoning expander (worker · merchant · blocked flag)
```

---

## Tech stack

| Layer | Technology | Decision rationale |
|---|---|---|
| Data pipeline | PySpark 3.5.3 + Delta Lake | ACID transactions, time travel, schema evolution |
| Storage | AWS S3 (Bronze/Silver/Gold) | Scalable, durable, Delta-compatible |
| Vector DB | Supabase pgvector (HNSW) | Colocation with transactional data, no extra service |
| Embeddings | all-MiniLM-L6-v2 (384-dim) | Local, free, no AWS API dependency |
| Agent framework | LangGraph StateGraph | Full control over graph structure and state schema |
| Persistence | MemorySaver (→ AsyncPostgresSaver) | Session state across conversation turns |
| Live data | 3 async FastAPI MCP servers | Clean separation of concerns, independent scaling |
| Connection pool | asyncpg (min=2, max=10) | Warm connections, stays under Supabase limit |
| Auth | psycopg2 + user_merchant_map | Server-side only — client never controls merchant_id |
| Rate limiting | slowapi 30 req/min | Protects against agent runaway loops |
| Retry | tenacity (3 attempts, exponential) | Handles MCP server restarts transparently |
| Observability | MLflow autolog + traces | Every routing decision tracked, PII protected |
| Evaluation | RAGAS (embedding similarity) | Faithfulness + relevancy + routing accuracy |
| UI | Streamlit + session_state | Rapid prototyping, agent reasoning expander |
| Containerisation | Docker + docker-compose | One command starts all 4 services |

---

## 4 Architect decisions

### 1 — Native Postgres role over OAuth tokens
Databricks OAuth tokens expire after 1 hour. In a long agent session
this causes silent mid-conversation authentication failures — the pool
reconnects but fails auth, and the user sees a confusing service error.

A native Postgres role with static password stored in AWS Secrets Manager
never expires. The asyncpg pool stays healthy for the lifetime of the process.

### 2 — Server-side merchant_id from user_merchant_map
After login, the server queries `user_merchant_map WHERE email = ?`
and stores the result in session state. The client never supplies
merchant_id and the server ignores any client-side value entirely.

Even if an attacker modifies their request to send `merchant_id=M002`,
the server uses the database-verified M001 value. Combined with the
`WHERE merchant_id = %s` filter on every pgvector query, this creates
two independent layers of merchant isolation.

### 3 — statement_cache_size=0 for Supabase PgBouncer
Supabase runs PgBouncer in transaction mode. PgBouncer routes each
transaction to a different underlying database connection. asyncpg
prepared statements are connection-specific — the second request
fails with `prepared statement "__asyncpg_stmt_1__" already exists`.

Setting `statement_cache_size=0` tells asyncpg to send plain text
queries every time. Eliminates the PgBouncer incompatibility entirely.

### 4 — data_worker keyword check before action_worker
"How many orders do I have?" contains the word "order" which matches
both the action_worker and data_worker keyword lists. Without ordering,
analytics queries route to the action worker which returns raw order
data instead of COUNT/SUM analytics.

Checking data_worker keywords (how many, total, revenue, count,
breakdown) before action_worker keywords improved routing from 80% to 100%.

---

## Key numbers

| Parameter | Value | Why this value |
|---|---|---|
| Chunk size | 1000 chars | Balances retrieval precision vs LLM context window |
| Chunk overlap | 200 chars (20%) | Prevents boundary sentence loss |
| Embedding dims | 384 | Local model, no API dependency |
| HNSW m | 16 | Standard for under 1M vectors, good recall |
| ef_construction | 128 | 2× default — better graph quality at build time |
| ef_search | 100 | Tunable at query time without index rebuild |
| Rate limit | 30 req/min | Covers recursion_limit=10 × 3 concurrent sessions |
| Pool min/max | 2/10 | Warm connections, stays under Supabase connection limit |
| Recursion limit | 10 | Prevents infinite supervisor loops |
| pgvector sufficient | <1M vectors | Switch to Pinecone above 10M vectors |

---

## Project structure

```
retail-chatbot/
├── agent/
│   ├── state.py            AgentState TypedDict (messages, merchant_id,
│   │                       active_worker, query_blocked, error_state)
│   ├── graph.py            LangGraph StateGraph — 6 nodes fully wired
│   └── mlflow_config.py    MLflow setup, autolog, PII protection
│
├── tools/
│   └── mcp_tools.py        4 LangChain @tools with tenacity retry
│                           and try/except error boundaries
│
├── mcp_servers/
│   ├── shared/
│   │   ├── auth.py         APIKeyMiddleware (X-API-Key header)
│   │   └── db.py           asyncpg pool (statement_cache_size=0)
│   ├── inventory/main.py   GET /inventory/{sku}    port 8001
│   ├── orders/main.py      GET /orders/{merchant}  port 8002
│   └── pricing/main.py     GET /pricing/{product}  port 8003
│
├── eval/
│   ├── test_questions.py   20 test questions with ground truth
│   └── ragas_eval.py       Embedding similarity evaluation + MLflow
│
├── app.py                  Streamlit UI — login + chat + reasoning expander
├── test_agent.py           5 integration tests
├── test_routing.py         10-query routing accuracy test
├── test_tools.py           Unit tests for all 4 LangChain tools
├── register_model.py       MLflow Model Registry registration
├── docker-compose.yml      4 services with health checks
├── Dockerfile              Single shared image for all services
├── requirements.txt        All packages pinned
└── .env.example            Credential template (real .env never committed)
```

---

## Quick start — Docker (recommended)

```bash
git clone https://github.com/harikareddy2026/retail-chatbot.git
cd retail-chatbot

# Create .env from template
cp .env.example .env
# Edit .env with your Supabase and AWS credentials

# Start all 4 services — health checks ensure correct startup order
docker compose up

# Open http://localhost:8501
# Login with: demo@test.com
```

---

## Manual setup (development)

```bash
# Create conda environment with Python 3.11
conda create -n retail-chatbot python=3.11 -y
conda activate retail-chatbot

# Install all dependencies
pip install -r requirements.txt

# Add credentials
cp .env.example .env
# Edit .env with your values

# Terminal 1
uvicorn mcp_servers.inventory.main:app --port 8001 --reload

# Terminal 2
uvicorn mcp_servers.orders.main:app --port 8002 --reload

# Terminal 3
uvicorn mcp_servers.pricing.main:app --port 8003 --reload

# Terminal 4
streamlit run app.py --server.port 8501
```

Login at http://localhost:8501 with `demo@test.com`

---

## Run tests

```bash
# 5 agent integration tests (routing, injection, persistence, concurrent)
python3 test_agent.py

# 10-query routing accuracy test
python3 test_routing.py

# Unit tests for all 4 LangChain tools
python3 test_tools.py

# 20-question RAGAS evaluation
python3 eval/ragas_eval.py

# Register model in MLflow Model Registry
python3 register_model.py
```

---

## MLflow tracking

```bash
mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
# Open http://127.0.0.1:5001
```

| Run name | Day | Key metrics |
|---|---|---|
| day1_ingest | 1 | silver_products: 200, silver_orders: 200 |
| day2_embeddings | 2 | embeddings_stored: 450+, cost_usd: 0.0 |
| day4_tools | 4 | tools_passing: 4 |
| day5_smoke_test | 5 | all_tools_passing: 1.0 |
| day7_routing_test | 7 | routing_accuracy: 1.0 |
| day8_ragas_eval | 8 | faithfulness: 0.722, answer_relevancy: 1.000 |

---

## Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
LAKEBASE_HOST=aws-0-us-east-1.pooler.supabase.com
LAKEBASE_PASSWORD=your-supabase-database-password
LAKEBASE_USER=postgres.your-project-ref
LAKEBASE_DB=postgres
LAKEBASE_PORT=6543
MCP_API_KEY=your-generated-mcp-api-key
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_DEFAULT_REGION=us-east-1
```

---

## Resume bullets

**Data pipeline:**
Built production PySpark ETL pipeline ingesting retail CSV data into
Bronze/Silver/Gold Delta Lake on AWS S3; chunked Silver into 1000-char
overlapping segments, embedded with all-MiniLM-L6-v2 (384-dim, local,
zero API cost), stored in Supabase pgvector with HNSW index (m=16,
ef_construction=128) and server-side merchant isolation enforcing
multi-tenant data security at database layer.

**Agent architecture:**
Architected LangGraph StateGraph retail merchant AI agent with classifier
(injection blocking) → supervisor (keyword routing) → 4 workers: RAG
(pgvector similarity search), action (3 async FastAPI MCP servers with
asyncpg pool + slowapi 30/min rate limiting + tenacity retry), data
(SQL analytics on Supabase), escalation; achieved 100% routing accuracy
on 10-query test across all worker types.

**Evaluation + observability:**
Implemented RAGAS evaluation framework: 20 test questions across all
4 workers, embedding similarity scoring (faithfulness 0.722, answer
relevancy 1.000, routing accuracy 100%); registered agent as
retail-merchant-chatbot v1 in MLflow Model Registry with per-version
metric tracking; deployed Streamlit UI with server-side merchant auth
and agent reasoning expander for debugging.

---

[LinkedIn](https://linkedin.com/in/harika-mutukuru) ·
[GitHub](https://github.com/harikareddy2026)

**Background:** PySpark · Delta Lake · Apache Hudi · Kafka · GCS · AWS · Databricks

**AI Engineering:** LangGraph · LangChain · pgvector · MLflow · sentence-transformers · FastAPI · Streamlit · Docker
