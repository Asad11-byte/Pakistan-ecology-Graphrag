# Pakistan Ecology Graph RAG

A Graph-based Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Neo4j AuraDB**, **Groq**, and **Jina AI**, applied to a curated dataset on Pakistan's ecosystems, protected areas, wildlife, and conservation policy.

Unlike traditional vector-based RAG, this system represents knowledge as an explicit **graph of entities and relationships**, enabling two distinct retrieval strategies — multi-hop **traversal** for specific connective questions, and **community summarization** for broad thematic questions — and compares two different strategies for building that graph in the first place.

---

## Live Demo

- **Tap here :** https://pak-ecology-graphrag.vercel.app


## Table of Contents

- [Why Graph RAG](#why-graph-rag)
- [Architecture Overview](#architecture-overview)
- [Core Concepts](#core-concepts)
- [Data Pipeline / Workflow](#data-pipeline--workflow)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Setup & Local Development](#setup--local-development)
- [Environment Variables](#environment-variables)
- [Deployment on Vercel](#deployment-on-vercel)
- [Schema Comparison: Methodology](#schema-comparison-methodology)
- [Known Limitations](#known-limitations)
- [Possible Extensions](#possible-extensions)

---

## Why Graph RAG

Standard RAG retrieves the *k* most similar text chunks to a query and hands them to an LLM. This works well for direct factual lookup, but breaks down on two common question types:

1. **Multi-hop questions** — *"How does glacial melt affect the Indus River Dolphin?"* The answer isn't in any single chunk; it requires connecting Glaciers → Indus River → Water Scarcity → Indus Delta → Dolphin Habitat across multiple source documents.
2. **Global / thematic questions** — *"What are the main environmental threats across Pakistan?"* No single chunk summarizes this; it requires synthesizing across the *entire* corpus, not just the top-k most similar chunks.

Graph RAG addresses both by first converting unstructured text into a **knowledge graph** (entities + typed relationships), then retrieving via graph operations — traversal for the first case, community detection + summarization for the second — instead of pure similarity search.

---

## Architecture Overview

```
                         ┌─────────────────────────┐
                         │   Pakistan Ecology       │
                         │   Dataset (20 docs, JSON)│
                         └────────────┬─────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     │      LangChain LLMGraphTransformer │
                     │            (powered by Groq)       │
                     └────────────────┬────────────────┘
              ┌───────────────────────┼───────────────────────┐
              │                                                │
   ┌──────────▼──────────┐                         ┌───────────▼───────────┐
   │  Predefined Schema    │                         │  LLM-Inferred Schema   │
   │  (constrained types)  │                         │  (model chooses types) │
   └──────────┬──────────┘                         └───────────┬───────────┘
              │                                                │
              └───────────────────┬────────────────────────────┘
                                   │  tagged with `schema_mode` property
                        ┌──────────▼──────────┐
                        │     Neo4j AuraDB      │
                        │  (nodes + relationships,
                        │   + Jina embeddings on
                        │   nodes for vector search)
                        └──────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                                          │
   ┌──────────▼──────────┐                   ┌───────────▼────────────┐
   │   Traversal Retrieval │                   │  Community Summarization│
   │  entity anchor (Jina) │                   │  networkx Louvain        │
   │  → N-hop Cypher walk  │                   │  → per-cluster summaries │
   │  → Groq synthesis     │                   │  → Groq map-reduce       │
   └──────────┬──────────┘                   └───────────┬────────────┘
              │                                          │
              └────────────────────┬────────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   React Chat UI       │
                         │  (mode + schema toggle)│
                         └───────────────────────┘
```

**Deployment topology on Vercel:**

```
  Browser
     │
     ▼
 Vercel Edge (single domain)
     │
     ├── /api/*  ──────────────►  Python Serverless Function (api/index.py)
     │                             └─ FastAPI app (backend/app/)
     │                                └─ LangChain + Groq + Jina + Neo4j driver
     │                                     └─ (network call) ──► Neo4j AuraDB (external, persistent)
     │
     └── /*  ──────────────────►  Static files (frontend/dist, from Vite build)
```

Neo4j AuraDB is the only stateful component — everything else (the serverless function, the static frontend) is stateless and rebuilt fresh on every deploy.

---

## Core Concepts

### 1. Two schema extraction strategies, compared

| | Predefined Schema | LLM-Inferred Schema |
|---|---|---|
| **Node types** | Fixed list: `Ecosystem`, `ProtectedArea`, `Species`, `Organization`, `Threat`, `Location`, `Policy` | Whatever the model decides per document (e.g. `Program`, `Treaty`, `RiverSystem`) |
| **Relationship types** | Fixed list: `LOCATED_IN`, `THREATENS`, `PROTECTS`, `MANAGES`, `FEEDS_INTO`, `PREYS_ON`, `PARTNERS_WITH`, `IMPLEMENTED_BY`, `SUPPORTS`, `HOSTS`, `PART_OF` | Freely generated |
| **Consistency** | High — every node fits a known category | Lower — near-duplicate types can appear across documents |
| **Nuance captured** | Lower — anything outside the fixed types is force-fit or dropped | Higher — captures domain-specific relationships the fixed schema didn't anticipate |
| **Query reliability** | High — Cypher patterns can assume a known label set | Lower — requires fuzzier matching since label vocabulary isn't fixed |
| **Cost** | ~1 Groq call per document | ~1 Groq call per document (same order of cost) |

Both extraction passes write into the **same** Neo4j database, distinguished only by a `schema_mode` property (`"predefined"` or `"llm_inferred"`) on every node and relationship — this avoids needing two separate databases while still allowing clean, independent querying of either graph.

### 2. Two retrieval strategies

**Traversal (multi-hop)** — for specific, connective questions.
1. Embed the user's question with Jina.
2. Find the closest matching entity node via Neo4j's native vector index (`db.index.vector.queryNodes`) — falls back to word-overlap scoring against node names if the vector search doesn't return a confident match.
3. Walk N hops of the graph outward from that anchor entity (Cypher variable-length path match).
4. Feed the resulting subgraph (as a list of relationship chains) to Groq, which synthesizes a natural-language answer grounded only in that subgraph.

**Community Summarization (global)** — for broad, thematic questions.
1. Pull the full graph (for a given `schema_mode`) into Python via the `neo4j` driver.
2. Build an in-memory graph with `networkx` and run **Louvain community detection** to find clusters of densely-interconnected entities.
3. **Map step:** summarize each community individually with Groq (e.g. "the northern mountains cluster: Deosai, Khunjerab, snow leopard, markhor...").
4. **Reduce step:** given the user's question and all community summaries, ask Groq to synthesize a final answer across all relevant clusters.
5. Summaries are cached in memory after the first build (or an explicit rebuild) to avoid re-summarizing on every query.

> **Why not Neo4j's Graph Data Science (GDS) library for community detection?** GDS is a separate paid product (AuraDS), not included in standard/free-tier AuraDB. Running Louvain in Python via `networkx` gets the same result without requiring a different Neo4j product tier — keeping the whole project deployable on a free AuraDB instance.

### 3. Entity-linking via Neo4j's native vector index

Every extracted entity gets a Jina embedding stored directly on its Neo4j node (all nodes also carry a shared `:Entity` label). A single vector index (`entity_embeddings`) is created over that label/property, letting traversal retrieval semantically match a user's question to the right anchor entity — even with typos or rephrasing — without needing a separate vector database like Qdrant or Pinecone.

---

## Data Pipeline / Workflow

### Step 1 — Ingestion (`POST /api/ingest/all`)

1. Load the 20-document Pakistan ecology dataset (`backend/app/data/pakistan_ecology_dataset.json`).
2. For each document, run **both** extraction strategies sequentially (predefined, then LLM-inferred), processing one document at a time — not as a single batch call — so a malformed generation on one document doesn't abort the whole run.
3. For each extracted graph document:
   - Batch-embed all newly-seen entity names in a single Jina API call (not one call per entity — keeps API usage and latency down).
   - Write nodes via `MERGE` (idempotent — safe to re-run without creating duplicates), tagged with `schema_mode` and `source_docs`.
   - Write relationships via `MERGE`, tagged with `schema_mode`.
4. Ensure the `entity_embeddings` vector index exists (created once, reused after).

### Step 2 — Schema comparison (`GET /api/ingest/compare`)

Runs aggregate Cypher queries filtered by `schema_mode`, returning node/relationship counts and the distinct set of node labels and relationship types actually produced by each strategy — the quantitative half of the schema comparison (see [below](#schema-comparison-methodology)).

### Step 3 — Community rebuild (`POST /api/query/community/rebuild`)

Pulls the `predefined`-schema graph into `networkx`, runs Louvain detection, and regenerates per-community summaries via Groq. Triggered automatically after ingestion, and can be re-triggered manually.

### Step 4 — Querying

- `POST /api/query/traversal` — anchor + N-hop walk + Groq synthesis (see [Core Concepts](#core-concepts)).
- `POST /api/query/community` — map-reduce over cached community summaries.

Both accept a `schema_mode` parameter, letting you compare how the same question is answered depending on which extraction strategy produced the underlying graph.

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| LLM | **Groq** (`llama-3.3-70b-versatile`) | Graph extraction, traversal answer synthesis, community summarization (map + reduce) |
| Graph extraction | **LangChain** (`LLMGraphTransformer`, `langchain-experimental`) | Converts unstructured text into typed nodes/relationships |
| Embeddings | **Jina AI** (`jina-embeddings-v3`) | Entity-linking — embeds entity names and user queries for semantic anchor matching |
| Graph database | **Neo4j AuraDB** | Persistent graph store; native vector index for entity search |
| Community detection | **networkx** (Louvain) | Runs in Python — avoids requiring Neo4j's paid GDS plugin |
| Backend | **FastAPI** | REST API, served as a Python serverless function on Vercel |
| Frontend | **React + Vite** | Chat interface with schema/retrieval-mode toggles and a live schema-comparison table |
| Hosting | **Vercel** | Single-domain deployment: static frontend + Python serverless function, routed via `vercel.json` |

---

## Project Structure

```
rag-graph/
├── api/
│   └── index.py                 # Vercel entrypoint — imports the FastAPI app
├── backend/
│   ├── .env.example
│   └── app/
│       ├── main.py              # FastAPI app, CORS, /api-prefixed router
│       ├── core/
│       │   └── config.py        # Pydantic settings (env vars)
│       ├── data/
│       │   └── pakistan_ecology_dataset.json
│       ├── services/
│       │   ├── neo4j_service.py             # Driver wrapper, schema_mode-tagged reads/writes
│       │   ├── jina_service.py              # Embeddings for entity-linking
│       │   ├── graph_extraction_service.py  # LLMGraphTransformer, both schema modes
│       │   ├── traversal_service.py         # Multi-hop retrieval
│       │   └── community_service.py         # Louvain detection + map-reduce summarization
│       └── routes/
│           ├── ingest.py        # /api/ingest/* endpoints
│           └── query.py         # /api/query/* endpoints
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Chat UI, ingestion controls, comparison table
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── requirements.txt              # Root-level — required for Vercel's Python builder
├── vercel.json
└── README.md
```

---

## API Reference

All routes are prefixed with `/api`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/` | App status/info |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/ingest/predefined` | Run predefined-schema extraction only |
| `POST` | `/api/ingest/llm-inferred` | Run LLM-inferred-schema extraction only |
| `POST` | `/api/ingest/all` | Run both extraction strategies |
| `GET` | `/api/ingest/compare` | Return comparison stats for both schema modes |
| `DELETE` | `/api/ingest/reset` | Wipe all nodes/relationships from Neo4j |
| `POST` | `/api/query/traversal` | `{ "question": str, "schema_mode": "predefined" \| "llm_inferred" }` → multi-hop answer |
| `POST` | `/api/query/community` | `{ "question": str, "schema_mode": ... }` → global/thematic answer |
| `POST` | `/api/query/community/rebuild?schema_mode=predefined` | Recompute communities + summaries |

---

## Setup & Local Development

**1. Install backend dependencies** (from project root):
```bash
pip install -r requirements.txt
```

**2. Configure environment variables:**
```bash
cp backend/.env.example backend/.env
# then fill in GROQ_API_KEY, JINA_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
```

**3. Run the backend:**
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
Verify at `http://127.0.0.1:8000/api/health`.

**4. Run the frontend:**
```bash
cd frontend
npm install
npm run dev
```
Visit `http://localhost:5173`. In development, Vite proxies `/api/*` to `http://127.0.0.1:8000` (see `vite.config.js`).

**5. Ingest data:** click **"Run Ingestion (both schemas)"** in the UI. This is a one-time step per fresh Neo4j instance — the graph persists in AuraDB across restarts, so you don't need to re-run it every session.

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key — powers all LLM calls (extraction, synthesis, summarization) |
| `GROQ_MODEL` | Defaults to `llama-3.3-70b-versatile` |
| `JINA_API_KEY` | Jina AI API key — powers entity-name and query embeddings |
| `JINA_MODEL` | Defaults to `jina-embeddings-v3` |
| `NEO4J_URI` | AuraDB connection URI (`neo4j+s://...`) |
| `NEO4J_USERNAME` | AuraDB username |
| `NEO4J_PASSWORD` | AuraDB password |

Set these in `backend/.env` for local development, and in **Vercel → Project Settings → Environment Variables** for production.

---

## Deployment on Vercel

1. Push the repository to GitHub and import it into Vercel.
2. **Project Settings → General:**
   - **Root Directory:** blank / `./`
3. **Project Settings → Environment Variables:** add all 6 variables listed above for the Production environment.
4. **`vercel.json`** ties the deployment together:
   ```json
   {
     "buildCommand": "cd frontend && npm install && npm run build",
     "outputDirectory": "frontend/dist",
     "functions": {
       "api/index.py": { "includeFiles": "backend/**" }
     },
     "rewrites": [
       { "source": "/(.*)", "destination": "/index.html" }
     ]
   }
   ```
   - `includeFiles` bundles the `backend/` directory into the Python function (otherwise only `api/index.py` itself would be deployed).
   - Vercel automatically routes all `/api/*` requests to `api/index.py` — no explicit rewrite is needed for that; only the SPA fallback rewrite (for client-side React routing) is required.
5. After the first deploy, click **"Run Ingestion"** once from the live app to populate your production Neo4j graph. This only needs to be done once — Neo4j AuraDB is persistent and independent of your Vercel deployment lifecycle.

---

## Schema Comparison: Methodology

The `/api/ingest/compare` endpoint (and the "Fetch Comparison Stats" button in the UI) reports, for each schema strategy:

- Total node count and relationship count
- The distinct set of node labels actually produced
- The distinct set of relationship types actually produced

**Expected qualitative pattern**, to be confirmed against your own run's numbers: the **predefined** schema should show fewer unique labels (bounded by the 7 allowed types) but reliably consistent categorization; the **LLM-inferred** schema should show a wider, messier variety of labels — capturing more nuance per document at the cost of cross-document consistency (e.g. the same real-world concept classified differently in different documents).

For a full write-up, run both `/api/ingest/all` and `/api/ingest/compare`, then manually spot-check 10-15 extracted triples from each mode in Neo4j Browser to assess extraction accuracy alongside the quantitative counts.

---

## Known Limitations

- **Entity deduplication is exact-string-based.** `MERGE` matches on the literal `name` property, so near-duplicate entities from inconsistent LLM capitalization (e.g. *"Ministry of Climate Change"* vs *"Pakistan's Ministry of Climate Change"*) can end up as separate nodes, fragmenting connectivity. A normalization step (lowercase/trim before merge, with a separate display-name property) would mitigate this.
- **Community summaries are cached in-memory**, not persisted — they're rebuilt from Neo4j on first use after every cold start (serverless functions don't retain memory between invocations on Vercel), so the very first community query after a period of inactivity will be slower than subsequent ones.
- **Groq's free tier has a daily token cap** (100,000 TPD at time of writing) — repeated ingestion/reset cycles during development can exhaust this quickly; production usage at scale would need a paid tier.
- **Traversal depth is fixed per request** (default 3 hops) — very distant multi-hop connections beyond that depth won't be found without increasing it.

## Possible Extensions

- Normalize entity names at write-time to reduce duplicate-node fragmentation.
- Persist community summaries in Neo4j itself (as `CommunitySummary` nodes) instead of an in-memory cache, so they survive cold starts without rebuilding.
- Add a hybrid retrieval mode that combines traversal and community summarization for questions that are partly specific and partly thematic.
- Expand the dataset beyond the initial 20 documents for broader coverage.