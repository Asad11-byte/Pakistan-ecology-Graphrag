# Pakistan Ecology Graph RAG

LangChain + Neo4j AuraDB + Groq + Jina. Extracts a knowledge graph from a
Pakistan ecology dataset two ways (predefined schema vs. LLM-inferred schema),
and answers questions two ways (multi-hop traversal vs. community summarization).

## Setup

1. **Fill in environment variables.**
   Copy `backend/.env.example` to `backend/.env` for local dev, and fill in:
   - `GROQ_API_KEY`
   - `JINA_API_KEY`
   - `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` (from your Neo4j AuraDB instance)

2. **Install backend deps** (from project root):
   ```bash
   pip install -r requirements.txt --break-system-packages
   ```

3. **Run backend locally:**
   ```bash
   cd backend
   uvicorn app.main:app --reload --port 8000
   ```

4. **Install & run frontend:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Visit `http://localhost:5173`.

## Using the app

1. Click **"Run Ingestion (both schemas)"** — this loads the 20-document
   Pakistan ecology dataset, runs both extraction strategies, and writes
   the results into Neo4j (tagged `schema_mode: predefined` /
   `schema_mode: llm_inferred`). It also rebuilds community summaries for
   the predefined graph.
2. Click **"Fetch Comparison Stats"** to see node/relationship counts and
   type coverage for each schema strategy side by side.
3. Ask questions using either **Traversal** (good for specific, connective
   questions — "How does X affect Y?") or **Community Summary** (good for
   broad questions — "What are the main themes across Pakistan's ecology?").

## Schema comparison — what to expect and why

- **Predefined schema** (`Ecosystem`, `ProtectedArea`, `Species`, `Organization`,
  `Threat`, `Location`, `Policy` / `LOCATED_IN`, `THREATENS`, `MANAGES`, etc.)
  tends to produce a smaller, cleaner, more consistent graph — every node fits
  a known category, which makes traversal queries and Cypher predictable.
  The tradeoff: anything that doesn't fit the predefined types gets dropped
  or force-fit, so some nuance in the source text is lost.

- **LLM-inferred schema** (no constraints) tends to produce a larger, messier
  set of node/relationship types — the model invents new categories per
  document (e.g. `Program`, `Treaty`, `RiverSystem`) which can capture more
  nuance but fragments the graph: near-duplicate types (`Organization` vs
  `Institution`) make traversal and aggregation less reliable, and there's
  no guarantee of consistency across documents processed separately.

- **Cost**: both approaches call Groq once per document via
  `LLMGraphTransformer`, so token cost is roughly equal per run. The
  difference is in retrieval-time cost/reliability afterward — the
  predefined graph is cheaper to query reliably since your Cypher patterns
  can assume a known label set; the inferred graph may need fuzzier queries.

Run `/api/ingest/compare` (or the "Fetch Comparison Stats" button) after
ingestion to see actual numbers from your run — the qualitative pattern
above should show up quantitatively too (fewer unique labels but similar or
higher node/edge counts for predefined; more unique labels, often fewer
total edges per node, for LLM-inferred).

## Deploying to Vercel

1. Push this project to GitHub.
2. Import the repo into Vercel.
3. **Important:** in Vercel Project Settings → General:
   - **Root Directory**: leave blank / `./`
   - **Framework Preset**: set to **"Other"** (this project uses the legacy
     `builds`/`routes` config in `vercel.json`, so Vercel's framework
     auto-detection should not override it)
4. In Project Settings → Environment Variables, add the same 5 keys as your
   `.env` file (`GROQ_API_KEY`, `JINA_API_KEY`, `NEO4J_URI`, `NEO4J_USERNAME`,
   `NEO4J_PASSWORD`) for the Production environment.
5. Deploy. After the first deploy, update `ALLOWED_ORIGINS` in
   `backend/app/main.py` with your actual `*.vercel.app` domain, and update
   `frontend/.env` (or set `VITE_API_URL` as a Vercel env var) if needed —
   by default the frontend calls `/api`, which works automatically since
   frontend and backend are served from the same Vercel domain.
6. Once deployed, open the app and click **"Run Ingestion"** — this
   populates your live Neo4j AuraDB graph from the deployed environment.
   You only need to do this once (or after a "Reset Graph").

## Architecture notes

- **Community detection** runs in Python via `networkx`'s Louvain
  implementation (not Neo4j's Graph Data Science plugin) — this is
  intentional, since GDS is not available on standard/free-tier AuraDB.
  The graph is pulled into memory and communities are detected there,
  keeping the whole app deployable on a plain Neo4j AuraDB instance.
- **Entity-linking for traversal** uses Jina embeddings stored directly on
  Neo4j nodes with a native Neo4j vector index (`db.index.vector.queryNodes`)
  — no separate vector database needed.
- Both schema extractions write into the **same** Neo4j database, tagged
  by a `schema_mode` property, so they can coexist without a second
  database instance.
