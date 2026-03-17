# GSoC 2026: Agentic RAG on Kubeflow — Working Document

## Current Implementation Analysis

### What Exists Today

| Component | Location | Status |
|-----------|----------|--------|
| Docs ingestion pipeline (KFP) | `pipelines/kubeflow-pipeline.py` | Working — scrapes kubeflow/website, chunks with RecursiveCharacterTextSplitter, embeds with all-mpnet-base-v2, stores in Milvus |
| Incremental update pipeline | `pipelines/incremental-pipeline.py` | Working — targeted file updates via delete+reinsert |
| `download_github_issues` component | `pipelines/kubeflow-pipeline.py:67-213` | Code exists but NOT wired into any pipeline |
| WebSocket API server | `server/app.py` | Working — single tool (search_kubeflow_docs), streams via WS |
| HTTPS REST API server | `server-https/app.py` | Working — FastAPI with SSE streaming |
| MCP server (FastMCP) | `kagent-feast-mcp/mcp-server/server.py` | Working — exposes search_kubeflow_docs over Streamable HTTP at /mcp |
| Kagent Agent CRD | `kagent-feast-mcp/manifests/kagent/setup.yaml` | Working — declares Agent + ModelConfig + RemoteMCPServer pointing to Groq |
| KServe InferenceService | `manifests/inference-service.yaml` | Working — Llama 3.1-8B via vLLM with tool calling enabled |
| Milvus deployment | `manifests/milvus-deployment.yaml` | Working — standalone with etcd + MinIO |
| Istio auth policies | `kagent-feast-mcp/manifests/istio/` | Working — allows Milvus traffic |
| CI/CD | `.github/workflows/` | Working — builds MCP server and API server images |

### Current Architecture (Single-Tool RAG)

```
User Query → LLM (tool_choice: auto) → search_kubeflow_docs → Milvus (docs_rag) → LLM summarizes → Response
```

One tool, one index, one shot. No multi-step reasoning, no routing, no issue/code search.

### Current Data Flow (Ingestion)

```
GitHub API (kubeflow/website) → download .md/.html → clean Hugo/HTML artifacts
  → RecursiveCharacterTextSplitter (1000 chars, 100 overlap)
  → sentence-transformers/all-mpnet-base-v2 (768-dim)
  → Milvus (docs_rag collection, IVF_FLAT, COSINE)
```

### Current Milvus Schema (docs_rag)

| Field | Type | Notes |
|-------|------|-------|
| id | INT64 | auto-generated primary key |
| file_unique_id | VARCHAR(512) | "repo:file/path" |
| repo_name | VARCHAR(256) | |
| file_path | VARCHAR(512) | |
| citation_url | VARCHAR(1024) | |
| chunk_index | INT64 | |
| content_text | VARCHAR(2000) | truncated for storage |
| vector | FLOAT_VECTOR(768) | all-mpnet-base-v2 |
| last_updated | INT64 | unix timestamp |

---

## What Needs to Be Built (GSoC Scope)

### Phase 1: Ingestion & Foundation

#### 1a. GitHub Issues Pipeline (Extend existing component)
- **Component exists:** `download_github_issues` in kubeflow-pipeline.py fetches issues with comments, rate limiting, pagination
- **Missing:** Not wired into a pipeline. Needs:
  - New KFP pipeline: `download_github_issues → chunk_and_embed → store_milvus`
  - Separate Milvus collection: `issues_rag`
  - Chunking strategy for issues differs from docs — issues have title+body+comments structure
  - Consider: keep title+body as one chunk, long comment threads as separate chunks, linked by issue number
- **Target repos:** kubeflow/kubeflow, kubeflow/pipelines, kubeflow/kfp-tekton, kubeflow/manifests

#### 1b. Code Ingestion Pipeline (New — hardest piece)
- **Does not exist.** Entirely new pipeline.
- **Target repo:** kubeflow/manifests (Kustomize overlays, YAML configs, Python scripts)
- **Challenge:** Can't just text-chunk YAML. Need structure-aware parsing:
  - YAML: parse by resource boundary (each `---` separated doc = one chunk). Store kind, name, namespace, spec as searchable content
  - Python: AST parsing to extract function signatures, class definitions, docstrings
  - Kustomize: understand overlay structure (base + patches)
- **New Milvus collection:** `code_rag`
- **Schema additions needed:** `resource_kind`, `resource_name`, `file_type` metadata fields

#### 1c. Pipeline Improvements
- Scheduled runs (cron trigger in KFP)
- "Golden Data" metadata tagging: `source_type`, `verified`, `freshness`
- Multiple source repos support

### Phase 2: Core Agent & Routing

#### 2a. New MCP Tools
- `search_github_issues(query, top_k)` — same pattern as docs, queries issues_rag collection
- `search_kubeflow_code(query, top_k)` — queries code_rag collection
- **Design decision:** single MCP server with 3 tools vs. 3 separate MCP servers
  - Start with single server (simpler), refactor later if needed
  - **ASK MENTORS:** preference on this

#### 2b. Agent CRD Update
- Update Kagent Agent CRD to reference all 3 tools
- LLM handles routing via tool_choice: "auto" (Kagent/ADK's ReAct loop naturally supports multi-turn tool calling)
- System prompt needs updating to describe all 3 tools and when to use each

#### 2c. "Thin Context" MCP Design
- Per the maintainer spec: agent returns small golden snippets (~150 tokens) + validation links
- NOT a chatbot — a precision retrieval service
- IDE integration: developers register our MCP in Cursor/Claude Desktop
- Frontend: website chat uses same MCP for best-effort answers

### Phase 3: Deployment, Security & UX

#### 3a. Infrastructure (Terraform/OCI)
- OKE cluster provisioning
- GPU node pools for KServe
- Object storage for Milvus backend
- Networking/Istio setup
- Separate "Core Kubeflow Services" (portable) from "Cloud-Specific Adapters" (OCI)

#### 3b. Helm Chart
- Package entire stack as a Helm chart
- Idempotent deployments (safe for ArgoCD git-sync)
- Configurable via values.yaml (LLM endpoints, DB connections, tool hosts)

#### 3c. KServe Scale-to-Zero
- Add `minReplicas: 0` to InferenceService
- Knative annotations for retention period tuning
- Address cold start challenge: GPU scheduling + model load = minutes
- Strategies: model caching on PVC, longer retention, quantized models

#### 3d. Frontend + Feedback Loop
- Chat UI with process transparency (show tool execution steps)
- Thumbs up/down feedback mechanism
- Log: (query, retrieved_context, response, feedback_score) for golden dataset
- Source citations with links

#### 3e. Eval Framework
- RAGAS metrics: context precision, context recall, faithfulness, answer relevancy
- Run as periodic KFP pipeline against golden dataset
- CI integration for regression testing

---

## Discussion Points for Mentors

### Architecture Decisions (Need Alignment)

1. **Single vs. multiple MCP servers?**
   - Single server with 3 tools is simpler to start. Separate servers give independent scaling.
   - Recommendation: start single, split when needed. What does the team prefer?

2. **Code ingestion scope for kubeflow/manifests:**
   - Just Kustomize/YAML resources? Or also Python/Go source code via AST?
   - How deep should structural parsing go? (resource-level chunks vs. field-level)

3. **Frontend scope in GSoC:**
   - Full functional frontend with feedback loop?
   - Or working MCP endpoint + documented IDE setup is sufficient?

4. **Router/intent detection approach:**
   - LLM-based routing (give all tools, let LLM pick) — Kagent does this naturally
   - Lightweight classifier before LLM — cheaper but more work to build
   - The spec mentions both. Which is GSoC scope?

5. **Items 9-17 in the hardened checklist:**
   - WebSockets, Redis state, OpenTelemetry, multi-tenancy, Kubecost, rate limiting, OAuth2
   - Which are GSoC scope vs. future work?

6. **Golden Dataset feedback loop:**
   - Simple logging pipeline? Or full RAGAS eval integrated into CI?

### Technical Questions

7. **Embedding model choice:**
   - Current: all-mpnet-base-v2 (768-dim, local)
   - MCP server uses remote embedding (OpenAI-compatible or TEI)
   - Should we standardize on one approach? Remote embedding is lighter but adds a dependency.

8. **Vector DB:**
   - Spec mentions pgvector as an alternative to Milvus
   - Should we support both? Or pick one?

9. **Kagent vs. custom agent:**
   - Kagent is Architecture B (primary target)
   - But Architecture A (KServe custom agent) and C (LangGraph/ADK) need "scaffolding"
   - How much scaffolding? Just manifests + README? Or working code?

---

## Understanding Stack (Study Notes)

### MCP Protocol
- JSON-RPC 2.0 over stdio or Streamable HTTP
- 3 core operations: `initialize`, `tools/list`, `tools/call`
- FastMCP auto-generates tool schemas from Python function signatures
- Server knows nothing about agents/LLMs — it's just a tool server
- Any MCP client can connect: Kagent, Claude Desktop, Cursor, custom scripts

### Kagent Architecture
- Kubernetes operator (Go controller) using controller-runtime
- CRDs: Agent, ModelConfig, RemoteMCPServer
- Agent runtime: Python FastAPI + Google ADK (migrated from AutoGen)
- Execution: ReAct loop — LLM call → tool call via MCP → append result → repeat
- Agent-to-Agent (A2A) protocol for inter-agent communication
- Session persistence via SQLite (dev) or PostgreSQL (prod)
- Human-in-the-loop: `requireApproval` field pauses loop until user confirms

### Kagent Agent Execution Flow
```
kubectl apply Agent CRD
  → Controller creates ConfigMap + Deployment + Service
  → Agent pod runs FastAPI + ADK
  → User sends query via A2A (POST /api/a2a/<ns>/<name>/task)
  → ADK Agent.run() ReAct loop:
      1. LLM call with system prompt + history + tool schemas
      2. LLM returns text (done) or tool_call (continue)
      3. Tool executed via MCP (HTTP to RemoteMCPServer URL)
      4. Result appended to history
      5. GOTO 1
  → Response streamed as SSE events
```

### KServe Scale-to-Zero
- Uses Knative Serving under the hood
- `minReplicas: 0` enables scale-to-zero
- Cold start chain: Knative Activator → K8s schedules pod → GPU allocated → vLLM loads model → ready
- GPU cold start: 2-5 minutes (vs. seconds for web apps)
- Tuning: `scaleToZeroPodRetentionPeriod`, `scaleTarget`, `scaleMetric`

---

## Implementation Starting Points

### Quick Wins (Good First PRs)
1. Wire `download_github_issues` into a KFP pipeline with its own Milvus collection
2. Add `search_github_issues` tool to the MCP server
3. Update Agent CRD with the new tool
4. Add tests for MCP server tools

### Medium Effort
5. YAML-aware chunking for kubeflow/manifests (code ingestion)
6. KServe scale-to-zero configuration + documentation
7. Feedback logging endpoint in the API server

### Large Effort
8. Full code ingestion pipeline with AST parsing
9. Terraform modules for OCI deployment
10. Helm chart for the entire stack
11. Frontend with feedback loop and process transparency
12. RAGAS eval pipeline

---

## Local Development Setup

### Prerequisites
- Python 3.9+
- Docker
- kubectl + access to a K8s cluster (or Minikube/Kind for local dev)
- Milvus (can run standalone via Docker)

### Running MCP Server Locally
```bash
cd kagent-feast-mcp/mcp-server
pip install -r requirements.txt

# Set env vars
export MILVUS_URI="http://localhost:19530"
export EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
export PORT=8000

python server.py
# Server runs at http://localhost:8000/mcp
```

### Running Milvus Locally
```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest
```

### Testing MCP Server
MCP Streamable HTTP requires the client to accept both JSON and SSE:
```bash
# Test tool discovery
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'

# Test tool call
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"search_kubeflow_docs","arguments":{"query":"KServe setup"}},"id":2}'
```
