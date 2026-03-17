# CLAUDE.md — kubeflow/docs-agent

## Project Overview

kubeflow/docs-agent is a RAG-based AI assistant for Kubeflow documentation. It uses Kagent (Kubernetes-native agent framework) with MCP (Model Context Protocol) tools to search across multiple indices (docs, GitHub issues, code).

**KEP:** KEP-867 — Kubeflow Documentation AI Assistant with RAG
**Issue:** [kubeflow/docs-agent#59](https://github.com/kubeflow/docs-agent/issues/59)
**Mentors:** @chasecadet, @tarekabouzeid, @SanthoshToorpu

## Architecture

```
User Query → Kagent Agent CRD (ReAct loop via A2A protocol)
  → LLM decides tool_choice → MCP server's _search_collection()
  → encode query → Milvus search → format markdown with scores + citations
  → LLM summarizes → Response
```

- **Agent:** Kagent CRD (mandatory, not LangGraph) — Kubernetes-native, declarative tools
- **Tools:** MCP server (FastMCP) exposes search tools over Streamable HTTP at `/mcp`
- **Vector DB:** Milvus with separate collections per data type (`docs_rag`, `issues_rag`, `code_rag`)
- **Embeddings:** `sentence-transformers/all-mpnet-base-v2` (768-dim)
- **LLM:** Llama 3.1-8B via KServe (or Groq API for dev — currently `qwen/qwen3-32b` works best)

## Key Files

| File | Role |
|------|------|
| `kagent-feast-mcp/mcp-server/server.py` | MCP server — search tools (`search_kubeflow_docs`, `search_github_issues`) |
| `pipelines/kubeflow-pipeline.py` | Docs ingestion KFP pipeline + unwired `download_github_issues` component |
| `pipelines/incremental-pipeline.py` | Incremental docs update pipeline |
| `pipelines/issues-pipeline.py` | GitHub issues ingestion pipeline (new, PR #140) |
| `pipelines/issues_utils.py` | Issues chunking/embedding utilities |
| `pipelines/utils.py` | Shared pipeline utils (`clean_content()`) |
| `server/app.py` | WebSocket server with tool calling |
| `server-https/app.py` | FastAPI HTTP server with SSE |
| `kagent-feast-mcp/manifests/kagent/setup.yaml` | Agent CRD + ModelConfig + RemoteMCPServer |
| `kagent-feast-mcp/mcp-server/Dockerfile` | MCP server container |
| `tests/` | Test suite (71 tests across 4 files) |
| `.github/workflows/tests.yml` | CI workflow running pytest |

## Current State (as of March 2026)

### What's Done
- **PR #140 open:** `search_github_issues` MCP tool + issues ingestion pipeline + Agent CRD update
- **Live tested on OCI:** 1,631 real issue chunks indexed across 6 repos (kubeflow/kubeflow, kubeflow/pipelines, kubeflow/manifests, kubeflow/katib, kserve/kserve, kubeflow/website)
- **71 tests built locally** on `feat/test-infrastructure` branch — project's first test suite
- **CI workflow** for MCP server image builds
- **Fixed:** hardcoded env vars in MCP server, added LLM_API_KEY support

### What's Next (Priority Order)
1. **Get PR #140 merged** — issues pipeline + MCP tool
2. **Submit test infrastructure PR** — 71 tests, CI workflow
3. **Code ingestion pipeline** — YAML/AST-aware chunking for `kubeflow/manifests` (Phase 2)
4. **`search_kubeflow_code` MCP tool** — 3rd search tool
5. **IDE integration** — Cursor/Claude Desktop MCP configs
6. **Feedback logging** — `log_feedback` MCP tool for golden dataset

## GSoC Contribution Strategy

### Areas I Own (Unclaimed by Others)
- GitHub Issues ingestion pipeline + MCP tool
- Test infrastructure (zero tests existed before)
- Code ingestion pipeline (YAML/AST-aware) — KEP Phase 1 deliverable
- IDE integration configs

### Areas to Avoid (Claimed by Others)
- `rag_core.py` extraction (himanshu748, PR #49)
- Milvus connection pooling (himanshu748, PR #66)
- API key auth middleware (himanshu748, PR #67)
- Agentic RAG router (himanshu748, PR #68)
- Eval framework (himanshu748, PR #69)
- Platform architecture ingestion (himanshu748, PR #70)
- Terraform OCI/OKE deployment (himanshu748, PR #71)
- Scale-to-Zero KServe (KUNDAN1334, Issue #47)
- Security/SAST (Champbreed, PR #96)
- Rate limiting (Siva-Sainath, PR #87)

### Guiding Principles
- **Kagent is mandatory** — all agent work must use Kagent CRDs, not LangGraph
- **Post proposals on Issue #59 before PRs** — mentor expectation
- **Ship small, merge fast** — mentors reward practical bug-finding PRs
- **Tests are a differentiator** — zero tests existed, any PR with tests stands out
- **Kubernetes-native approach** preferred (CRDs over custom Python state machines)
- **MCP server is the tool host** — search logic in MCP tools, Agent CRD handles routing

## Development Setup

```bash
# Milvus
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest

# MCP Server
cd kagent-feast-mcp/mcp-server
pip install -r requirements.txt
export MILVUS_URI="http://localhost:19530"
export EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
python server.py

# Tests
pip install -r requirements-test.txt
pytest tests/ -v
```

## Technical Decisions

- **Separate collections over partitions:** Issues need `issue_number`, `issue_state`, `issue_labels`. Code needs `resource_kind`, `resource_name`, `file_type`. Separate collections with dedicated schemas are cleaner.
- **Single MCP server with N tools:** All tools share `_search_collection` helper and same `SentenceTransformer` instance. Adding a tool is ~20 lines. Split only when independent scaling is needed.
- **`yaml.safe_load` over `ruamel.yaml`:** Rejects custom tags cleanly, lighter dependency. Falls back to text chunking on failure.
- **Thin Context design:** MCP tools return ~150 tokens + validation links (per KEP spec). Not a chatbot — a precision retrieval service.

## Known Bugs

- **Destructive pipeline:** `store_milvus` at `pipelines/kubeflow-pipeline.py:338` drops the entire collection before reinserting. If GitHub API rate-limits mid-pipeline, collection ends up empty.
- **Double `@dsl.component` decorator** on `chunk_and_embed` in kagent pipeline — outer decorator wins with wrong packages.
- **`content_text` truncated to 2000 chars** in Milvus `VARCHAR(2000)` — silently drops context from long documents.
- **Llama models on Groq** generate malformed `<function=name>` tool calls through kagent's Google ADK layer. Use `qwen/qwen3-32b` instead.

## Milvus Schemas

### docs_rag (existing)
`id` (INT64 PK), `file_unique_id` (VARCHAR), `repo_name`, `file_path`, `citation_url`, `chunk_index`, `content_text` (VARCHAR 2000), `vector` (FLOAT_VECTOR 768), `last_updated`

### issues_rag (PR #140)
Same as docs_rag plus: `issue_number` (INT64), `issue_state` (VARCHAR), `issue_labels` (VARCHAR), `source_type` (VARCHAR)

### code_rag (planned)
Same as docs_rag plus: `resource_kind` (VARCHAR), `resource_name` (VARCHAR), `resource_namespace` (VARCHAR), `file_type` (VARCHAR)

## Reference Documents

- `plan.md` — Detailed contribution plan with phases, PR order, risks
- `plan_proposal.md` — Full GSoC proposal text with architecture decisions, timeline, qualifications
- `gsoc.md` — Working document with implementation analysis and study notes
- `diagrams/` — SVG architecture diagrams (7 files)
