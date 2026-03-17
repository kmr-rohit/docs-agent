# GSoC 2026 Contribution Plan: Agentic RAG on Kubeflow

## Context

**KEP:** KEP-867 — Kubeflow Documentation AI Assistant with RAG
**Issue:** [kubeflow/docs-agent#59](https://github.com/kubeflow/docs-agent/issues/59)
**Repo:** [kubeflow/docs-agent](https://github.com/kubeflow/docs-agent)
**Mentors:** @chasecadet, @tarekabouzeid, @SanthoshToorpu
**Difficulty:** Hard | **Duration:** 350 hours

---

## Requirements Restatement

The goal is to evolve `kubeflow/docs-agent` from a single-tool RAG system into a multi-index agentic retrieval service. KEP-867 requires:

1. **Agentic architecture** using Kagent (mentor-mandated) with intelligent query routing across documentation, GitHub issues, and platform code
2. **Ingestion pipelines** (KFP) to scrape, chunk, and index "Golden Data" from multiple sources
3. **KServe serving** with scale-to-zero for the LLM (Llama 3.1-8B)
4. **Deployment reference** via Terraform/manifests for OCI
5. **Automated testing** with < 5s response time target
6. **GitHub ETL** for issues and PRs (daily sync)
7. **Feedback loop** for golden dataset accumulation

**What exists today:** Single-tool MCP server (`search_kubeflow_docs`), docs ingestion pipeline (KFP), unwired `download_github_issues` component, Kagent Agent CRD with Groq, zero tests.

---

## KEP Requirements vs Current State

| KEP Requirement | Current State | Gap |
|----------------|---------------|-----|
| Agentic architecture (Kagent) | Single-tool MCP server + Kagent CRD | No multi-tool routing, no issues/code search |
| Ingestion pipelines for Golden Data | Docs pipeline works; issues component exists but unwired | Issues pipeline not connected; code pipeline missing entirely |
| KServe serving with scale-to-zero | InferenceService exists (Llama 3.1-8B) | `minReplicas: 0` not configured; cold start unaddressed |
| Terraform/OCI deployment reference | Deployed with @jaiakash via [deploy-kubeflow](https://github.com/jaiakash/deploy-kubeflow) | Upstream Terraform modules + docs needed |
| Automated testing & eval | Zero tests in repo | Need unit tests, integration tests, RAGAS eval pipeline |
| GitHub ETL (PRs + Issues) | `download_github_issues` component exists (lines 67-213) | Not wired; PR fetcher from KEP not implemented |
| Feedback loop | No feedback mechanism | Need logging endpoint + golden dataset accumulation |
| < 5s response time target | Unknown (not benchmarked) | Need performance baseline |

---

## Community Landscape (from Issue #59)

### Already Claimed / Active PRs

| Area | Contributor | PRs |
|------|------------|-----|
| Shared `rag_core.py` extraction | himanshu748 | #49 |
| Milvus connection pooling | himanshu748 | #66 |
| API key auth middleware | himanshu748 | #67 |
| Agentic RAG router (Kagent) | himanshu748 | #68 |
| Eval framework (35 golden queries) | himanshu748 | #69 |
| Platform architecture ingestion | himanshu748 | #70 |
| Terraform OCI/OKE deployment | himanshu748 | #71 |
| Scale-to-Zero (KServe) | KUNDAN1334 | Issue #47 |
| Security/SAST/Makefile | Champbreed | #96 |
| Rate limiting (ingress) | Siva-Sainath | #87 |
| Codebase structural bugs | GunaPalanivel | Issue #72 |
| Namespace hardcoding fix | KUNDAN1334 | #91 |

### Open / Under-Claimed Areas

1. **GitHub Issues ingestion pipeline** — component exists, needs wiring + collection
2. **Code ingestion (YAML/AST-aware)** — entirely new, KEP Phase 1 deliverable
3. **GitHub PR ETL pipeline** — specified in KEP but not started
4. **Developer IDE mode** — MCP for Cursor/Claude Desktop (Phase 1 KEP deliverable)
5. **Tests** — zero test files in repo (high-impact, low-competition)
6. **Milvus partition isolation** — Issue #10, foundational
7. **Schema convergence** — no decision between flexible vs explicit schema
8. **Feedback logging endpoint** — needed for golden dataset

### Key Mentor Decisions

- **Kagent is mandatory** — SanthoshToorpu explicitly rejected LangGraph as primary orchestration
- **Kubernetes-native approach** preferred (CRDs over custom Python state machines)
- **Post on Issue #59 before PRs** — mentors want design discussion first
- **MCP server is the tool host** — search logic in MCP tools, Agent CRD handles routing

---

## Strategic Positioning

**Your differentiation:** Every other contributor is building new features on top of an untested codebase. You will be the person who:

1. **Wires up the existing-but-abandoned `download_github_issues` component** that nobody else has touched
2. **Introduces the project's first test suite** — zero tests exist today
3. **Builds the code ingestion pipeline** — a KEP Phase 1 deliverable with zero code written

**What to avoid:** Do not touch rag_core (#49), connection pooling (#66), auth (#67), router (#68), eval (#69), platform ingestion (#70), terraform (#71), scale-to-zero (#47), SAST (#96), or rate limiting (#87). These are claimed.

### Guiding Principles

1. **Avoid overlap** — don't duplicate himanshu748's extensive work; differentiate through unclaimed areas
2. **Ship small, merge fast** — mentors rewarded GunaPalanivel for practical bug-finding PRs
3. **Tests are a differentiator** — zero tests exist; any PR with tests stands out
4. **Post proposals before PRs** — mentor-stated expectation
5. **Kagent-first** — all agent work must use Kagent architecture

---

## Phase 0: Community Signal (This Week — No Code)

**Goal:** Claim territory publicly, get mentor alignment before writing code.

### Step 0.1: Post Proposal on Issue #59

Write a comment describing your intended contribution areas:
1. GitHub Issues pipeline + MCP tool
2. Test infrastructure
3. Code ingestion pipeline (YAML/AST-aware)

Reference the existing `download_github_issues` component at `pipelines/kubeflow-pipeline.py:67-213` and explain it is built but unwired. Ask mentors whether they prefer a separate `issues_rag` collection or partition isolation within `docs_rag` (ties into Issue #10). Tag @SanthoshToorpu.

### Step 0.2: Open Tracking Issues

Open 3 focused GitHub issues:
1. "Wire `download_github_issues` into a KFP pipeline with `issues_rag` collection"
2. "Add unit + integration test suite for MCP server and pipeline components"
3. "YAML/Kustomize-aware code ingestion pipeline for `kubeflow/manifests`"

### Step 0.3: Local Dev Setup

```bash
# Milvus
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest

# MCP Server
cd kagent-feast-mcp/mcp-server
pip install -r requirements.txt
export MILVUS_URI="http://localhost:19530"
python server.py

# Verify
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'
```

**Dependencies:** None
**Risk:** Low. Worst case, someone claims the same area and you negotiate a split. Speed matters — post today.

---

## Phase 1: Test Infrastructure (First PR — Highest Impact, Lowest Risk)

**Goal:** Introduce the project's first test suite. Conflicts with nobody. Makes every subsequent PR more credible.

### PR 1: "Add test infrastructure and MCP server unit tests"

**Files to create:**

| File | Purpose |
|------|---------|
| `tests/conftest.py` | pytest config, shared fixtures (mock MilvusClient, mock SentenceTransformer) |
| `tests/test_mcp_server.py` | Unit tests for `search_kubeflow_docs` tool |
| `tests/test_pipeline_components.py` | Unit tests for KFP component logic |
| `pipelines/utils.py` | Extract `clean_content()` from chunk_and_embed for testability |
| `.github/workflows/tests.yml` | CI workflow running pytest on every PR |

**MCP server tests (`test_mcp_server.py`):**
- Returns formatted results when Milvus returns hits
- Returns "No results found" when Milvus returns empty
- Handles MilvusClient connection failure gracefully
- Respects `top_k` parameter (passes through to `client.search`)
- `_init()` is idempotent (calling twice doesn't create two clients)
- Output format includes score, source URL, file path, content

**Pipeline component tests (`test_pipeline_components.py`):**
- `download_github_issues` handles rate limiting (mock 403 with rate limit headers)
- `download_github_issues` skips pull requests (items with `pull_request` key)
- `clean_content` removes Hugo frontmatter, HTML tags, template syntax
- `chunk_and_embed` skips files shorter than 50 chars after cleaning

**Key design decisions:**
- Mock Milvus and SentenceTransformer at the boundary — never require a running Milvus for unit tests
- Extract cleaning logic from `chunk_and_embed` (kubeflow-pipeline.py:246-268) into `pipelines/utils.py` as `clean_content(text: str) -> str`. Smallest possible refactor that enables testability.
- KFP `@dsl.component` makes functions hard to test directly — test the inner logic by extracting it into plain Python functions

**Dependencies:** None
**Risk:** Low. Pure addition, no behavior changes.
**Estimated complexity:** Small-Medium (~200-300 lines test code, ~20 lines CI YAML)

---

## Phase 2: GitHub Issues Pipeline + MCP Tool (Core Differentiator) ✅ DONE

> **PR:** [#140 — feat: add search_github_issues MCP tool and issues ingestion pipeline](https://github.com/kubeflow/docs-agent/pull/140)
> **Branch:** `feat/github-issues-pipeline` (single commit on `origin/main`)
> **Live tested:** OCI cluster with 1,631 real issue chunks across 6 repos (kubeflow/kubeflow, kubeflow/pipelines, kubeflow/manifests, kubeflow/katib, kserve/kserve, kubeflow/website)
>
> **Key learnings from live testing:**
> - Llama models (3.1-8b, 3.3-70b) on Groq generate malformed `<function=name>` tool calls through kagent's Google ADK layer. Switched to `qwen/qwen3-32b`.
> - Removed `repo`/`state` filter params from `search_github_issues` — LLM over-filtered queries (e.g., always adding `repo: "kserve/kserve"`) which excluded the best results. Unfiltered semantic search performs better.

### PR 2: "Wire `download_github_issues` into a KFP pipeline with `issues_rag` collection"

**File:** `pipelines/issues-pipeline.py` (new)

**Pipeline:** `download_github_issues → chunk_and_embed_issues → store_milvus`

**Parameters:**
- `repos` (default: `"kubeflow/kubeflow,kubeflow/pipelines,kubeflow/manifests"`)
- `labels`, `state` (default: `"all"`), `max_issues_per_repo` (default: 200)
- `github_token`, `milvus_host`, `milvus_port`
- `collection_name` (default: `"issues_rag"`)

**Issues-specific chunking strategy:**
```
Title + metadata (repo, number, URL, labels, state) = prefix on EVERY chunk
  → ensures each chunk is self-contained for retrieval

Short issues (body + comments < chunk_size):
  → single chunk

Long issues:
  → split at comment boundaries first (each "---" separator)
  → apply RecursiveCharacterTextSplitter if individual comments > chunk_size
```

**Milvus schema for `issues_rag`:**

| Field | Type | Notes |
|-------|------|-------|
| id | INT64 | auto PK |
| file_unique_id | VARCHAR(512) | "repo:issues/1234" |
| repo_name | VARCHAR(256) | e.g., "kubeflow/pipelines" |
| issue_number | INT64 | GitHub issue number |
| issue_state | VARCHAR(32) | open/closed |
| issue_labels | VARCHAR(1024) | comma-separated labels |
| citation_url | VARCHAR(1024) | GitHub issue URL |
| chunk_index | INT64 | 0 = title+body, 1+ = comment chunks |
| content_text | VARCHAR(2000) | |
| vector | FLOAT_VECTOR(768) | all-mpnet-base-v2 |
| last_updated | INT64 | unix timestamp |
| source_type | VARCHAR(64) | "issue" — supports future Golden Data tagging |

**Tests:** `tests/test_issues_pipeline.py` — chunking strategy, short issues → 1 chunk, long issues split at comment boundaries.

**Dependencies:** PR 1 (test framework). Can be submitted in parallel if needed.
**Risk:** Medium. Must verify Milvus schema creation for new collection. Test with local Docker.
**Estimated complexity:** Medium (~300 lines pipeline, ~100 lines tests)

### PR 3: "Add `search_github_issues` MCP tool + update Agent CRD"

**Modify:** `kagent-feast-mcp/mcp-server/server.py`

```python
@mcp.tool()
def search_github_issues(query: str, repo: str = "", state: str = "", top_k: int = 5) -> str:
    """Search Kubeflow GitHub issues and discussions for troubleshooting,
    bug reports, and community solutions."""
```

**Refactor:** Extract shared search-and-format logic into `_search_collection(collection_name, query, top_k, filter_expr, extra_fields)`. Both tools call this. Reduces duplication.

**New env var:** `ISSUES_COLLECTION_NAME` (default: `"issues_rag"`)

**Modify:** `kagent-feast-mcp/manifests/kagent/setup.yaml`
- Add `search_github_issues` to Agent CRD `toolNames`
- Update `systemMessage`:
  - `search_kubeflow_docs` → official docs, setup guides, API references, concepts
  - `search_github_issues` → error messages, known bugs, workarounds, community solutions

**Tests:** Extend `test_mcp_server.py` — test `search_github_issues`, test repo/state filtering, test `_search_collection` helper.

**Dependencies:** PR 2 (`issues_rag` collection must exist)
**Risk:** Low-medium. MCP server is 67 lines. Adding a tool is straightforward.
**Estimated complexity:** Small (~60 lines server, ~50 lines manifest, ~80 lines tests)

---

## Phase 3: Code Ingestion Pipeline (KEP Phase 1 Deliverable, Unclaimed)

This is the highest-value unclaimed work. The KEP explicitly requires code ingestion for `kubeflow/manifests` and nobody has started it.

### PR 4: "Add YAML/Kustomize-aware code ingestion pipeline"

**Files to create:**

| File | Purpose |
|------|---------|
| `pipelines/code-pipeline.py` | New KFP pipeline |
| `pipelines/yaml_parser.py` | Testable pure functions for YAML/AST parsing |

**New component: `download_github_code`**
- Similar to `download_github_directory` but targets `.yaml`, `.yml`, `.py`, `.json`
- Filters: skip files > 100KB (binary artifacts, generated files)
- Target repo: `kubeflow/manifests` (Kustomize overlays)

**New component: `chunk_yaml_resources`** (the hard piece)

YAML-aware chunking:
- Split multi-document YAML at `---` boundaries (each K8s resource = one chunk)
- For each YAML document, extract: `apiVersion`, `kind`, `metadata.name`, `metadata.namespace`
- Store as searchable fields in Milvus
- If single YAML doc exceeds chunk_size (rare for K8s resources), fall back to `RecursiveCharacterTextSplitter`
- For `kustomization.yaml` files, preserve entire file as one chunk (small, losing structure hurts retrieval)

Python AST handling:
- Use `ast` module: each function/class = one chunk with name + docstring + signature
- File path and line range stored as metadata
- If AST fails (syntax errors), fall back to text chunking

**Milvus collection: `code_rag`**

| Field | Type | Notes |
|-------|------|-------|
| id | INT64 | auto PK |
| file_unique_id | VARCHAR(512) | |
| repo_name | VARCHAR(256) | |
| file_path | VARCHAR(512) | |
| resource_kind | VARCHAR(128) | Deployment, Service, etc. |
| resource_name | VARCHAR(256) | metadata.name |
| resource_namespace | VARCHAR(256) | metadata.namespace |
| file_type | VARCHAR(64) | yaml, python, kustomize, json |
| citation_url | VARCHAR(1024) | |
| chunk_index | INT64 | |
| content_text | VARCHAR(2000) | |
| vector | FLOAT_VECTOR(768) | |
| last_updated | INT64 | |

**`pipelines/yaml_parser.py`** — testable pure functions:
- `parse_yaml_documents(content: str) -> list[dict]` — splits on `---`, parses with `yaml.safe_load`, extracts K8s metadata
- `parse_python_ast(content: str, file_path: str) -> list[dict]` — AST extraction

**Tests:** `tests/test_yaml_parser.py` — multi-doc YAML splitting, K8s metadata extraction, invalid YAML handling, Python AST extraction, fallback when AST fails.

**Dependencies:** PR 1 (test framework). Independent of PRs 2-3.
**Risk:** Medium-high. YAML parsing edge cases (anchors, aliases, non-K8s YAML). Mitigate with `yaml.safe_load` (rejects custom tags) + graceful fallback to text chunking.
**Estimated complexity:** Large (~400 lines pipeline, ~150 lines parser, ~200 lines tests)

### PR 5: "Add `search_kubeflow_code` MCP tool"

**Modify:** `kagent-feast-mcp/mcp-server/server.py`

```python
@mcp.tool()
def search_kubeflow_code(query: str, resource_kind: str = "", file_type: str = "", top_k: int = 5) -> str:
    """Search Kubeflow Kubernetes manifests, Kustomize overlays, and Python scripts."""
```

Uses `_search_collection` helper from PR 3. Optional filters: `resource_kind`, `file_type`.

**Update Agent CRD** with third tool. System prompt now describes 3 tools:
- `search_kubeflow_docs` → official docs, setup, concepts
- `search_github_issues` → bugs, errors, workarounds
- `search_kubeflow_code` → K8s manifests, Kustomize overlays, deployment configs, Python scripts

**Dependencies:** PR 3 (`_search_collection` refactor), PR 4 (`code_rag` collection)
**Risk:** Low (same pattern as PR 3)
**Estimated complexity:** Small (~40 lines server, ~30 lines manifest, ~50 lines tests)

---

## Phase 4: Developer IDE Integration (Days 36-42)

**Goal:** Make the MCP endpoint usable from Cursor, Claude Desktop, and other IDE clients.

### PR 6: "Add MCP configuration for Cursor and Claude Desktop"

**Files to create:**

| File | Purpose |
|------|---------|
| `ide/cursor-mcp-config.json` | Ready-to-use Cursor MCP configuration |
| `ide/claude-desktop-config.json` | Ready-to-use Claude Desktop configuration |
| `ide/README.md` | Step-by-step setup instructions with example queries |

**Thin context response format** (per KEP + mentor spec):
- Responses should be ~150 tokens + validation links
- NOT a chatbot — a precision retrieval service
- Modify MCP tool responses to return concise golden snippets + source URLs + confidence score

**Dependencies:** PRs 3 and 5 (all 3 tools exist)
**Risk:** Low. Documentation + 2 small JSON files.
**Estimated complexity:** Small (~200 lines documentation)

---

## Phase 5: Feedback Logging (Unclaimed KEP Requirement)

### PR 7: "Add feedback logging endpoint to MCP server"

**Modify:** `kagent-feast-mcp/mcp-server/server.py`

Add new MCP tool:
```python
@mcp.tool()
def log_feedback(query: str, response_summary: str, score: int, comment: str = "") -> str:
    """Log user feedback on response quality for golden dataset accumulation."""
```

Stores feedback as JSON lines in a mounted volume. Schema per line: `{"timestamp", "query", "response_summary", "score", "comment"}`

**New file:** `pipelines/feedback-export.py` — KFP pipeline that reads feedback logs and exports to a "golden dataset" JSONL for RAGAS evaluation.

**Dependencies:** PR 1 (tests)
**Risk:** Low. Simple append-to-file logging.
**Estimated complexity:** Small (~60 lines code, ~40 lines tests)

---

## PR Submission Order

Ordered by dependency chain and merge velocity:

| # | PR | Description | Deps | Size | Status |
|---|-----|------------|------|------|--------|
| 1 | Test infrastructure | MCP server + pipeline unit tests, CI workflow | None | S-M | Ready to PR |
| 2+3 | Issues pipeline + MCP tool | [PR #140](https://github.com/kubeflow/docs-agent/pull/140) — `search_github_issues` + pipeline + Agent CRD | None | M | **Open PR** |
| 4 | Code pipeline | YAML/AST-aware chunker + `code_rag` collection | None | L | Not started |
| 5 | Code MCP tool | `search_kubeflow_code` + Agent CRD update | PR 2+3, 4 | S | Not started |
| 6 | IDE integration | Cursor/Claude Desktop configs + developer docs | PR 2+3, 5 | S | Not started |
| 7 | Feedback logging | `log_feedback` MCP tool + export pipeline | PR 1 | S | Not started |

**Parallelization:** PRs 2 and 4 can be developed simultaneously (no dependency). PR 7 can be developed alongside Phases 2-3.

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Another contributor claims Issues pipeline before you post on #59 | HIGH | Post today. Component has been unwired since creation. Speed matters. |
| Overlap with himanshu748's 7 stacked PRs | HIGH | Focus exclusively on unclaimed areas: issues pipeline, tests, code ingestion. |
| Mentors prefer partition isolation over separate collections | MEDIUM | Ask explicitly in #59 comment. Either approach works — `collection_name` is a parameter. |
| himanshu748's `rag_core` PR (#49) restructures MCP server | MEDIUM | Your changes are additive (new tools). If they merge first, rebase. Your `_search_collection` refactor is clean. |
| Code ingestion YAML parsing edge cases | MEDIUM | Use `yaml.safe_load` (rejects custom tags). Fallback to text chunking on failure. Handle 90% case well. |
| KFP component testing is awkward (`@dsl.component`) | LOW | Extract business logic into plain Python functions. Test those. Component wrapper is just I/O. |
| Mentor bandwidth limited | MEDIUM | Keep PRs small and reviewable. Post clear proposals. |
| KEP scope creep (350 hours) | MEDIUM | Prioritize Phases 0-3 (core value). Phases 4-5 are stretch goals. |
| GPU access for KServe testing | LOW | Use Groq API as fallback (already in Agent CRD). OCI deployment already working via jaiakash/deploy-kubeflow. |

---

## Success Criteria

- [x] Proposal posted on Issue #59 with positive mentor acknowledgment
- [ ] First PR (tests) merged — establishes quality-focused contributor reputation
- [x] `issues_rag` pipeline runs successfully on test cluster (1,631 chunks indexed across 6 repos on OCI)
- [x] MCP server exposes 2 tools (`search_kubeflow_docs`, `search_github_issues`) — [PR #140](https://github.com/kubeflow/docs-agent/pull/140)
- [x] Agent CRD updated with 2 tools and two-tool routing system prompt
- [ ] MCP server exposes 3rd tool (`search_kubeflow_code`)
- [ ] `code_rag` pipeline correctly chunks YAML at resource boundaries
- [ ] IDE configuration works in Cursor and/or Claude Desktop
- [ ] All PRs have passing CI tests
- [x] Zero overlap with other contributors' claimed work

---

## Key Files Reference

| File | Role | Lines |
|------|------|-------|
| `kagent-feast-mcp/mcp-server/server.py` | MCP server, single tool | 67 |
| `pipelines/kubeflow-pipeline.py` | Docs pipeline + unwired issues component | ~450 |
| `pipelines/incremental-pipeline.py` | Incremental docs update pipeline | ~401 |
| `server/app.py` | WebSocket server with tool calling | ~455 |
| `server-https/app.py` | FastAPI HTTP server with SSE | ~463 |
| `kagent-feast-mcp/manifests/kagent/setup.yaml` | Agent CRD + ModelConfig + RemoteMCPServer | ~85 |
| `kagent-feast-mcp/mcp-server/Dockerfile` | MCP server container | 17 |
| `.github/workflows/build-mcp-image.yml` | CI for MCP image | ~58 |

---

## Immediate Next Steps

1. **Get PR #140 reviewed and merged** — issues pipeline + MCP tool is open, live-tested on OCI cluster
2. **Open PR for test infrastructure** — 71 tests across 4 files, CI workflow, ready to submit on `feat/test-infrastructure` branch
3. **Start Phase 3: Code ingestion pipeline** — YAML/AST-aware chunking for `kubeflow/manifests`
4. **Post update on Issue #59** — share PR #140 and live deployment results with mentors
