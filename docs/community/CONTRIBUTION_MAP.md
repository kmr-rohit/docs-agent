# Community contribution map (docs-agent)

Use this at community calls and in issue triage. It separates **maintainer-owned agentic RAG scope** from work that is safe and valuable for community contributors.

## Current architecture (what exists today)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  kubeflow.org (optional)                                                │
│    └── frontend/docs_scripts/chatbot.js  →  Kagent A2A (JSON-RPC)       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  docs-agent namespace                                                   │
│    kagent (Helm)  →  Agent CRD  →  ModelConfig (KServe Qwen)          │
│                  └→  RemoteMCPServer  →  MCP server :8000              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌──────────────────────────────┐    ┌──────────────────────────────────────┐
│  ml-infra namespace          │    │  kubeflow namespace (pipelines)      │
│  Milvus (operator + CR)      │    │  KFP runs: docs / issues / code      │
│  TEI embeddings (KServe)     │    │  ingestion → Milvus collections      │
│  Qwen LLM (KServe, GPU)      │    └──────────────────────────────────────┘
└──────────────────────────────┘

Collections: kubeflow_docs | issues_rag | code_rag
```

**Validation path:** `tests.yml` + `oke-cicd.yaml` (operator forks) → build MCP image → optional OKE deploy → `smoke_tools.py`.

See also: [gsoc2026_agentic_rag.md](../../gsoc2026_agentic_rag.md) for the longer-term target architecture.

---

## Maintainer-owned (agentic RAG core)

Please **open a design discussion issue first** before large PRs in these areas. Maintainers (GSoC / core) own the product direction here.

| Area | Why maintainer-owned |
|------|----------------------|
| Agent orchestration & routing | Semantic router, LangGraph/ADK, multi-agent graphs — not implemented yet; architecture decisions pending |
| MCP tool contracts | Tool names, schemas, retrieval strategy, context formatting in `mcp-server/server.py` |
| Kagent `Agent` / `systemMessage` | Prompt routing rules, tool-mandatory behavior, citation policy in `manifests/kagent/setup.yaml` |
| RAG quality & eval design | Golden datasets, RAGAS/LangSmith strategy, Katib tuning experiments |
| LLM gateway architecture | LiteLLM vs kgateway, quota model, auth integration (design first) |
| Argo CD / GitOps layout | Cluster operator decisions; community can help after scaffold is agreed |

**Label:** `maintainer-only` on GitHub — discuss before coding.

---

## Great for community contributions

These improve reliability, onboarding, and UX **without** changing agent reasoning design.

### Tests & CI (highest impact, lowest risk)

| Work | Paths | Notes |
|------|-------|-------|
| Unit tests for untested modules | `embeddings_client.py`, `smoke_tools.py` | Mock HTTP; no cluster needed |
| Collection name parity tests | `rag_collections.py` ↔ `pipelines/utils.py` | Prevent drift |
| Integration marker + Docker MCP smoke | `AGENTS.md` curl flow | Optional CI job |
| `terraform validate` / `fmt` in CI | `docs-agent-mcp/terraform/` | No cloud creds required |
| kubeconform on manifests | `docs-agent-mcp/manifests/` | Schema validation |
| Compile `incremental-pipeline.py` in CI | `oke-cicd.yaml` | Currently omitted |
| Deduplicate overlapping CI jobs | `tests.yml` vs `oke-cicd.yaml` | Document canonical workflow |

### Documentation

| Work | Paths |
|------|-------|
| Fix README drift (legacy vs MCP/Kagent primary path) | `README.md` |
| Remove Feast contradiction | `docs-agent-mcp/README.md` |
| TEI vs sentence-transformers in pipeline README | `docs-agent-mcp/pipelines/README.md` |
| `code_rag` ingestion runbook | new `docs/INGESTION.md` |
| Repo-specific contributing guide | `CONTRIBUTING.md` |

### Pipelines & data (operational, not agent logic)

| Work | Paths |
|------|-------|
| Migrate `incremental-pipeline.py` to TEI | `pipelines/incremental-pipeline.py` |
| Deduplicate `clean_content` in docs pipeline | `kubeflow-pipeline.py` + `utils.py` |
| Run / document `code-pipeline.py` for `code_rag` | `pipelines/`, `submit_run.py` |
| GitHub token secret docs for KFP | `pipelines/README.md` |

### Frontend (UX only)

| Work | Paths |
|------|-------|
| Thumbs up/down + webhook stub | `frontend/docs_scripts/chatbot.js` |
| Copy-to-clipboard on answers | `frontend/` |
| Tool-step / “searching…” transparency | `frontend/` |
| JS unit tests (storage helpers) | `frontend/` |

### Infra packaging (not platform WG core)

| Work | Paths |
|------|-------|
| Helm chart scaffold for app manifests | `manifests/mcp-server`, `manifests/kagent` |
| Deduplicate Istio policies (TF vs YAML) | `terraform/istio_policies.tf`, `manifests/istio/` |
| Terraform operator README | `docs-agent-mcp/terraform/` |

### MCP hardening (small, scoped PRs welcome)

| Work | Paths |
|------|-------|
| Retry/backoff on TEI failures | `embeddings_client.py` |
| Clearer error messages (empty collection vs down) | `server.py` |
| Optional `MCP_API_KEY` env gate | `server.py` — **after** maintainer approves issue |

---

## Labels (GitHub)

| Label | Use |
|-------|-----|
| `good first issue` | ≤1 day, clear acceptance criteria, no cluster |
| `help wanted` | Maintainer wants external help |
| `gsoc-2026` | Tied to [gsoc2026_agentic_rag.md](../../gsoc2026_agentic_rag.md) |
| `maintainer-only` | Agentic RAG core — discuss before PR |
| `area/tests` | pytest, coverage, smoke |
| `area/ci` | GitHub Actions |
| `area/docs` | README, runbooks |
| `area/pipelines` | KFP ingestion |
| `area/mcp-server` | MCP tools & retrieval |
| `area/frontend` | Website chat widget |
| `area/infra` | Terraform, manifests, Helm |
| `size/S` / `size/M` / `size/L` | Rough effort |

---

## How to pick up work

1. Find an issue labeled `good first issue` or `help wanted` (not `maintainer-only`).
2. Comment **“I’d like to work on this”** — maintainers will assign.
3. Fork → branch → `pip install -r requirements-test.txt` → `pytest -v` → `ruff check`.
4. PR with `Fixes #NNN` and DCO sign-off (`Signed-off-by:`).

Questions: Kubeflow Slack / community call / GitHub Discussions (if enabled).
