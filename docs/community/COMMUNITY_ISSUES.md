# Community issues backlog (community call)

Copy these into GitHub Issues on `kubeflow/docs-agent` (or your fork). Run labels first:

```bash
./scripts/sync-github-labels.sh kubeflow/docs-agent
```

---

## Issue 1 — Unit tests for `embeddings_client.py`

**Title:** `test: add unit tests for embeddings_client.py`

**Labels:** `good first issue`, `help wanted`, `area/tests`, `size/S`, `gsoc-2026`

**Body:**

### Goal
Add pytest coverage for `docs-agent-mcp/mcp-server/embeddings_client.py` (currently untested; logic mirrors `pipelines/utils.py` TEI client).

### Acceptance criteria
- [ ] Tests for empty input, single text, batching (`batch_size`), truncation (`MAX_TEI_INPUT_CHARS`)
- [ ] Mock HTTP with `unittest.mock` or `responses` — no live TEI cluster
- [ ] Error cases: missing `EMBEDDINGS_URL`, malformed JSON response
- [ ] `pytest -v` passes

### References
- `tests/test_pipeline_utils.py::TestEmbedTexts`
- GSoC checklist #5 (retry logic is a follow-up issue)

**Not in scope:** changing embedding model or MCP tool behavior.

---

## Issue 2 — Collection name parity test

**Title:** `test: assert rag_collections.py matches pipelines/utils.py`

**Labels:** `good first issue`, `area/tests`, `size/S`

**Body:**

### Goal
Prevent drift between MCP and pipeline Milvus collection constants.

### Acceptance criteria
- [ ] New test imports `DOCS_COLLECTION`, `ISSUES_COLLECTION`, `CODE_COLLECTION` from both:
  - `docs-agent-mcp/mcp-server/rag_collections.py`
  - `docs-agent-mcp/pipelines/utils.py`
- [ ] Assert all three pairs are equal (`kubeflow_docs`, `issues_rag`, `code_rag`)

---

## Issue 3 — kubeconform manifest validation in CI

**Title:** `ci: validate docs-agent-mcp/manifests with kubeconform`

**Labels:** `help wanted`, `area/ci`, `area/infra`, `size/M`, `gsoc-2026`

**Body:**

### Goal
Catch invalid Kubernetes YAML before deploy.

### Acceptance criteria
- [ ] Add CI job (in `tests.yml` or new workflow) running kubeconform on `docs-agent-mcp/manifests/**/*.yaml`
- [ ] Document any schemas/CRDs needed (KServe InferenceService, kagent CRDs) — skip or use fixture schemas if required
- [ ] Job passes on `main`

### References
- GSoC Phase 3: idempotent deployments

---

## Issue 4 — `terraform validate` in CI

**Title:** `ci: terraform fmt and validate for docs-agent-mcp/terraform`

**Labels:** `help wanted`, `area/ci`, `area/infra`, `size/S`

**Body:**

### Goal
Static validation of Terraform without OKE credentials.

### Acceptance criteria
- [ ] Job runs `terraform fmt -check` and `terraform validate` in `docs-agent-mcp/terraform/`
- [ ] Use `-backend=false` or init without remote state
- [ ] Document in `docs-agent-mcp/terraform/README.md` (create if missing)

---

## Issue 5 — Fix README architecture drift

**Title:** `docs: align root README with MCP/Kagent primary path`

**Labels:** `good first issue`, `area/docs`, `size/M`, `gsoc-2026`

**Body:**

### Problem
Root `README.md` still emphasizes legacy FastAPI + Llama KServe. Post-#210, the primary path is `docs-agent-mcp/` (MCP + Kagent + Milvus + TEI).

### Acceptance criteria
- [ ] Add prominent link to `docs-agent-mcp/README.md` as primary deployment guide
- [ ] Fix wrong paths (`server/app.py` → `legacy/server/app.py`)
- [ ] Document env vars for MCP (`MILVUS_URI`, `EMBEDDINGS_URL`) vs legacy
- [ ] Add "Running tests" section (`requirements-test.txt`, `pytest -v`)
- [ ] Link to `gsoc2026_agentic_rag.md` and `docs/community/CONTRIBUTION_MAP.md`

**Not in scope:** rewriting agent prompts or MCP tools.

---

## Issue 6 — Remove Feast contradiction in docs-agent-mcp README

**Title:** `docs: fix Feast reference in docs-agent-mcp/README.md`

**Labels:** `good first issue`, `area/docs`, `size/S`

**Body:**

### Problem
`docs-agent-mcp/README.md` says pipelines register features in Feast, but the stack uses direct Milvus (no Feast).

### Acceptance criteria
- [ ] Update pipeline step description to pymilvus → Milvus collections
- [ ] Reconcile Milvus install section (Helm manual vs Terraform operator) with a short "choose one" note

---

## Issue 7 — Migrate incremental pipeline to TEI

**Title:** `pipelines: migrate incremental-pipeline.py to TEI HTTP embeddings`

**Labels:** `help wanted`, `area/pipelines`, `size/L`, `gsoc-2026`

**Body:**

### Problem
`incremental-pipeline.py` still uses sentence-transformers/GPU patterns. Other pipelines use TEI via `utils.embed_texts`. CI does not compile incremental pipeline today.

### Acceptance criteria
- [ ] Incremental pipeline uses TEI HTTP (same as `kubeflow-pipeline.py`)
- [ ] `python incremental-pipeline.py` compiles in CI (`oke-cicd.yaml`)
- [ ] Update `docs-agent-mcp/pipelines/README.md` embedding section
- [ ] Tests for any extracted pure functions (delete detection logic)

**Maintainer note:** does not change MCP retrieval or agent routing.

---

## Issue 8 — `code_rag` ingestion runbook

**Title:** `docs: runbook for populating code_rag collection`

**Labels:** `help wanted`, `area/docs`, `area/pipelines`, `size/M`

**Body:**

### Problem
`code_rag` Milvus collection may exist with **0 entities** until the code ingestion pipeline is run (see `AGENTS.md`).

### Acceptance criteria
- [ ] New `docs/INGESTION.md` (or section in pipelines README) covering:
  - When to run docs vs issues vs code pipelines
  - `code-pipeline.py` parameters and default repos (`kubeflow/manifests`, etc.)
  - KFP secret for `github_token`
  - How to verify with MCP `search_kubeflow_code`
- [ ] Link from `CONTRIBUTING.md`

---

## Issue 9 — Frontend feedback (thumbs up/down)

**Title:** `frontend: add thumbs up/down feedback with webhook stub`

**Labels:** `help wanted`, `area/frontend`, `size/M`, `gsoc-2026`

**Body:**

### Goal
GSoC spec §5.3 / checklist #6 — start feedback loop for golden dataset.

### Acceptance criteria
- [ ] Thumbs up/down on assistant messages in `frontend/docs_scripts/chatbot.js`
- [ ] POST to configurable webhook URL (document in `frontend/README.md`; default no-op or console)
- [ ] Payload includes: user message, assistant message, timestamp, vote
- [ ] Accessible buttons (aria-labels)

**Not in scope:** backend storage design, Katib integration, agent changes.

---

## Issue 10 — Frontend JS unit tests

**Title:** `test: unit tests for chatbot localStorage helpers`

**Labels:** `good first issue`, `area/frontend`, `area/tests`, `size/S`

**Body:**

### Goal
Test pure JS helpers without a browser.

### Acceptance criteria
- [ ] Tests for chat persistence helpers in `chatbot.js` (extract if needed)
- [ ] Use `node:test` or Vitest — document run command in `frontend/README.md`
- [ ] CI job optional but appreciated

---

## Issue 11 — MCP TEI retry with backoff

**Title:** `mcp: add retry/backoff to embeddings_client on transient failures`

**Labels:** `help wanted`, `area/mcp-server`, `size/S`, `gsoc-2026`

**Body:**

### Goal
GSoC checklist #5 — resilient tool calls.

### Acceptance criteria
- [ ] Retry on 5xx / connection errors with exponential backoff + jitter
- [ ] Configurable max retries via env (default 3)
- [ ] Unit tests with mocked failures
- [ ] No change to tool schemas or ranking logic

**Maintainer review:** small behavioral change to MCP server — OK for community with tests.

---

## Issue 12 — Compile incremental pipeline in CI

**Title:** `ci: compile incremental-pipeline.py in oke-cicd test job`

**Labels:** `good first issue`, `area/ci`, `size/S`

**Body:**

### Goal
Catch KFP DSL breakage early.

### Acceptance criteria
- [ ] Add `python incremental-pipeline.py` to compile step in `.github/workflows/oke-cicd.yaml`
- [ ] Fix or file follow-up if compile fails due to TEI migration

---

## Issue 13 — Deduplicate Istio AuthorizationPolicy sources

**Title:** `infra: single source of truth for Istio allow policies`

**Labels:** `help wanted`, `area/infra`, `size/M`

**Body:**

### Problem
Istio policies exist in both `docs-agent-mcp/terraform/istio_policies.tf` (9 policies) and `docs-agent-mcp/manifests/istio/` (4 policies).

### Acceptance criteria
- [ ] Pick one source (recommend: manifests + Helm chart later)
- [ ] Remove or generate duplicates
- [ ] Document in `docs/community/CONTRIBUTION_MAP.md` or terraform README

**Discuss with maintainers** before large Terraform deletes.

---

## Issue 14 — Helm chart scaffold (app layer only)

**Title:** `infra: Helm chart scaffold for MCP server and kagent CRs`

**Labels:** `help wanted`, `area/infra`, `size/L`, `gsoc-2026`

**Body:**

### Goal
GSoC spec mentions `deployments/helm/docs-agent/` — package existing YAML.

### Acceptance criteria
- [ ] Chart templates for `manifests/mcp-server/mcp-server.yaml` and `manifests/kagent/setup.yaml`
- [ ] `values.yaml` for image, namespace, collection names
- [ ] `helm template` documented; no requirement to replace Terraform yet

**Not in scope:** replacing cluster bootstrap (Istio/KServe/Knative).

---

## Maintainer-only examples (do NOT label good first issue)

Create these to set boundaries; label `maintainer-only`:

| Title | Why |
|-------|-----|
| Design: semantic router (docs vs code vs issues) | Agentic RAG core |
| Design: LLM gateway (LiteLLM vs kgateway) + quota model | Security architecture |
| Evolve Kagent systemMessage tool-routing policy | Prompt/agent behavior |
| RAG evaluation framework (RAGAS / golden dataset schema) | Quality strategy |
| Argo CD app-of-apps layout for OKE dogfood cluster | Operator decision |

---

## Suggested community call talking points

1. **#210 merged** — 3-tool MCP, TEI, pipelines, tests landed upstream.
2. **You own** agent brain: routing, MCP contracts, evals, gateway design.
3. **We want help** on tests, docs truth, CI hygiene, frontend UX, pipeline ops, packaging.
4. **Pick issues** labeled `good first issue` — comment to get assigned.
5. **DCO** required on all PRs (`git commit -s`).
