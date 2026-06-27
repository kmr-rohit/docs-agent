# Kubeflow docs-agent — Community meetup agenda

**Date:** 25 June 2026  
**Project:** [kubeflow/docs-agent](https://github.com/kubeflow/docs-agent) · GSoC 2026 · [KEP-867](https://github.com/kubeflow/community/issues/867)  
**Spec:** [gsoc2026_agentic_rag.md](../../gsoc2026_agentic_rag.md)

---

## Meeting goals (30–45 min)

1. Show **what shipped** in upstream (#210) and how the stack works today.
2. Clarify **maintainer vs community** ownership so people know what to pick up.
3. File **good-first issues** and get volunteers for this week.
4. Preview **near-term roadmap** (GitOps, Helm, security) without over-committing dates.

---

## Agenda

| Time | Topic | Owner |
|------|--------|-------|
| 0–5 min | Welcome, GSoC context, link to repo + spec | Rohit |
| 5–15 min | **Current architecture** (diagram + live path) | Rohit |
| 15–20 min | **What merged in #210** + validation story | Rohit |
| 20–28 min | **Community contributions** — what we want help with | Rohit |
| 28–35 min | **This week’s focus** + open issues to assign | All |
| 35–40 min | **Q&A** — cluster access, KFP, IDE MCP | All |
| 40–45 min | **Next meetup** / Slack follow-ups | Rohit |

---

## 1. Current architecture (share this slide)

### Runtime path

```
User (kubeflow.org chat OR Cursor/IDE)
        │
        ▼
┌─────────────────── docs-agent namespace ───────────────────┐
│  Kagent UI / Runner                                           │
│    ├── ModelConfig  →  KServe Qwen (OpenAI-compatible LLM)   │
│    └── Agent CRD    →  RemoteMCPServer                        │
│                              │                                │
│                              ▼                                │
│                    MCP server (:8000, FastMCP)                  │
│                    • search_kubeflow_docs                     │
│                    • search_github_issues                     │
│                    • search_kubeflow_code                       │
└──────────────────────────────┬───────────────────────────────┘
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
┌──────────────────── ml-infra ─────────────────┐   ┌──── kubeflow (KFP) ────┐
│ Milvus (vector DB)                             │   │ Docs pipeline          │
│ TEI embeddings (all-mpnet-base-v2, 768-dim)   │   │ Issues pipeline        │
│ Qwen-2.5-14B (GPU inference)                   │   │ Code pipeline          │
└────────────────────────────────────────────────┘   └──────────┬─────────────┘
                                                                │
                    Collections: kubeflow_docs │ issues_rag │ code_rag
```

### Design principle: thin MCP, smart agent

- **MCP** returns ranked snippets + citation URLs (no long answers).
- **Kagent + LLM** reasons, routes tools, and writes the final reply.
- **Pipelines** keep Milvus fresh; MCP does not crawl GitHub at query time.

### Namespaces

| Namespace | What lives there |
|-----------|------------------|
| `docs-agent` | Kagent, MCP Deployment, Agent CRDs |
| `ml-infra` | Milvus, TEI embeddings, Qwen KServe |
| `kubeflow` | Kubeflow Pipelines (ingestion jobs) |

### How we validate today

| Layer | Mechanism |
|-------|-----------|
| PRs | `tests.yml` — ruff, compile, pytest |
| Merge to main | Same + pipeline compile in `oke-cicd.yaml` |
| Operator fork (OKE) | Build MCP image → deploy → `smoke_tools.py` + Milvus healthy check |

---

## 2. What landed upstream (#210)

- **3-tool MCP server** — docs, GitHub issues, Kubeflow code/manifests.
- **TEI embeddings** — shared HTTP service (no PyTorch in MCP image).
- **KFP pipelines** — docs, issues, code ingestion paths.
- **Kagent wiring** — `ModelConfig`, `RemoteMCPServer`, `Agent` in manifests.
- **Unit tests** — MCP server, pipeline utils, issues/code helpers.
- **CI** — compile/lint/test on every PR; OKE deploy gated (`ENABLE_OKE_DEPLOY` on operator forks).
- **Official image** — `ghcr.io/kubeflow/mcp-kubeflow-docs:v0.1.0`.

### Known gaps (be honest)

| Gap | Impact |
|-----|--------|
| `code_rag` may be **empty** until code pipeline is run | `search_kubeflow_code` returns nothing |
| README / docs **drift** from MCP+Kagent reality | Confusing onboarding |
| No **auth / rate limits** on public paths yet | Phase 3 security |
| **Incremental pipeline** still old embedding path; not in CI compile | Drift risk |
| Istio policies duplicated (Terraform vs YAML) | Maintainer headache |
| GSoC “enterprise” items (evals, gateway, Argo CD) | **Not started** — design phase |

---

## 3. Ownership model (set expectations)

### Maintainer-owned — **agentic RAG core** (discuss before large PRs)

- MCP tool contracts and retrieval behavior
- Kagent `systemMessage` and tool-routing policy
- Semantic router / multi-agent design (LangGraph, ADK)
- RAG evals, golden datasets, quality strategy
- LLM gateway architecture (LiteLLM / kgateway, quotas)

### Community-friendly — **please help**

- Test suite expansion (unit + optional integration smoke)
- Documentation fixes and runbooks
- CI hygiene (terraform validate, kubeconform, dedupe workflows)
- Frontend UX (feedback, citations, accessibility) — not agent logic
- Pipeline ops (`code_rag` runbook, incremental → TEI migration)
- Infra packaging (Helm scaffold, manifest dedup)

**Labels:** `good first issue` · `help wanted` · `maintainer-only`  
**Guide:** [docs/community/CONTRIBUTION_MAP.md](./CONTRIBUTION_MAP.md)  
**Ready issues:** [docs/community/COMMUNITY_ISSUES.md](./COMMUNITY_ISSUES.md)

---

## 4. This week’s focus

### Rohit / maintainers (agentic RAG + ops)

| Priority | Task | Outcome |
|----------|------|---------|
| P0 | Merge community triage docs + issue templates (PR #19 or upstream equivalent) | Labels/issues live on GitHub |
| P0 | Run `./scripts/sync-github-labels.sh kubeflow/docs-agent` + file top 5 community issues | Volunteers can self-assign |
| P1 | Run **code pipeline** on cluster → populate `code_rag` | Demo `search_kubeflow_code` end-to-end |
| P1 | Draft **maintainer-only** design issues (router, LLM gateway, evals) | Boundaries clear for contributors |
| P2 | Spike: **Argo CD vs GHA** split (build in CI, sync in GitOps) | One-pager for next week |
| P2 | Review fork sync with `kubeflow/main` — upstream any fork-only test/doc fixes | Single source of truth |

### Community (pick one — comment on issue to claim)

| Issue theme | Label | Size |
|-------------|-------|------|
| Unit tests for `embeddings_client.py` | `good first issue` | S |
| Collection name parity test (`rag_collections` ↔ `utils`) | `good first issue` | S |
| Fix root README → MCP/Kagent as primary path | `good first issue` | M |
| `code_rag` ingestion runbook (`docs/INGESTION.md`) | `help wanted` | M |
| Frontend thumbs up/down (webhook stub) | `help wanted` | M |
| `terraform validate` + kubeconform in CI | `help wanted` | M |
| Compile `incremental-pipeline.py` in CI | `good first issue` | S |

### Not this week (preview only)

- Full Argo CD rollout
- LiteLLM / kgateway in production
- OAuth on public ingress
- LangGraph / ADK router implementation

---

## 5. Links to share in chat

| Resource | URL |
|----------|-----|
| Repo | https://github.com/kubeflow/docs-agent |
| GSoC spec | https://github.com/kubeflow/docs-agent/blob/main/gsoc2026_agentic_rag.md |
| Contribution map | https://github.com/kubeflow/docs-agent/blob/main/docs/community/CONTRIBUTION_MAP.md |
| Community issues backlog | https://github.com/kubeflow/docs-agent/blob/main/docs/community/COMMUNITY_ISSUES.md |
| Contributing | https://github.com/kubeflow/docs-agent/blob/main/CONTRIBUTING.md |
| Merged PR #210 | https://github.com/kubeflow/docs-agent/pull/210 |

---

## 6. Asks for the room

1. **Volunteers** — Who can take a `good first issue` this week? (tests or docs preferred.)
2. **Platform WG** — Any constraints on Argo CD / ingress on the OKE dogfood cluster?
3. **Feedback** — Is the 3-tool split (docs / issues / code) the right MCP surface for IDE users?
4. **Docs** — Biggest onboarding pain point in the README today?

---

## 7. After the meetup (action items)

- [ ] Post recording / notes in Kubeflow Slack
- [ ] Create GitHub labels + issues from `COMMUNITY_ISSUES.md`
- [ ] Assign at least 2 issues to volunteers
- [ ] Schedule follow-up in ~2 weeks (demo: `code_rag` + one merged community PR)

---

*Copy this file into the meetup doc or share the repo path: `docs/community/MEETUP_AGENDA_2026-06-25.md`*
