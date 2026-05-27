# OKE CD + Qwen/KServe Migration Plan

> **Status:** Implemented on `main` (PR #5).  
> **Runbook:** see **[OKE-OPERATIONS.md](./OKE-OPERATIONS.md)** for architecture, security, CI/CD, scripts, checklist, and next steps.

This file retains the original planning notes for context.

---

## Original planning context

**Cluster:** OKE `us-ashburn-1` · namespace `docs-agent`

### Implemented on main

- GHCR CD workflow (`oke-cicd.yaml`)
- KServe Qwen on GPU with `v0.17.0-gpu`
- Kagent → KServe via `qwen-llm` internal service
- GPU LVM expand job
- 3-tool MCP (docs, issues, code) + pytest CI
- Code/issues ingestion pipelines

See [OKE-OPERATIONS.md](./OKE-OPERATIONS.md) for details.

---

## Original state (historical)

| Component | Status at planning time |
|-----------|-------------------------|
| **OKE nodes** | 2× CPU + 1× GPU |
| **Kagent** | UI LB public; Groq LLM |
| **MCP** | GHCR image |
| **KServe** | Stopped ISVC |

*(Remaining sections below are historical planning notes.)*
