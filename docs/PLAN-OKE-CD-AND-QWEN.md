# OKE CD + Qwen/KServe Migration Plan

> **Status:** Largely implemented on branch `plan/cd-and-qwen-kserve`.  
> **Current runbook:** see **[OKE-OPERATIONS.md](./OKE-OPERATIONS.md)** for architecture, security, CI/CD, scripts, checklist, and next steps.

This file retains the original planning notes for context.

---

## Original planning context

**Branch:** `plan/cd-and-qwen-kserve` (from `SanthoshToorpu/docs-agent:test-pr`)  
**Cluster:** `context-cp5iuhfpl7a` · OKE `us-ashburn-1` · namespace `docs-agent`  
**Maintainers:** You + Santosh

### Implemented since this plan was written

- GHCR CD workflow (replaced OCIR plan)
- KServe Qwen on GPU with `v0.17.0-gpu`
- Kagent → KServe via `qwen-llm` internal service
- GPU LVM expand job
- Kagent ModelConfig cutover from Groq
- Milvus ingestion runbook

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
