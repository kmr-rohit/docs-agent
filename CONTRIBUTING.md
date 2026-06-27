# Kubeflow Contributor Guide

Welcome to the Kubeflow project! We'd love to accept your patches and 
contributions to this project. Please read the 
[contributor's guide in our docs](https://www.kubeflow.org/docs/about/contributing/).

The contributor's guide:

* Shows you where to find the Developer Certificate of Origin (DCO) that you need 
  to agree to
* Helps you get started with your first contribution to Kubeflow
* Describes the pull request and review workflow in detail, including the
  OWNERS files and automated workflow tool

## docs-agent specific

* **[Contribution map](docs/community/CONTRIBUTION_MAP.md)** — what maintainers own (agentic RAG core) vs community-friendly work (tests, docs, CI, frontend UX, pipelines).
* **GSoC 2026 spec:** [gsoc2026_agentic_rag.md](gsoc2026_agentic_rag.md)
* **Local tests:** `pip install -r requirements-test.txt && pytest -v`
* **Lint:** `ruff check docs-agent-mcp/mcp-server tests docs-agent-mcp/pipelines`
* **MCP validation (operators):** see [AGENTS.md](AGENTS.md)

Pick issues labeled `good first issue` or `help wanted`. Do not start large PRs on issues labeled `maintainer-only` without maintainer agreement.
