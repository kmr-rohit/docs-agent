# Contributing

Thanks for helping. docs-agent follows the
[Kubeflow contributor guide](https://www.kubeflow.org/docs/about/contributing/):
sign your commits for the DCO (`git commit -s`), open a pull request against
`main`, and a reviewer approves it with `/lgtm` and `/approve`.

## Where to start

| You want to | Look at |
|---|---|
| Improve answers | Prompts in `docs-agent-mcp/charts/docs-agent/files/`, retrieval in `docs-agent-mcp/mcp-server/` |
| Improve the chat widget | `frontend/docs_scripts/chatbot.js`, `frontend/docs_styles/chatbot.css` |
| Add or fix ingestion | `docs-agent-mcp/pipelines/` |
| Deployment and charts | `docs-agent-mcp/charts/`, `deploy/` |
| Evals | `tests/eval/` |
| A first issue | [Known gaps](../operations/known-gaps.md), and issues from the nightly [triage report](../evaluation/feedback-triage.md) |

## Before you open a PR

```bash
pip install -r requirements-test.txt ruff==0.15.9
ruff check .
ruff format --check docs-agent-mcp/mcp-server tests
pytest -q
helm lint docs-agent-mcp/charts/docs-agent
```

CI runs the same checks, compiles the Kubeflow pipelines in their own
environment, and renders every chart.

## Opening issues

Use a title of the form `<type>(<area>): <summary>`, so the issue triage
workflow can attach source context and labels.

- Types: `bug`, `feat`, `chore`, `docs`, `test`, `ci`, `security`
- Areas: `mcp`, `pipelines`, `frontend`, `kagent`, `gateway`, `terraform`,
  `embeddings`, `ci`, `tests`, `docs`, `infra`

Example: `bug(mcp): search_kubeflow_docs returns Search failed on empty collection`.

## After your PR merges

Your change goes to staging automatically and is evaluated. See
[How a change reaches users](change-flow.md) for the full path and for how to
find your eval run.

## This documentation

The site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
from `docs/` and `mkdocs.yml`. Pages under Architecture, Operations and
Decisions include the READMEs that sit next to the code, so edit those files and
the site picks them up.

```bash
pip install "mkdocs-material==9.6.*"
mkdocs serve   # http://127.0.0.1:8000
```
