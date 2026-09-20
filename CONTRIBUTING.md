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

## Opening issues

Use a title of the form `<type>(<area>): <summary>` so the issue triage workflow
can attach source context and labels.

- Types: `bug`, `feat`, `chore`, `docs`, `test`, `ci`, `security`
- Areas: `mcp`, `pipelines`, `frontend`, `kagent`, `gateway`, `terraform`,
  `embeddings`, `ci`, `tests`, `docs`, `infra`

Example: `bug(mcp): search_kubeflow_docs returns Search failed on empty collection`.

The analyzer reads [`docs/agents/architecture.md`](docs/agents/architecture.md)
and the files listed for that area in [`docs/agents/area-map.json`](docs/agents/area-map.json),
then applies `kind/*` and `area/*` labels (`area/mcp`, `area/pipelines`, …).
See [`docs/agents/README.md`](docs/agents/README.md).
