# Operations

Everything in the cluster is described in Git and deployed by Argo CD. This
section is for maintainers and anyone running their own copy.

| Page | For |
|---|---|
| [Environments and delivery](environments.md) | Production and staging in one cluster, and promoting a tested commit to production |
| [Argo CD](argocd.md) | The Applications and their sync policy, what lives where, deploys and rollbacks |
| [Observability](observability.md) | Tracing to Langfuse, the live and evaluation projects, data retention |
| [Ingestion pipelines](ingestion.md) | Loading docs, issues and code into Milvus with Kubeflow Pipelines |

## Ground rules

- **Git is the only writer.** No `kubectl edit` on managed namespaces. A change
  is a pull request; Argo CD shows anything else as drift.
- **Secrets never go in Git.** Charts reference Kubernetes Secrets by name. The
  list of Secrets to create is in [`deploy/argocd/README.md`](https://github.com/kubeflow/docs-agent/blob/main/deploy/argocd/README.md).
- **Production moves by promotion.** It runs one commit of `main`, changed only
  by a promotion PR; reverting that PR rolls back.
- **GPU releases and Langfuse are synced by hand.** `llm-runtime` and
  `llm-judge` restart a GPU pod and reload a model, and `langfuse` holds the
  trace data, so they never auto-sync.
- **CI never touches the cluster.** It tests every PR and, on `main`, builds and
  pushes images. Argo CD deploys.
