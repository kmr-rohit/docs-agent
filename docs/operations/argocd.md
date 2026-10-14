# Argo CD

Argo CD deploys everything in the cluster from Git and shows any difference
between Git and what is running.

## Applications

| Application | Source | Namespace | Sync |
|---|---|---|---|
| `docs-agent` | `docs-agent-mcp/charts/docs-agent` + `deploy/envs/prod/docs-agent.yaml` | `docs-agent` | Manual |
| `docs-agent-staging` | the same chart + `deploy/envs/staging/docs-agent.yaml` | `docs-agent-staging` | Automatic, self-heal, prune |
| `gateway-guardrails` | `docs-agent-mcp/charts/gateway-guardrails` + `deploy/envs/prod/` | `docs-agent` | Manual |
| `llm-runtime` | `docs-agent-mcp/charts/llm-runtime` | `ml-infra` | Manual (GPU) |
| `llm-judge` | the same chart, Qwen profile | `ml-infra` | Manual (GPU) |
| `otel-collector` | `deploy/platform/otel-collector` | `docs-agent` | Manual |
| `langfuse` | Langfuse Helm chart + `deploy/envs/prod/langfuse/values.yaml` | `langfuse` | Manual |

Each Application has two sources: the chart, and the values file from the same
repository (referenced as `$values`). The Application manifests are in
`deploy/argocd/apps/`.

## What lives where

| Kind of data | Where | Changed by |
|---|---|---|
| Charts and Applications | `docs-agent-mcp/charts/`, `deploy/argocd/` | Pull request |
| Environment values (hosts, CORS origins, image tags) | `deploy/envs/<env>/` | Pull request |
| Secrets (API keys, Langfuse keys, passwords) | Kubernetes Secrets, referenced by name | Cluster admins |

Values files are public, so nothing secret or personal goes in them.

## Day-to-day

- **See what would change:** open the Application, then **App Diff**.
- **Deploy production:** after the change is on `main` and the staging eval passed,
  click **Sync** on `docs-agent` (or `gateway-guardrails`).
- **Roll back:** revert the pull request and sync, or use **History and
  Rollback**, then revert in Git so the two agree.
- **Drift:** a hand edit shows as OutOfSync. Staging reverts it automatically;
  in production, put the change in Git or sync to remove it.

## Access

The UI is behind the Istio gateway with HTTPS. There are two local accounts:
`admin` for maintainers and `viewer`, which is read-only (it can see every
Application, diff and log, but cannot sync or delete). Anonymous access and
`exec` are disabled.

Install and account setup: [`deploy/argocd/README.md`](https://github.com/kubeflow/docs-agent/blob/main/deploy/argocd/README.md).
