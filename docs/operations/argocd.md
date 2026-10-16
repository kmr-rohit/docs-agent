# Argo CD

Argo CD deploys everything in the cluster from Git and shows any difference
between Git and what is running.

## Applications

The root Application `flo-apps` (`deploy/argocd/root.yaml`) applies every
Application in `deploy/argocd/apps/` from `main`, so a new Application or a
merged promotion takes effect without anyone running `kubectl`.

| Application | Source | Tracks | Namespace | Sync |
|---|---|---|---|---|
| `flo-apps` | `deploy/argocd/apps/` | `main` | `argocd` | Automatic |
| `docs-agent` | `docs-agent-mcp/charts/docs-agent` + `deploy/envs/prod/docs-agent.yaml` | promoted commit | `docs-agent` | Automatic |
| `gateway-guardrails` | `docs-agent-mcp/charts/gateway-guardrails` + `deploy/envs/prod/` | promoted commit | `docs-agent` | Automatic |
| `docs-agent-staging` | the docs-agent chart + `deploy/envs/staging/docs-agent.yaml` | `main` | `docs-agent-staging` | Automatic, self-heal, prune |
| `otel-collector` | `deploy/platform/otel-collector` | `main` | `docs-agent` | Automatic |
| `langfuse-retention` | `deploy/platform/langfuse-retention` | `main` | `langfuse` | Automatic |
| `llm-runtime` | `docs-agent-mcp/charts/llm-runtime` | `main` | `ml-infra` | Manual (GPU) |
| `llm-judge` | the same chart, Qwen profile | `main` | `ml-infra` | Manual (GPU) |
| `langfuse` | Langfuse Helm chart + `deploy/envs/prod/langfuse/values.yaml` | `main` | `langfuse` | Manual (stateful) |

Chart Applications have two sources: the chart, and the values file from the
same repository (referenced as `$values`). Both production sources are pinned to
the promoted commit with a `# prod-revision` comment, which
[`deploy/promote.sh`](environments.md#from-merge-to-production) moves. Image tags
come from the synced commit (`$ARGOCD_APP_REVISION_SHORT`), not from values files.

`llm-runtime` and `llm-judge` stay manual because a sync restarts a GPU pod and
reloads the model. `langfuse` stays manual because it holds the trace data;
review its diff before syncing.

## What lives where

| Kind of data | Where | Changed by |
|---|---|---|
| Charts and Applications | `docs-agent-mcp/charts/`, `deploy/argocd/` | Pull request |
| Environment values (hosts, CORS origins) | `deploy/envs/<env>/` | Pull request |
| Production revision | `# prod-revision` lines in `deploy/argocd/apps/` | Promotion PR (`deploy/promote.sh`) |
| Secrets (API keys, Langfuse keys, passwords) | Kubernetes Secrets, referenced by name | Cluster admins |

Values files are public, so nothing secret or personal goes in them.

## Day-to-day

- **See what would change:** open the Application, then **App Diff**, or run
  `argocd app diff <app> --grpc-web` with the read-only account.
- **Deploy production:** open a promotion PR with `deploy/promote.sh`; merging it
  deploys.
- **Roll back:** revert the promotion PR. Avoid **History and Rollback** in the
  UI: it pauses automatic sync and leaves Git and the cluster disagreeing.
- **Drift:** a hand edit on an automatically synced Application is reverted
  (self-heal). On the manual ones it shows as OutOfSync until a sync or a change
  in Git resolves it.
- **GPU models and Langfuse:** review the diff, then click **Sync**.

## Access

The UI is behind the Istio gateway with HTTPS. There are two local accounts:
`admin` for maintainers and `viewer`, which is read-only (it can see every
Application, diff and log, but cannot sync or delete). Anonymous access and
`exec` are disabled.

Install and account setup: [`deploy/argocd/README.md`](https://github.com/kubeflow/docs-agent/blob/main/deploy/argocd/README.md).
