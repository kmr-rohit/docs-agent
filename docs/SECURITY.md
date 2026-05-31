# Security model (docs-agent stack)

## Secrets (never in Git)

| Secret | Where stored | Consumed by |
|--------|----------------|-------------|
| `MILVUS_PASSWORD` | K8s `mcp-server-secret` (docs-agent), `milvus-auth` (kubeflow) | MCP Deployment, KFP store steps |
| `Github_Pat` | K8s `github-pat` (profile / kubeflow ns) | Pipeline download steps |
| OCI / GHCR | GitHub Environment `kubeflow` | `oke-cicd.yaml` only |

Do not put passwords or PATs in ConfigMaps, pipeline source, or Terraform variables committed to the repo.

## Network (Istio)

Terraform `istio_policies.tf` uses explicit **ALLOW** policies for:

- MCP → Milvus, embeddings-service
- KFP (`kubeflow` ns) → Milvus, embeddings-service
- Kagent → MCP, Qwen LLM

Assume a deny-by-default mesh elsewhere. After `terraform apply`, verify AuthorizationPolicies exist in `ml-infra` and `docs-agent`.

## MCP server

- Query embeddings via in-cluster TEI URL only (no bundled model download).
- Milvus credentials from env/secret; server fails fast if `MILVUS_PASSWORD` is unset.
- Tool filters validate user input before Milvus expressions (`_safe_filter_value`).
- MCP Service is `ClusterIP` only (not public).

## Pipelines

- GitHub tokens read from K8s secrets via `k8s.use_secret_as_env`, not pipeline parameters in Git.
- Embeddings and Milvus are cluster-internal DNS names.

## Operations checklist

1. Rotate `mcp-server-secret` and `milvus-auth` together when Milvus password changes.
2. Restrict who can `kubectl exec` into `docs-agent` / `ml-infra`.
3. Use private GHCR + `imagePullSecrets` for MCP (already in CD).
4. Re-run ingestion after embedding or schema changes; do not expose Milvus port publicly.
