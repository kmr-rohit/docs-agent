# GitHub Actions secrets for OKE CD

> Full runbook (architecture, security, CI/CD steps, checklists): **[OKE-OPERATIONS.md](./OKE-OPERATIONS.md)**

Configure at **Settings → Secrets and variables → Actions**.

| Secret | Value | Required for |
|--------|-------|--------------|
| `GHCR_USERNAME` | Your GitHub username or org | Build/push MCP image to GHCR |
| `GHCR_TOKEN` | GitHub PAT with `write:packages`, `read:packages` | GHCR login + cluster pull secret |
| `OKE_CLUSTER_OCID` | OKE cluster OCID | kubeconfig generation |
| `OCI_USER_OCID` | OCI user OCID | OCI CLI in Actions |
| `OCI_TENANCY_OCID` | Tenancy OCID | OCI CLI |
| `OCI_REGION` | e.g. `us-ashburn-1` | OCI CLI |
| `OCI_FINGERPRINT` | API key fingerprint | OCI CLI |
| `OCI_KEY_FILE` | PEM private key (multiline) | OCI CLI |

**Do not commit tokens to the repository.** Rotate any token that was shared in chat or logs.

## CD flow (on push to `main`)

1. CI: compile MCP server + pytest
2. Build `ghcr.io/<GHCR_USERNAME>/mcp-kubeflow-docs:<sha>`
3. Push to GHCR
4. OCI kubeconfig → OKE
5. Deploy MCP, Kagent, KServe manifests
6. Smoke test KServe chat endpoint

## Manual deploy (without Actions)

```bash
export GHCR_USERNAME=<your-github-username>
export GHCR_TOKEN=...   # from GitHub PAT

docker login ghcr.io -u "$GHCR_USERNAME" -p "$GHCR_TOKEN"
docker build -t ghcr.io/$GHCR_USERNAME/mcp-kubeflow-docs:local \
  -f kagent-feast-mcp/mcp-server/Dockerfile kagent-feast-mcp/mcp-server
docker push ghcr.io/$GHCR_USERNAME/mcp-kubeflow-docs:local

kubectl create secret docker-registry ghcrsecret -n docs-agent \
  --docker-server=ghcr.io \
  --docker-username="$GHCR_USERNAME" \
  --docker-password="$GHCR_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f kagent-feast-mcp/manifests/mcp-server/mcp-server.yaml
kubectl set image deployment/mcp-kubeflow-docs \
  mcp-server=ghcr.io/$GHCR_USERNAME/mcp-kubeflow-docs:local -n docs-agent
```
