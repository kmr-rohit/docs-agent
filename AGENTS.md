# AGENTS.md

## Cursor Cloud specific instructions

### Primary validation target

This repo's validation target is the **Kagent MCP deployment path**, not local FastAPI/WebSocket server startup. The important runtime path is:

```
User / Kagent UI → Kagent Agent CRD → MCP server (docs-agent-mcp/mcp-server/) → Milvus collections → LLM backend (Groq / KServe)
```

The MCP server at `docs-agent-mcp/mcp-server/server.py` is the main thing to validate. The legacy servers under `legacy/server/` and `legacy/server-https/` only need basic syntax/import sanity checks.

### MCP server overview

| Tool | Milvus Collection | Status |
|---|---|---|
| `search_kubeflow_docs` | `kubeflow_docs` | Populated and loaded (matches upstream #207) |
| `search_github_issues` | `issues_rag` | Populated and loaded |
| `search_kubeflow_code` | `code_rag` | Collection may exist with **0** entities until the code ingestion pipeline is rerun |

Milvus collection names use **underscores** only (hyphens are invalid in Milvus). Docs collection stays `kubeflow_docs` (upstream #207); issues/code use `issues_rag` and `code_rag`. Constants: `docs-agent-mcp/pipelines/utils.py` and `docs-agent-mcp/mcp-server/rag_collections.py`.

The server uses FastMCP with `streamable-http` transport on port 8000. The MCP endpoint is at `/mcp`. The checked-in `docs-agent-mcp/mcp-server/server.py` may lag the cluster image (for example, additional tools); validate with `tools/list` against the live deployment.

### Code change → validation loop

When code changes are made to the MCP server:

1. **Build the Docker image** (context is `docs-agent-mcp/`, Dockerfile at `docs-agent-mcp/mcp-server/Dockerfile`):
   ```bash
   docker build -t ghcr.io/kmr-rohit/mcp-kubeflow-docs:<tag> \
     -f docs-agent-mcp/mcp-server/Dockerfile docs-agent-mcp/
   ```
2. **Push to GHCR**:
   ```bash
   docker push ghcr.io/kmr-rohit/mcp-kubeflow-docs:<tag>
   ```
3. **Deploy to cluster**:
   ```bash
   kubectl set image deployment/mcp-kubeflow-docs \
     mcp-server=ghcr.io/kmr-rohit/mcp-kubeflow-docs:<tag> \
     -n docs-agent
   kubectl rollout status deployment/mcp-kubeflow-docs -n docs-agent
   ```
4. **Validate MCP tools** against live Milvus-backed cluster service.

**Kagent CRDs vs the container image:** `RemoteMCPServer` and `Agent` in `docs-agent-mcp/manifests/kagent/setup.yaml` point at the **Kubernetes Service URL** (for example `http://mcp-kubeflow-docs.docs-agent.svc.cluster.local:8000/mcp`). When you only change MCP server code, you update the **`Deployment` image** (`mcp-kubeflow-docs`); you do **not** need to edit those CRDs unless you rename the Service, change the port, or change the path. If you reuse a fixed tag like `latest` and push a new digest, ensure `imagePullPolicy: Always` or bump the tag so nodes pull the new image.

The CI workflow at `.github/workflows/build-mcp-image.yml` automatically builds and pushes to GHCR on pushes to `main` that touch `docs-agent-mcp/mcp-server/**`.

### Cursor Cloud secrets: autonomous `docker push` + OKE validation

GitHub Container Registry (**ghcr.io**) does **not** use “Docker Hub” credentials. A Cloud Agent needs:

| Secret (suggested name) | Required? | Purpose |
|---|---|---|
| `KUBECONFIG_CONTENT` | Yes (already) | `kubectl` access to OKE |
| `OCI_CONFIG_B64` | Yes (already) | `oci` exec auth for kubeconfig |
| `OCI_API_KEY_B64` | Yes (already) | Private key for OCI profile |
| `GHCR_USERNAME` | For push from VM | GitHub username that has **push** rights to `ghcr.io/kmr-rohit/mcp-kubeflow-docs` (often `kmr-rohit`) |
| `GHCR_TOKEN` | For push from VM | **GitHub Personal Access Token** used as the registry password. Classic PAT: scopes **`write:packages`** and **`read:packages`** (add **`repo`** if the GitHub repo is private). Fine-grained PAT: grant **Packages: Read and write** for the relevant org/user and repository. |

**Login before `docker push`:**

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USERNAME" --password-stdin
```

**Optional (alternative to pushing from the VM):** store a **`GH_TOKEN`** (PAT with `repo` + `workflow` as needed) if the agent should call `gh workflow run` to trigger `.github/workflows/build-mcp-image.yml`; GitHub Actions then pushes using `GITHUB_TOKEN` inside the workflow (no VM push secret needed for the push itself, only for triggering).

**If the GHCR package is private:** the OKE cluster must pull with an **`imagePullSecret`** in `docs-agent` (create once with `kubectl create secret docker-registry ...` using the same or a read-only PAT). That is cluster configuration, not necessarily a Cursor secret, unless you want the agent to create/rotate it.

**Feast / gRPC (when you switch the MCP server to Feast online serving over gRPC):** add secrets that match whatever the new code reads, for example:

| Secret (examples) | Purpose |
|---|---|
| `FEAST_SERVING_HOST` / `FEAST_SERVING_PORT` or a single `FEAST_GRPC_TARGET` | Online store / serving endpoint (`host:port`) |
| `FEAST_API_KEY` or `FEAST_AUTH_TOKEN` | If Feast Server or a proxy requires auth |
| `FEAST_TLS_CA_B64` | PEM of CA for TLS verify, if not using public CAs |

Exact names should follow the env vars you implement in `docs-agent-mcp/mcp-server/`. Today’s MCP image talks to **Milvus over HTTP** (`MILVUS_URI`); a Feast-gRPC path is a **code + deploy env** change, not something the registry secrets replace.

### Local Docker validation (without cluster)

You can build and run the MCP container locally to verify the image builds and the FastMCP server starts:

```bash
# Ensure dockerd is running (see "Starting Docker" below)
docker build -t mcp-kubeflow-docs:local-test \
  -f docs-agent-mcp/mcp-server/Dockerfile docs-agent-mcp/
docker run -d --name mcp-test -p 8003:8000 mcp-kubeflow-docs:local-test

# Test MCP initialize
curl -s -X POST http://localhost:8003/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'

# Capture session ID from response headers, then list tools:
curl -s -X POST http://localhost:8003/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: <session-id>" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
```

Tool calls (e.g. `search_kubeflow_docs`) will fail without a reachable Milvus instance, but `initialize` and `tools/list` should succeed.

### Starting Docker in the Cloud Agent VM

Docker requires special configuration in the Cloud Agent VM (nested container in Firecracker):

```bash
sudo dockerd &  # Start in background; fuse-overlayfs + iptables-legacy already configured by update script
```

Wait ~5 seconds for dockerd to start, then verify with `sudo docker info`.

### OCI / kubectl cluster access

The OKE cluster uses `oci ce cluster generate-token` as the kubectl exec credential plugin. Three secrets are required:

| Secret | Description |
|---|---|
| `KUBECONFIG_CONTENT` | Raw kubeconfig YAML (prefer real newlines; if the secret is a single collapsed line, see **Kubeconfig repair** below) |
| `OCI_CONFIG_B64` | Base64-encoded `~/.oci/config` |
| `OCI_API_KEY_B64` | Base64-encoded OCI API private key PEM |

#### Recommended setup script (strict)

Use `set -euo pipefail` and write files exactly as provided (this matches the project’s documented flow):

```bash
set -euo pipefail

: "${OCI_CONFIG_B64:?missing OCI_CONFIG_B64}"
: "${OCI_API_KEY_B64:?missing OCI_API_KEY_B64}"
: "${KUBECONFIG_CONTENT:?missing KUBECONFIG_CONTENT}"

mkdir -p ~/.oci ~/.kube

printf '%s' "$OCI_CONFIG_B64" | base64 -d > ~/.oci/config
printf '%s' "$OCI_API_KEY_B64" | base64 -d > ~/.oci/oci_api_key.pem
printf '%s' "$KUBECONFIG_CONTENT" > ~/.kube/config

chmod 600 ~/.oci/config ~/.oci/oci_api_key.pem ~/.kube/config

# oci-cli lives in the repo venv in Cloud Agent VMs; add to PATH for this shell
export PATH="/workspace/.venv/bin:$PATH"

oci --profile KUBEFLOW-GSCO26 os ns get

kubectl config current-context
kubectl get pods -n docs-agent
kubectl get svc -n docs-agent
```

#### Post-setup fixes (common in Cloud Agent VMs)

1. **`key_file` path** — decoded `~/.oci/config` may still point at a laptop path (for example `/Users/.../.oci/oci_api_key.pem`). Point it at the VM key:
   ```bash
   sed -i 's|^key_file=.*|key_file=/home/ubuntu/.oci/oci_api_key.pem|' ~/.oci/config
   chmod 600 ~/.oci/config ~/.oci/oci_api_key.pem
   ```

2. **Named OCI profile only** — if `~/.oci/config` has only `[KUBEFLOW-GSCO26]` (no `[DEFAULT]`), `kubectl`’s bare `oci ...` exec will fail with “config invalid / missing DEFAULT”. Either duplicate that profile as `[DEFAULT]`, **or** add this under the kubeconfig user’s `exec:` block (same indentation as `env:`):
   ```yaml
   env:
   - name: OCI_CLI_PROFILE
     value: KUBEFLOW-GSCO26
   ```
   (Keep `interactiveMode` / `provideClusterInfo` if your kubeconfig already has them.)

3. **Fingerprint** — the private key must match the fingerprint in the profile you use:
   ```bash
   openssl pkey -in ~/.oci/oci_api_key.pem -pubout -outform DER 2>/dev/null | openssl md5 -c | awk '{print $2}'
   grep fingerprint ~/.oci/config
   ```

#### Kubeconfig repair (collapsed single-line YAML)

If `kubectl` errors with `yaml: mapping values are not allowed in this context`, `KUBECONFIG_CONTENT` is often one long line (spaces instead of newlines). Re-save the kubeconfig from the OCI console with normal newlines, **or** repair locally (the Cloud Agent previously used a small Python normalizer keyed off OKE’s usual `apiVersion` / `clusters` / `exec` structure). After repair, re-run `chmod 600 ~/.kube/config`.

### Cluster inspection commands

```bash
kubectl get all -n docs-agent
kubectl logs -n docs-agent deploy/mcp-kubeflow-docs --tail=100 -f
kubectl get svc -n docs-agent -o wide
kubectl get svc -n docs-agent | grep -i milvus
```

### Milvus validation from inside the cluster

Milvus (operator) in `ml-infra` is typically `milvus-milvus` (`kubectl get svc -n ml-infra | grep -i milvus`).

`kubectl run ... --rm -it` fails in non-interactive shells (“`--rm` should only be used for attached containers`”). Prefer creating a short-lived pod, waiting for completion, then reading logs:

```bash
kubectl delete pod milvus-check -n docs-agent --ignore-not-found=true
kubectl run milvus-check --restart=Never -n docs-agent \
  --image=python:3.11-slim \
  -- bash -c 'pip install -q pymilvus && python -c "
from pymilvus import connections, utility, Collection
connections.connect(
    \"default\",
    host=\"milvus-milvus.ml-infra.svc.cluster.local\",
    port=\"19530\",
    user=\"root\",
    password=\"Milvus\",
)
for name in [\"kubeflow_docs\", \"issues_rag\", \"code_rag\"]:
    print(\"collection\", name, \"exists=\", utility.has_collection(name))
    if utility.has_collection(name):
        c = Collection(name)
        print(\"entities=\", c.num_entities)
"'
kubectl wait --for=condition=Ready pod/milvus-check -n docs-agent --timeout=120s
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/milvus-check -n docs-agent --timeout=300s
kubectl logs milvus-check -n docs-agent
kubectl delete pod milvus-check -n docs-agent
```

Example output observed in this environment: `kubeflow_docs` and `issues_rag` populated; `code_rag` exists with **0** entities until the code ingestion pipeline is rerun.

### MCP “hello world” (live Milvus + `search_kubeflow_docs`)

End-to-end check against the in-cluster MCP service:

```bash
kubectl port-forward -n docs-agent svc/mcp-kubeflow-docs 18004:8000 &
# Then call Streamable HTTP MCP: initialize → notifications/initialized → tools/call
```

Use `Accept: application/json, text/event-stream`, read `mcp-session-id` / `Mcp-Session-Id` from the `initialize` response headers, then `POST` `tools/call` with `name: search_kubeflow_docs` and `arguments: {"query": "what is kubeflow", "top_k": 3}`. A healthy stack returns ranked doc snippets with `Source:` URLs from `kubeflow.org`.

### Namespace boundaries

- **Safe target namespace:** `docs-agent`
- KFP workflows usually live in `user`
- **Never** make destructive changes in shared control-plane namespaces: `kubeflow`, `kubeflow-system`, `istio-system`, `knative-serving`
- **Never** drain/delete shared cluster nodes

### Python dependencies (local dev)

All components share a virtualenv at `/workspace/.venv`. Activate with `source /workspace/.venv/bin/activate`. Dependencies are installed from:
- `server/requirements.txt`
- `server-https/requirements.txt`
- `docs-agent-mcp/mcp-server/requirements.txt`

### Environment variables (MCP server)

| Variable | Default | Description |
|---|---|---|
| `MILVUS_URI` | `http://milvus-milvus.ml-infra.svc.cluster.local:19530` | Milvus connection URI |
| `MILVUS_USER` | `root` | Milvus username |
| `MILVUS_PASSWORD` | *(required, from secret)* | Milvus password — never in ConfigMap |
| `EMBEDDINGS_URL` | `http://embeddings-service-predictor.ml-infra.svc.cluster.local/embed` | TEI embeddings service |
| `COLLECTION_NAME` | `kubeflow_docs` | Docs Milvus collection (upstream default) |
| `ISSUES_COLLECTION_NAME` | `issues_rag` | GitHub issues collection |
| `CODE_COLLECTION_NAME` | `code_rag` | Kubeflow code collection |
| `PORT` | `8000` | Server listen port |

### Important caveats

- **No automated test suite** exists in this repo. Validation is done by building Docker images, deploying, and testing MCP tool calls.
- **No linter configuration** is present. Use `python -m py_compile <file>` for syntax checks.
- Embeddings are served by **TEI** (`embeddings-service` in `ml-infra`); MCP and pipelines call `EMBEDDINGS_URL` over HTTP (no torch in images).
- The `docs-agent-mcp/pipelines/` directory contains Kubeflow Pipeline definitions designed for K8s clusters, not local execution.
- Platform infrastructure is defined under `docs-agent-mcp/terraform/` (Knative, KServe, Milvus operator, KFP standalone, kagent, Istio policies).
- The `oci-cli` Python package is installed in the venv for kubectl exec credential plugin. If you see `oci` import errors, ensure the venv is activated or `oci` is on PATH.
