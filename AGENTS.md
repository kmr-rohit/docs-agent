# AGENTS.md

## Cursor Cloud specific instructions

### Primary validation target

This repo's validation target is the **Kagent MCP deployment path**, not local FastAPI/WebSocket server startup. The important runtime path is:

```
User / Kagent UI → Kagent Agent CRD → MCP server (kagent-feast-mcp/mcp-server/) → Milvus collections → LLM backend (Groq / KServe)
```

The MCP server at `kagent-feast-mcp/mcp-server/server.py` is the main thing to validate. The legacy servers at `server/` and `server-https/` only need basic syntax/import sanity checks.

### MCP server overview

| Tool | Milvus Collection | Status |
|---|---|---|
| `search_kubeflow_docs` | `docs_rag` | Populated and loaded |
| `search_github_issues` | `issues_rag` | Populated and loaded |
| `search_kubeflow_code` | `code_rag` | Currently empty — acceptable until code pipeline is rerun |

The server uses FastMCP with `streamable-http` transport on port 8000. The MCP endpoint is at `/mcp`.

### Code change → validation loop

When code changes are made to the MCP server:

1. **Build the Docker image** (context is `kagent-feast-mcp/`, Dockerfile at `kagent-feast-mcp/mcp-server/Dockerfile`):
   ```bash
   docker build -t ghcr.io/kmr-rohit/mcp-kubeflow-docs:<tag> \
     -f kagent-feast-mcp/mcp-server/Dockerfile kagent-feast-mcp/
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

The CI workflow at `.github/workflows/build-mcp-image.yml` automatically builds and pushes to GHCR on pushes to `main` that touch `kagent-feast-mcp/mcp-server/**`.

### Local Docker validation (without cluster)

You can build and run the MCP container locally to verify the image builds and the FastMCP server starts:

```bash
docker build -t mcp-kubeflow-docs:local-test \
  -f kagent-feast-mcp/mcp-server/Dockerfile kagent-feast-mcp/
docker run -d --name mcp-test -p 8003:8000 mcp-kubeflow-docs:local-test
# Verify: curl -s -X POST http://localhost:8003/mcp \
#   -H "Content-Type: application/json" \
#   -H "Accept: application/json, text/event-stream" \
#   -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
```

Tool calls will fail without a reachable Milvus instance, but `initialize` and `tools/list` should succeed.

### Cluster inspection commands

```bash
kubectl get all -n docs-agent
kubectl logs -n docs-agent deploy/mcp-kubeflow-docs --tail=100 -f
kubectl get svc -n docs-agent -o wide
kubectl get svc -n docs-agent | grep -i milvus
```

### Namespace boundaries

- **Safe target namespace:** `docs-agent`
- KFP workflows usually live in `user`
- **Never** make destructive changes in shared control-plane namespaces: `kubeflow`, `kubeflow-system`, `istio-system`, `knative-serving`
- **Never** drain/delete shared cluster nodes

### Python dependencies (local dev)

All components share a virtualenv at `/workspace/.venv`. Activate with `source /workspace/.venv/bin/activate`. Dependencies are installed from:
- `server/requirements.txt`
- `server-https/requirements.txt`
- `kagent-feast-mcp/mcp-server/requirements.txt`

### Environment variables (MCP server)

| Variable | Default | Description |
|---|---|---|
| `MILVUS_URI` | `http://localhost:19530` | Milvus connection URI |
| `MILVUS_USER` | `root` | Milvus username |
| `MILVUS_PASSWORD` | `Milvus` | Milvus password |
| `COLLECTION_NAME` | `kubeflow_docs_docs_rag` | Milvus collection name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | Sentence-transformer model |
| `PORT` | `8000` | Server listen port |

### Important caveats

- **No automated test suite** exists in this repo. Validation is done by building Docker images, deploying, and testing MCP tool calls.
- **No linter configuration** is present. Use `python -m py_compile <file>` for syntax checks.
- The `sentence-transformers` embedding model (~400 MB) is baked into the Docker image at build time (see Dockerfile `RUN python -c "from sentence_transformers ..."`). First local import also downloads it to `~/.cache/huggingface/`.
- The `pipelines/` directory contains Kubeflow Pipeline definitions designed for K8s clusters, not local execution.
- Docker is required in the Cloud Agent VM for image builds. See the setup steps: install Docker, configure `fuse-overlayfs` storage driver and `iptables-legacy`, then start `dockerd`.
