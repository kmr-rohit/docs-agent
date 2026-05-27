# Milvus empty — docs ingestion required

## Why the MCP tool fails

```
collection not found[database=default][collection=kubeflow_docs_docs_rag]
```

**Kagent and KServe are fine.** The MCP server connects to Milvus, but Milvus has **zero collections**:

```bash
kubectl exec -n docs-agent deploy/mcp-kubeflow-docs -- python3 -c "
from pymilvus import MilvusClient
c = MilvusClient(uri='http://my-release-milvus.docs-agent.svc.cluster.local:19530', user='root', password='Milvus')
print(c.list_collections())
"
# → []
```

The MCP ConfigMap expects collection **`kubeflow_docs_docs_rag`**, which is created by the **docs ingestion Kubeflow Pipeline**, not by MCP or Kagent.

## Common causes

1. **Ingestion pipeline never completed** (or stuck — check `kubectl get workflows -n user`)
2. **Wrong collection name** — default in `pipelines/github_rag_pipeline.yaml` is `docs_rag`, but MCP uses `kubeflow_docs_docs_rag`
3. **Wrong Milvus host** — pipeline default `milvus-standalone-final...` ≠ actual service `my-release-milvus.docs-agent.svc.cluster.local`
4. **Milvus redeploy without persistence** — no PVCs in `docs-agent`; standalone Milvus data can be lost on reinstall

## Fix: run docs ingestion once

### Option A — Kubeflow Pipelines UI

1. Open **Pipelines → Upload pipeline** → `pipelines/github_rag_pipeline.yaml`
2. **Create run** with:

| Parameter | Value |
|-----------|--------|
| `milvus_host` | `my-release-milvus.docs-agent.svc.cluster.local` |
| `milvus_port` | `19530` |
| `collection_name` | `kubeflow_docs_docs_rag` |
| `repo_owner` | `kubeflow` |
| `repo_name` | `website` |
| `directory_path` | `content/en/docs` |
| `base_url` | `https://www.kubeflow.org/docs` |
| `github_token` | GitHub PAT with `public_repo` read (strongly recommended) |

3. Wait for run to **Succeeded** (~15–45 min depending on cluster)

**Without `github_token`:** GitHub API rate limits cause 403 errors; KServe/Pipelines docs are skipped (only ~17 files indexed in testing). Always provide a token for full corpus.

### Option B — Feast pipeline (same collection name)

Use `kagent-feast-mcp/pipelines/github_rag_pipeline.yaml` with:

| Parameter | Value |
|-----------|--------|
| `feast_online_store_host` | `http://my-release-milvus.docs-agent.svc.cluster.local` |
| `feast_project` | `kubeflow_docs` |

Creates collection `{feast_project}_docs_rag` → `kubeflow_docs_docs_rag`

## Verify after ingestion

```bash
kubectl exec -n docs-agent deploy/mcp-kubeflow-docs -- python3 -c "
from pymilvus import MilvusClient
c = MilvusClient(uri='http://my-release-milvus.docs-agent.svc.cluster.local:19530', user='root', password='Milvus')
print('collections:', c.list_collections())
col = 'kubeflow_docs_docs_rag'
if col in c.list_collections():
    print('entities:', c.get_collection_stats(col))
"
```

Then retry in Kagent UI — `search_kubeflow_docs` should return citations.

## Stuck old workflows

Delete or ignore stale runs:

```bash
kubectl get workflows -n user
# kubectl delete workflow github-rag-rntjb -n user   # if stuck >24h
```

### ImageInspectError — `short name mode is enforcing`

If a run shows **Running** forever and the `chunk-and-embed` pod has `ImageInspectError`:

```
Failed to inspect image "": short name mode is enforcing, but image name
pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime returns ambiguous list
```

**Cause:** OKE nodes require **fully qualified** image names (e.g. `docker.io/pytorch/...`, not `pytorch/pytorch:...`).

**Fix:**

1. Terminate the stuck run in KFP UI (or `kubectl delete workflow <name> -n user`)
2. Re-upload **`pipelines/github_rag_pipeline.yaml`** from this repo (images are prefixed with `docker.io/`)
3. Start a new run with the same parameters

Check pod status:

```bash
kubectl get workflows -n user
kubectl get pods -n user | rg ImageInspect
kubectl describe pod -n user <chunk-and-embed-impl-pod> | rg -A2 Events
```

## Long-term

- Enable **Milvus persistence** (PVC) in Helm values so reindex survives restarts
- Schedule **incremental pipeline** (`pipelines/github_rag_incremental_pipeline.yaml`) on doc updates
- Align pipeline YAML defaults with MCP ConfigMap (`collection_name`, `milvus_host`)
