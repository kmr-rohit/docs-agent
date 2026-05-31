# Manual RAG Ingestion (Milvus)

CD deploys the MCP server and Kagent only. **It does not run ingestion pipelines** and does not
drop or recreate Milvus collections. Ingestion is a separate, manual step you run when you want
to add or refresh indexed content.

## Collections

| Milvus collection | Pipeline | MCP tool |
|-------------------|----------|----------|
| `kubeflow_docs_docs_rag` | `docs-agent-mcp/pipelines/kubeflow-pipeline.py` | `search_kubeflow_docs` |
| `issues_rag` | `docs-agent-mcp/pipelines/issues-pipeline.py` | `search_github_issues` |
| `code_rag` | `docs-agent-mcp/pipelines/code-pipeline.py` | `search_kubeflow_code` |

Milvus host (in-cluster, **ml-infra**): `milvus-milvus.ml-infra.svc.cluster.local:19530`

Embeddings (TEI): `http://embeddings-service-predictor.ml-infra.svc.cluster.local/embed`

Legacy Milvus in `docs-agent` (`my-release-milvus`) is **not** used by the new MCP/pipelines unless you override parameters.

**Incremental behavior:** `store_milvus` upserts by `file_unique_id` — it deletes old chunks for
files being re-indexed, then inserts new ones. It does **not** drop the whole collection on each
run. Existing chunks for files you do not re-ingest are left untouched.

---

## Prerequisites

- `kubectl` context pointed at OKE
- Python 3.10+ locally (or use Kubeflow Pipelines UI only)
- GitHub PAT with **`repo`** read (for docs/issues/code download rate limits)

```bash
pip install kfp kfp-kubernetes
pip install -r docs-agent-mcp/pipelines/requirements.txt
```

---

## 1. Compile pipeline packages

From the repo root:

```bash
cd docs-agent-mcp/pipelines
python kubeflow-pipeline.py          # → github_rag_pipeline.yaml
python issues-pipeline.py            # → github_issues_rag_pipeline.yaml
python code-pipeline.py              # → code_rag_pipeline.yaml
```

---

## 2. Secrets (profile namespace + kubeflow)

Replace `user` with **your** profile namespace from `kubectl get profile` (e.g. `user`, `amlc-bruce`, `amlc-carl`).

### GitHub PAT (`github-pat`)

```bash
export GITHUB_PAT=ghp_your_token_here
PROFILE_NS=user   # your profile

kubectl create secret generic github-pat -n "${PROFILE_NS}" \
  --from-literal=Github_Pat="${GITHUB_PAT}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

### Milvus password (`milvus-auth`) — store steps

Pipeline **store** components read `MILVUS_PASSWORD` from `milvus-auth` in the **`kubeflow`** namespace:

```bash
kubectl create secret generic milvus-auth -n kubeflow \
  --from-literal=MILVUS_PASSWORD='Milvus' \
  --dry-run=client -o yaml | kubectl apply -f -
```

Use the same password as `mcp-server-secret` in `docs-agent`.

---

## 3. Run pipelines

### Option A — Kubeflow Pipelines UI via Dex (recommended)

**Multi-user mode:** every run must belong to a **profile namespace**. If namespace is blank you get:

`An experiment cannot have an empty namespace in multi-user mode`

#### Step-by-step (Dex login)

1. **Find your profile namespace** (where runs and `github-pat` live):
   ```bash
   kubectl get profile
   ```
   Example: `user`, `amlc-bruce`, or `amlc-carl`.

2. **Port-forward the Central Dashboard** (includes Dex login flow):
   ```bash
   kubectl port-forward -n kubeflow svc/centraldashboard 8080:80
   ```
   Keep this terminal open.

3. Open **http://localhost:8080** in a browser.

4. **Log in with Dex** (email / OIDC — use the identity your cluster admin configured).

5. After login, confirm the UI shows your **namespace** (profile name from step 1) in the header or namespace picker.

6. Go to **Pipelines** (left menu) → **Upload pipeline** → choose a compiled YAML from `docs-agent-mcp/pipelines/`:
   - `github_rag_pipeline.yaml`
   - `github_issues_rag_pipeline.yaml`
   - `code_rag_pipeline.yaml`

7. **Create run** → pick or create an **experiment** in your namespace.

8. **Parameters** — defaults already target ml-infra (no change needed unless you use a custom Milvus):
   | Parameter | Default (ml-infra) |
   |-----------|-------------------|
   | `embeddings_service_url` | `http://embeddings-service-predictor.ml-infra.svc.cluster.local/embed` |
   | `milvus_host` | `milvus-milvus.ml-infra.svc.cluster.local` |
   | `milvus_port` | `19530` |

9. Leave `github_token` **empty** if `github-pat` exists in your profile namespace.

10. **Create** and watch the run graph. Pods appear in your profile namespace (`kubectl get pods -n user`).

If namespace errors persist, use **Option B** (Python client with explicit `namespace=`).

Run order (each is independent; run only what you need):

1. **Docs** — `github_rag_pipeline.yaml` (~30–60 min with PAT; embed step is CPU-heavy)
2. **Issues** — `github_issues_rag_pipeline.yaml`
3. **Code** — `code_rag_pipeline.yaml`

### Option B — KFP Python client (local)

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888 &
```

```python
import kfp

# REQUIRED in multi-user mode — use YOUR profile namespace from: kubectl get profile
NAMESPACE = "user"  # or amlc-bruce / amlc-carl

client = kfp.Client(host="http://localhost:8888", namespace=NAMESPACE)

# Create experiment in that namespace first (avoids empty-namespace error)
client.create_experiment(name="rag-ingestion", namespace=NAMESPACE)

for name, path in [
    ("docs", "docs-agent-mcp/pipelines/github_rag_pipeline.yaml"),
    ("issues", "docs-agent-mcp/pipelines/github_issues_rag_pipeline.yaml"),
    ("code", "docs-agent-mcp/pipelines/code_rag_pipeline.yaml"),
]:
    result = client.create_run_from_pipeline_package(
        pipeline_file=path,
        run_name=f"{name}-rag-manual",
        experiment_name="rag-ingestion",
        namespace=NAMESPACE,
        enable_caching=False,
        arguments={"github_token": ""},
    )
    print(name, result.run_id)
```

Monitor runs:

```bash
kubectl get workflows -n user
```

---

## 4. Useful pipeline parameters

### Docs (`github_rag_pipeline.yaml`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `directory_path` | `content/en/docs` | Kubeflow website docs root |
| `github_token` | `""` | Use secret mount or paste PAT |
| `collection_name` | `kubeflow_docs_docs_rag` | Do not change unless MCP config changes |

### Issues (`github_issues_rag_pipeline.yaml`)

| Parameter | Default |
|-----------|---------|
| `repos` | `kubeflow/kubeflow,kubeflow/pipelines,kubeflow/manifests` |
| `max_issues_per_repo` | `200` |
| `collection_name` | `issues_rag` |

### Code (`code_rag_pipeline.yaml`)

| Parameter | Default |
|-----------|---------|
| `repos` | `kubeflow/manifests` |
| `directory_paths` | `apps/pipeline/upstream,apps/katib,...` |
| `collection_name` | `code_rag` |

---

## 5. Verify Milvus row counts

From the MCP pod (read-only):

```bash
kubectl exec -n docs-agent deploy/mcp-kubeflow-docs -- python3 -c "
from pymilvus import connections, utility, Collection
connections.connect('default', host='milvus-milvus.ml-infra.svc.cluster.local', port='19530', user='root', password='Milvus')
for name in ['kubeflow_docs_docs_rag', 'issues_rag', 'code_rag']:
    if utility.has_collection(name):
        col = Collection(name)
        print(f'{name}: {col.num_entities} entities')
    else:
        print(f'{name}: (missing)')
"
```

---

## 6. Test MCP tools from the pod

### Docs search

```bash
kubectl exec -n docs-agent deploy/mcp-kubeflow-docs -- python3 -c "
from server import search_kubeflow_docs
print(search_kubeflow_docs('KServe InferenceService setup', top_k=3))
"
```

### Issues search

```bash
kubectl exec -n docs-agent deploy/mcp-kubeflow-docs -- python3 -c "
from server import search_github_issues
print(search_github_issues('pipeline compile error', top_k=3))
"
```

### Code search

```bash
kubectl exec -n docs-agent deploy/mcp-kubeflow-docs -- python3 -c "
from server import search_kubeflow_code
print(search_kubeflow_code('InferenceService yaml', top_k=3))
"
```

### End-to-end via Kagent UI

Ask: *"How do I deploy a model with KServe on Kubeflow?"* — agent should call
`search_kubeflow_docs` and cite indexed URLs.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `experiment cannot have an empty namespace in multi-user mode` | Set namespace to your profile (`kubectl get profile` → e.g. `user`). Use Python client with `namespace=` or log in via Central Dashboard before creating a run |
| Pipeline pod `secret "github-pat" not found` | Create secret in **your profile** namespace (step 2), not only `docs-agent` |
| Download very slow / 403 in logs | Set `GITHUB_PAT` / `github-pat` secret |
| `issues_rag` / `code_rag` tool returns empty | Collection not created yet — run that pipeline |
| Embed step slow | Normal on CPU; chunk step uses PyTorch without GPU |
| CD must not wipe Milvus | CD does not run pipelines; only redeploys MCP/Kagent |

---

## What CD does (and does not do)

**CD deploys:** MCP image, Kagent agent, `qwen-llm` Service, optional KServe (only when manifests change).

**CD does not:** run ingestion, drop Milvus collections, or modify indexed vectors.
