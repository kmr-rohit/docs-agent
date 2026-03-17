# Code Pipeline Testing Prompt

Paste the following into your local Claude Code session that has kubectl access to your OCI cluster.

---

## Context: What was implemented

I implemented the YAML/code ingestion pipeline (Phase 2 of the GSoC plan) for the kubeflow/docs-agent project. Here's what changed:

### New files:
1. **`pipelines/code_utils.py`** — YAML-aware and Python AST-aware code chunking utilities
   - `parse_yaml_documents()`: Splits YAML at `---` boundaries, extracts K8s metadata (kind, name, namespace) via `yaml.safe_load`. Detects kustomization files. Falls back gracefully on Helm templates.
   - `parse_python_ast()`: Uses `ast` module to split Python at function/class boundaries. Extracts module header, function names, class names. Falls back to whole-file on syntax errors.
   - `parse_json_file()`: Indexes JSON as single chunks.
   - `chunk_code_file()`: Routes by file extension, sub-splits oversized chunks with `RecursiveCharacterTextSplitter`.

2. **`pipelines/code-pipeline.py`** — KFP pipeline with 3 components:
   - `download_github_code`: Recursively fetches code files from GitHub repos with rate limit handling
   - `chunk_and_embed_code`: Inlines the chunking logic (KFP can't import local modules), embeds with `all-mpnet-base-v2` (768-dim)
   - `store_code_milvus`: Creates `code_rag` Milvus collection with extended schema

### Modified files:
3. **`kagent-feast-mcp/mcp-server/server.py`** — Added `search_kubeflow_code` MCP tool
   - Extracted shared `_search_collection()` helper (DRY refactor)
   - New tool returns results with resource metadata (kind, name, namespace) in code blocks
   - New env var: `CODE_COLLECTION_NAME` (default: `code_rag`)

4. **`kagent-feast-mcp/manifests/kagent/setup.yaml`** — Updated Agent CRD
   - Added `search_kubeflow_code` to toolNames
   - Updated system message with 3-tool routing (docs vs code vs both)

5. **`pipelines/requirements.txt`** — Added `pyyaml`

### Milvus `code_rag` collection schema:
Same as `docs_rag` (id, file_unique_id, repo_name, file_path, file_name, citation_url, chunk_index, content_text VARCHAR(2000), vector FLOAT_VECTOR(768), last_updated) PLUS:
- `resource_kind` VARCHAR(128) — Deployment, Service, ConfigMap, function, class, etc.
- `resource_name` VARCHAR(256) — metadata.name or function/class name
- `resource_namespace` VARCHAR(256) — metadata.namespace
- `file_type` VARCHAR(64) — yaml, kustomize, python, json, text

---

## What I need you to do: Test on my OCI cluster

I have kubectl access configured for my OCI cluster with the docs-agent namespace. Please help me test this end-to-end. Here are the steps:

### Step 1: Compile the pipeline
```bash
cd pipelines
python code-pipeline.py
```
This should produce `code_rag_pipeline.yaml`.

### Step 2: Run the pipeline on KFP
I need to submit `code_rag_pipeline.yaml` to my KFP instance. My configuration:
- KFP endpoint: check via `kubectl get svc -n kubeflow | grep ml-pipeline`
- Milvus host: `milvus-standalone-final.docs-agent.svc.cluster.local`
- Milvus port: `19530`
- Collection name: `code_rag`
- Start with a small test: single repo `kubeflow/manifests`, single directory `apps/pipeline/upstream`
- I'll provide my GitHub token when prompted

Port-forward KFP if needed:
```bash
kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8080:80
```

Then submit via SDK:
```python
import kfp
client = kfp.Client(host="http://localhost:8080")
client.create_run_from_pipeline_package(
    "code_rag_pipeline.yaml",
    arguments={
        "repos": "kubeflow/manifests",
        "directory_paths": "apps/pipeline/upstream",
        "file_extensions": "yaml,yml",
        "github_token": "<MY_TOKEN>",
        "milvus_host": "milvus-standalone-final.docs-agent.svc.cluster.local",
        "milvus_port": "19530",
        "collection_name": "code_rag",
    },
)
```

### Step 3: Verify the Milvus collection after pipeline completes
Port-forward Milvus:
```bash
kubectl port-forward svc/milvus-standalone-final -n docs-agent 19530:19530
```

Then verify:
```python
from pymilvus import MilvusClient
client = MilvusClient(uri="http://localhost:19530")

# Check collection exists and has data
stats = client.get_collection_stats("code_rag")
print(f"Entity count: {stats}")

# Query by resource_kind to verify YAML metadata extraction
results = client.query(
    "code_rag",
    filter="resource_kind == 'Deployment'",
    limit=5,
    output_fields=["resource_kind", "resource_name", "resource_namespace", "file_path", "file_type"]
)
for r in results:
    print(f"  {r['resource_kind']} {r['resource_name']} ns={r['resource_namespace']} file={r['file_path']}")

# Test semantic search
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
query_vec = model.encode("pipeline controller deployment").tolist()
hits = client.search("code_rag", data=[query_vec], limit=3, output_fields=["content_text", "resource_kind", "resource_name", "file_path"])
for hit in hits[0]:
    print(f"Score: {hit['distance']:.4f} | {hit['entity']['resource_kind']} {hit['entity']['resource_name']} | {hit['entity']['file_path']}")
```

### Step 4: Test the MCP server with the new tool
Update the MCP server deployment with the new env var and restart:
```bash
kubectl set env deployment/mcp-kubeflow-docs CODE_COLLECTION_NAME=code_rag -n docs-agent
kubectl rollout restart deployment/mcp-kubeflow-docs -n docs-agent
kubectl rollout status deployment/mcp-kubeflow-docs -n docs-agent
```

Verify the MCP server image is up to date. If needed, rebuild and push:
```bash
cd kagent-feast-mcp/mcp-server
docker build -t <registry>/mcp-kubeflow-docs:latest .
docker push <registry>/mcp-kubeflow-docs:latest
kubectl rollout restart deployment/mcp-kubeflow-docs -n docs-agent
```

### Step 5: Update the Agent CRD
Replace `<YOUR_NAMESPACE>` with the actual namespace and apply:
```bash
kubectl apply -f kagent-feast-mcp/manifests/kagent/setup.yaml
```

### Step 6: End-to-end agent test
Send queries through the agent that should route to `search_kubeflow_code`:
- "Show me the pipeline controller Deployment manifest"
- "What Services are defined for Katib?"
- "How is Istio configured in kubeflow/manifests?"

And queries that should still route to `search_kubeflow_docs`:
- "How do I install Kubeflow Pipelines?"
- "What is KServe?"

### Things to watch for:
- **GitHub API rate limits**: The pipeline drops the collection before inserting. If rate-limited mid-run, `code_rag` ends up empty. Use a PAT with sufficient quota.
- **Start small**: Test with just `apps/pipeline/upstream` first before running all 4 directories.
- **KFP logs**: Check `chunk_and_embed_code` step logs — it prints per-file chunk counts to verify YAML splitting works.
- **content_text truncation**: VARCHAR(2000) silently drops content from large files. Check if critical manifests are being truncated.
- **Llama on Groq**: Use `qwen/qwen3-32b` model, not Llama — Llama generates malformed tool calls through kagent.
