# Embeddings service (TEI) model options

Current stack uses **768-dimensional** vectors in Milvus (`dim=768`). Any replacement model must also output **768 dims** unless you drop and re-ingest all collections.

## Current model

| Setting | Value |
|---------|--------|
| Model | `sentence-transformers/all-mpnet-base-v2` |
| Max tokens **per input** | **384** (hard model limit) |
| Vector dim | **768** |
| TEI flags (recommended) | `--auto-truncate`, `--max-client-batch-size 64` |

Without `--auto-truncate`, TEI returns **413** when any string tokenizes to more than 384 tokens (even with `embedding_batch_size: 8` in pipelines).

## Drop-in alternatives (768 dim, longer context)

These work with TEI on CPU and keep Milvus schema without changing `dim=768`:

| Model | Context (typical) | MTEB / notes | TEI CPU |
|-------|-------------------|--------------|---------|
| `nomic-ai/nomic-embed-text-v1.5` | **8192** tokens | Strong open embedder | Supported |
| `jinaai/jina-embeddings-v2-base-en` | **8192** tokens | Good for English docs | Supported |
| `sentence-transformers/all-mpnet-base-v2` | **384** tokens | Current default | Supported |

To switch, set `embeddings_model_id` in Terraform (or patch the InferenceService args), roll out TEI, then **re-run all ingestion pipelines** (vectors change even at 768d).

```hcl
# docs-agent-mcp/terraform/variables.tf
embeddings_model_id = "nomic-ai/nomic-embed-text-v1.5"
```

Increase memory limits on the TEI container if the pod OOMs on first load (~500MB+ for nomic).

## Not drop-in (different dimensions)

| Model | Issue |
|-------|--------|
| `Alibaba-NLP/gte-large-en-v1.5` | 1024 dim |
| `Qwen/Qwen3-Embedding-*` | 1024+ dim |
| `intfloat/multilingual-e5-large-instruct` | 1024 dim |

Using these requires changing every `FieldSchema(..., dim=...)` in pipelines and MCP, then full re-ingest.

## Batch size

| Limit | Default | Tuned |
|-------|---------|-------|
| Pipeline `embedding_batch_size` | 8 | Can raise to 16–32 after `--auto-truncate` |
| TEI `--max-client-batch-size` | 32 | **64** (cluster patch / Terraform) |
| TEI `--max-batch-tokens` | 16384 | Total tokens per internal batch |

Larger batch sizes speed ingestion; per-input length is still capped by the model (or auto-truncate).

## Pipeline-side truncation

Repo pipelines also truncate to **1000 characters** before TEI (`max_tei_chars` in component code). Re-compile and upload YAML after pulling the fix branch. TEI `--auto-truncate` is a safety net when an old pipeline version is still deployed.
