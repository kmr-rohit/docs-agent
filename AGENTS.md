# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

This is the **Kubeflow Documentation AI Assistant** — a RAG system with three API server variants (WebSocket, HTTPS/FastAPI, MCP) that query a Milvus vector database and an LLM (KServe/vLLM or Groq). See `README.md` for full architecture.

### Running tests

```bash
pytest tests/ -v --tb=short
```

All tests use mocks and require no external services. Dependencies: `pip install -r requirements-test.txt`.

### Linting

No linter is configured in the repo. Use `flake8 --max-line-length=120` for basic checks.

### Starting API servers locally

- **HTTPS server:** `python server-https/app.py` (FastAPI on port 8000). Health: `GET /health`, root: `GET /`, docs: `GET /docs`.
- **WebSocket server:** `python server/app.py` (port 8000). Health: `GET /health`.
- **MCP server:** `python kagent-feast-mcp/mcp-server/server.py` (FastMCP on port 8000, streamable-http transport).

All three servers default to port 8000 — only run one at a time.

### Key caveats

- The `/chat` endpoint on both WS and HTTPS servers requires a KServe LLM backend (`KSERVE_URL`) and a Milvus instance (`MILVUS_HOST`). These are Kubernetes-internal services and are **not available** in the Cloud Agent VM. The servers will start and serve health/docs endpoints, but chat requests will fail with connection errors.
- The MCP server also requires a live Milvus instance for the tool functions; it starts fine without one but tool calls will fail at runtime.
- To point servers at alternative backends, set environment variables: `KSERVE_URL`, `MILVUS_HOST`, `MILVUS_PORT`, `MODEL`.
- The `sentence-transformers/all-mpnet-base-v2` model is downloaded from HuggingFace Hub on first use (~420 MB). This happens automatically at server startup or when tests import related code.
- PyTorch is installed as CPU-only (`--extra-index-url https://download.pytorch.org/whl/cpu`) to keep the image small.
