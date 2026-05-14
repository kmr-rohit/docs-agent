# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a **Kubeflow Documentation AI Assistant** — a RAG chatbot with three runnable Python services:

| Service | Path | Command | Port | Description |
|---|---|---|---|---|
| HTTPS API (FastAPI) | `server-https/` | `python app.py` | 8000 (default) | REST API with `/chat`, `/health`, `/docs` endpoints |
| WebSocket API | `server/` | `python app.py` | 8000 (default) | Real-time bidirectional chat via WebSocket |
| MCP Server | `kagent-feast-mcp/mcp-server/` | `python server.py` | 8000 (default) | FastMCP server exposing `search_kubeflow_docs` tool |

All three share the same virtualenv at `/workspace/.venv`.

### Running services locally

Activate the virtualenv first: `source /workspace/.venv/bin/activate`

To avoid port conflicts when running multiple servers, override `PORT` per service:
```bash
PORT=8001 python server-https/app.py   # HTTPS API
PORT=8002 python server/app.py          # WebSocket API
```

### Environment variables

See the README table for full details. Key variables:
- `KSERVE_URL` — LLM inference endpoint (defaults to a K8s service URL; override for local dev)
- `MODEL` — model name for LLM (default: `llama3.1-8B`)
- `MILVUS_HOST` / `MILVUS_PORT` — Milvus vector DB connection (defaults to K8s service URL)
- `MILVUS_COLLECTION` — collection name (default: `docs_rag`)
- `LLM_API_KEY` — API key for the LLM endpoint (HTTPS server only)
- `PORT` — server listen port (default: `8000`)

### Important caveats

- **No automated tests exist** in this repo. Validation is done by running the servers and hitting endpoints.
- **No linter configuration** is present (no `pyproject.toml`, `setup.cfg`, `ruff.toml`, etc.). You can run `python -m py_compile <file>` to check syntax.
- The `/chat` endpoint requires a reachable LLM endpoint (`KSERVE_URL`) and Milvus instance. Without these, chat requests return a graceful error but the server itself runs fine — `/health` and `/` endpoints work independently.
- The `pipelines/` directory contains Kubeflow Pipeline definitions designed to run on a Kubernetes cluster, not locally.
- The `sentence-transformers` model downloads on first import (~400 MB). It is cached in `~/.cache/huggingface/`.
- PyTorch is installed CPU-only via `--extra-index-url https://download.pytorch.org/whl/cpu` (per `server/requirements.txt`).
