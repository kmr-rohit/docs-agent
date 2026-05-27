#!/usr/bin/env bash
# Upload fixed docs RAG pipeline and start Milvus ingestion on KFP.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "${ROOT_DIR}/scripts/submit-docs-ingestion.py" "$@"
