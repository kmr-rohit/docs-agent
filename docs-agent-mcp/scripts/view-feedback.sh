#!/usr/bin/env bash
# View docs-agent feedback rows via port-forward + export API.
set -euo pipefail

NAMESPACE="${NAMESPACE:-docs-agent}"
LOCAL_PORT="${LOCAL_PORT:-18081}"
MIN_RATING="${MIN_RATING:-}"
LIMIT="${LIMIT:-50}"

kubectl port-forward -n "$NAMESPACE" "svc/docs-feedback" "${LOCAL_PORT}:8080" >/tmp/view-feedback-pf.log 2>&1 &
PF_PID=$!
trap 'kill "$PF_PID" 2>/dev/null || true' EXIT
sleep 2

URL="http://127.0.0.1:${LOCAL_PORT}/api/feedback/export?limit=${LIMIT}"
if [ -n "$MIN_RATING" ]; then
  URL="${URL}&min_rating=${MIN_RATING}"
fi

if command -v jq >/dev/null; then
  curl -sf "$URL" | jq .
else
  curl -sf "$URL"
fi
