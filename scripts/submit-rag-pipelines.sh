#!/usr/bin/env bash
# Submit docs / issues / code RAG pipelines to Kubeflow Pipelines on OKE.
#
# Usage:
#   GITHUB_PAT=ghp_... ./scripts/submit-rag-pipelines.sh
#   ./scripts/submit-rag-pipelines.sh docs
#   ./scripts/submit-rag-pipelines.sh issues code
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINES_DIR="${ROOT_DIR}/pipelines"
KFP_NAMESPACE="${KFP_NAMESPACE:-user}"
KFP_HOST="${KFP_HOST:-http://127.0.0.1:8888}"
PORT_FORWARD_PID=""

if [[ $# -gt 0 ]]; then
  PIPELINES=("$@")
else
  PIPELINES=(docs issues code)
fi

cleanup() {
  if [[ -n "$PORT_FORWARD_PID" ]]; then
    kill "$PORT_FORWARD_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

resolve_github_pat() {
  if [[ -n "${GITHUB_PAT:-}" ]]; then
    return 0
  fi
  for ns in docs-agent "$KFP_NAMESPACE" kubeflow; do
    if kubectl get secret github-pat -n "$ns" &>/dev/null; then
      GITHUB_PAT="$(kubectl get secret github-pat -n "$ns" \
        -o jsonpath='{.data.Github_Pat}' | base64 -d)"
      export GITHUB_PAT
      echo "==> Loaded Github_Pat from secret/github-pat in namespace ${ns}"
      return 0
    fi
  done
  echo "WARN: GITHUB_PAT not set and github-pat secret not found — rate limits will apply"
}

ensure_github_pat_secrets() {
  local pat="${GITHUB_PAT:-}"
  for ns in docs-agent "$KFP_NAMESPACE"; do
    kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    kubectl create secret generic github-pat \
      --namespace "$ns" \
      --from-literal=Github_Pat="${pat}" \
      --dry-run=client -o yaml | kubectl apply -f -
    if [[ -n "$pat" ]]; then
      echo "==> Ensured github-pat secret (with token) in namespace ${ns}"
    else
      echo "==> Ensured github-pat secret placeholder in namespace ${ns} (set GITHUB_PAT for auth)"
    fi
  done
}

start_port_forward() {
  if curl -sf "${KFP_HOST}/apis/v1beta1/healthz" >/dev/null 2>&1; then
    echo "==> KFP API already reachable at ${KFP_HOST}"
    return 0
  fi
  echo "==> Port-forwarding ml-pipeline.kubeflow:8888"
  kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888 >/tmp/kfp-pf.log 2>&1 &
  PORT_FORWARD_PID=$!
  for _ in $(seq 1 30); do
    if curl -sf "${KFP_HOST}/apis/v1beta1/healthz" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "ERROR: KFP API not reachable at ${KFP_HOST}"
  cat /tmp/kfp-pf.log || true
  exit 1
}

compile_pipelines() {
  echo "==> Compiling pipeline YAML packages"
  (cd "$PIPELINES_DIR" && python3 kubeflow-pipeline.py)
  (cd "$PIPELINES_DIR" && python3 issues-pipeline.py)
  (cd "$PIPELINES_DIR" && python3 code-pipeline.py)
}

run_pipelines() {
  python3 - <<PY
import os
import time
from datetime import datetime

import kfp

host = os.environ["KFP_HOST"]
namespace = os.environ["KFP_NAMESPACE"]
pipelines_dir = os.environ["PIPELINES_DIR"]
selected = os.environ["SELECTED_PIPELINES"].split(",")

mapping = {
    "docs": f"{pipelines_dir}/github_rag_pipeline.yaml",
    "issues": f"{pipelines_dir}/github_issues_rag_pipeline.yaml",
    "code": f"{pipelines_dir}/code_rag_pipeline.yaml",
}

client = kfp.Client(host=host)

for name in selected:
    name = name.strip()
    path = mapping.get(name)
    if not path or not os.path.isfile(path):
        raise SystemExit(f"Unknown or missing pipeline: {name}")

    run_name = f"{name}-rag-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    print(f"==> Submitting {name}: {run_name}")
    result = client.create_run_from_pipeline_package(
        pipeline_file=path,
        run_name=run_name,
        namespace=namespace,
        enable_caching=False,
        arguments={"github_token": ""},
    )
    run_id = result.run_id
    print(f"    run_id={run_id}")

    for _ in range(720):
        run = client.get_run(run_id)
        state = getattr(run, "state", None) or getattr(run, "status", None)
        print(f"    state={state}")
        if state in ("SUCCEEDED", "FAILED", "ERROR", "SKIPPED"):
            if state != "SUCCEEDED":
                raise SystemExit(f"Pipeline {name} ({run_id}) failed: {state}")
            print(f"==> {name} pipeline succeeded")
            break
        time.sleep(10)
    else:
        raise SystemExit(f"Timed out waiting for pipeline {name} ({run_id})")

print("==> All requested pipelines finished")
PY
}

resolve_github_pat
ensure_github_pat_secrets
compile_pipelines
start_port_forward

export KFP_HOST KFP_NAMESPACE PIPELINES_DIR
export SELECTED_PIPELINES="$(IFS=,; echo "${PIPELINES[*]}")"
run_pipelines
