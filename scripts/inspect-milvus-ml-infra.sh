#!/usr/bin/env bash
# Local Milvus / ml-infra inspect for OKE CD failures on
# "Wait for ml-infra dependencies" (status Unhealthy).
#
# Usage (with your kubeconfig pointing at the target cluster):
#   export KUBECONFIG=/path/to/your/kubeconfig   # optional if default
#   ./scripts/inspect-milvus-ml-infra.sh
set -euo pipefail

NS="${MILVUS_NAMESPACE:-ml-infra}"
MCP_NS="${MCP_NAMESPACE:-docs-agent}"

echo "== context =="
kubectl config current-context
kubectl cluster-info 2>/dev/null | head -n 2 || true

echo
echo "== milvus CR =="
kubectl get milvus -n "$NS" -o wide || true
kubectl get milvus milvus -n "$NS" -o yaml 2>/dev/null \
  | sed -n '/^status:/,/^[a-zA-Z]/p' | head -n 80 || true

echo
echo "== pods / services in ${NS} =="
kubectl get pods,svc,deploy,sts,pvc -n "$NS" -o wide || true

echo
echo "== milvus-related pods (describe + recent logs) =="
mapfile -t PODS < <(kubectl get pods -n "$NS" -o name 2>/dev/null | grep -iE 'milvus|etcd|minio|pulsar|kafka' || true)
if [ "${#PODS[@]}" -eq 0 ]; then
  echo "(no milvus/etcd/minio pods matched by name)"
  kubectl get pods -n "$NS" --field-selector=status.phase!=Running -o wide || true
else
  for p in "${PODS[@]}"; do
    echo "--- describe ${p} ---"
    kubectl describe -n "$NS" "$p" | tail -n 40 || true
    echo "--- logs ${p} (tail 80) ---"
    kubectl logs -n "$NS" "$p" --all-containers --tail=80 2>/dev/null || true
  done
fi

echo
echo "== embeddings InferenceService =="
kubectl get inferenceservice -n "$NS" 2>/dev/null || true
kubectl get pods -n "$NS" -l serving.kserve.io/inferenceservice=embeddings-service -o wide 2>/dev/null || true

echo
echo "== MCP deploy (${MCP_NS}) =="
kubectl get deploy,pods,svc -n "$MCP_NS" -l app=mcp-kubeflow-docs -o wide 2>/dev/null || \
  kubectl get deploy,pods,svc -n "$MCP_NS" 2>/dev/null | head -n 40 || true

echo
echo "== image pull events (mcp) =="
kubectl get events -n "$MCP_NS" --sort-by='.lastTimestamp' 2>/dev/null \
  | grep -iE 'mcp|pull|image|failed|error' | tail -n 30 || true

echo
echo "Done. If milvus STATUS is Unhealthy, check pod CrashLoop/Pending and PVC/storage next."
