#!/usr/bin/env bash
# Remove legacy Helm Milvus (my-release-*) from docs-agent.
# Current stack uses milvus-milvus in ml-infra (Terraform / Milvus operator).
set -euo pipefail

NS="${DOCS_AGENT_NS:-docs-agent}"

echo "Legacy Milvus in ${NS} is unused by MCP and pipelines (ml-infra is canonical)."
echo "Deleting my-release workloads and services in ${NS}..."

if command -v helm >/dev/null 2>&1; then
  helm uninstall my-release -n "${NS}" --ignore-not-found 2>/dev/null || true
fi

kubectl delete deployment -n "${NS}" \
  my-release-milvus-standalone my-release-minio \
  --ignore-not-found=true
kubectl delete statefulset -n "${NS}" my-release-etcd --ignore-not-found=true
kubectl delete svc -n "${NS}" \
  my-release-milvus my-release-minio my-release-etcd my-release-etcd-headless \
  --ignore-not-found=true
kubectl delete pvc -n "${NS}" -l app.kubernetes.io/instance=my-release --ignore-not-found=true 2>/dev/null || true

echo "Done. Remaining Milvus: kubectl get svc -n ml-infra milvus-milvus"
