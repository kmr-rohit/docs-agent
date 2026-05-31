#!/usr/bin/env bash
# Tear down replaced app workloads in docs-agent, apply Terraform (ml-infra stack),
# then rely on oke-cicd or manual kubectl for MCP/Kagent.
#
# Does NOT delete shared cluster control planes (kubeflow, istio-system, etc.).
#
# Usage:
#   ./scripts/redeploy-ml-infra-stack.sh teardown
#   ./scripts/redeploy-ml-infra-stack.sh terraform-plan
#   ./scripts/redeploy-ml-infra-stack.sh terraform-apply
#   ./scripts/redeploy-ml-infra-stack.sh apply-manifests
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="${ROOT_DIR}/docs-agent-mcp/terraform"
DOCS_AGENT_NS="${DOCS_AGENT_NS:-docs-agent}"

teardown_docs_agent() {
  echo "==> Removing replaced workloads in namespace ${DOCS_AGENT_NS} (not namespace itself)"
  kubectl delete deployment/mcp-kubeflow-docs -n "${DOCS_AGENT_NS}" --ignore-not-found=true
  kubectl delete inferenceservice/qwen -n "${DOCS_AGENT_NS}" --ignore-not-found=true
  kubectl delete svc/qwen-llm -n "${DOCS_AGENT_NS}" --ignore-not-found=true
  kubectl delete -f "${ROOT_DIR}/legacy/manifests/qwen-llm-service.yaml" --ignore-not-found=true 2>/dev/null || true

  echo "==> Optional: remove legacy Helm Milvus in docs-agent (uncomment if migrating)"
  # helm uninstall my-release -n "${DOCS_AGENT_NS}" || true
}

terraform_plan() {
  cd "${TF_DIR}"
  terraform init -input=false
  terraform plan -input=false
}

terraform_apply() {
  cd "${TF_DIR}"
  terraform init -input=false
  terraform apply -input=false -auto-approve
}

apply_k8s_manifests() {
  echo "==> Apply Qwen + stable Service in ml-infra"
  kubectl apply -f "${ROOT_DIR}/docs-agent-mcp/manifests/vllm/kserve-qwen.yaml"

  echo "==> Apply Istio allow policies (if not fully owned by Terraform yet)"
  kubectl apply -f "${ROOT_DIR}/docs-agent-mcp/manifests/istio/" || true
}

case "${1:-}" in
  teardown) teardown_docs_agent ;;
  terraform-plan) terraform_plan ;;
  terraform-apply) terraform_apply ;;
  apply-manifests) apply_k8s_manifests ;;
  *)
    echo "Usage: $0 {teardown|terraform-plan|terraform-apply|apply-manifests}"
    exit 1
    ;;
esac
