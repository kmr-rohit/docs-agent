#!/usr/bin/env bash
# Sync GitHub labels for docs-agent community triage.
# Usage: ./scripts/sync-github-labels.sh [owner/repo]   (default: current gh repo)
set -euo pipefail

REPO="${1:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"

create_label() {
  local name="$1" color="$2" description="$3"
  if gh label list --repo "$REPO" --json name -q '.[].name' | grep -Fxq "$name"; then
    gh label edit "$name" --repo "$REPO" --color "$color" --description "$description" 2>/dev/null || true
  else
    gh label create "$name" --repo "$REPO" --color "$color" --description "$description"
  fi
}

create_label "good first issue" "0E8A16" "Small, scoped task for new contributors"
create_label "help wanted"       "1D76DB" "Maintainers want community help"
create_label "gsoc-2026"         "5319E7" "Tied to gsoc2026_agentic_rag.md"
create_label "maintainer-only"   "B60205" "Agentic RAG core — discuss before PR"
create_label "kind/bug"          "D93F0B" "Something is broken"
create_label "area/tests"        "FBCA04" "pytest, coverage, smoke tests"
create_label "area/ci"           "FBCA04" "GitHub Actions workflows"
create_label "area/docs"         "0075CA" "README, runbooks, contributing"
create_label "area/pipelines"    "1D76DB" "KFP ingestion pipelines"
create_label "area/mcp-server"   "1D76DB" "MCP retrieval server"
create_label "area/frontend"     "E99695" "Website chat widget"
create_label "area/infra"        "C5DEF5" "Terraform, manifests, Helm"
create_label "size/S"            "EDEDED" "Few hours"
create_label "size/M"            "EDEDED" "About 1–2 days"
create_label "size/L"            "EDEDED" "Multi-day effort"

echo "Labels synced on $REPO"
