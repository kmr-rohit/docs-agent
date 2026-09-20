"""Tests for issue-triage area maps and title parsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "select-triage-context.py"
AREA_MAP = REPO_ROOT / "docs" / "agents" / "area-map.json"


def _load_selector():
    spec = importlib.util.spec_from_file_location("select_triage_context", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def selector():
    return _load_selector()


@pytest.fixture(scope="module")
def area_map(selector):
    return selector.load_area_map(AREA_MAP)


class TestAreaMapIntegrity:
    def test_json_loads(self, area_map):
        assert "areas" in area_map
        assert "mcp" in area_map["areas"]
        assert area_map["areas"]["mcp"]["label"] == "area/mcp"

    def test_always_read_files_exist(self, area_map):
        missing = [path for path in area_map["always_read"] if not (REPO_ROOT / path).exists()]
        assert missing == []

    def test_every_area_file_exists(self, area_map):
        missing = []
        for area, meta in area_map["areas"].items():
            for rel in meta["files"]:
                if not (REPO_ROOT / rel).exists():
                    missing.append(f"{area}: {rel}")
        assert missing == []

    def test_area_and_kind_labels_are_allowlisted(self, area_map):
        allowed = set(area_map["allowed_labels"])
        for meta in area_map["types"].values():
            assert meta["kind_label"] in allowed
            for extra in meta.get("extra_labels") or []:
                assert extra in allowed
        for area, meta in area_map["areas"].items():
            assert meta["label"] in allowed, area


class TestTitleSelection:
    def test_mcp_bug_selects_server_and_architecture(self, selector, area_map):
        result = selector.select_context(
            "bug(mcp): search_kubeflow_docs returns Search failed on empty collection",
            area_map,
        )
        assert result["valid"] is True
        assert result["issue_type"] == "bug"
        assert result["issue_area"] == "mcp"
        assert result["kind_label"] == "kind/bug"
        assert result["area_label"] == "area/mcp"
        assert "docs/agents/architecture.md" in result["context_files"]
        assert "docs-agent-mcp/mcp-server/server.py" in result["context_files"]

    def test_mcp_server_alias_resolves_to_mcp(self, selector, area_map):
        result = selector.select_context("feat(mcp-server): add thin context mode", area_map)
        assert result["valid"] is True
        assert result["issue_area"] == "mcp"
        assert result["area_label"] == "area/mcp"

    def test_frontend_security_uses_kind_security(self, selector, area_map):
        result = selector.select_context(
            "security(frontend): citations allow javascript URLs",
            area_map,
        )
        assert result["valid"] is True
        assert result["kind_label"] == "kind/security"
        assert result["area_label"] == "area/frontend"
        assert "frontend/docs_scripts/chatbot.js" in result["context_files"]

    def test_pipelines_selects_ingestion_sources(self, selector, area_map):
        result = selector.select_context(
            "bug(pipelines): issues chunk_size exceeds TEI limit",
            area_map,
        )
        assert result["issue_area"] == "pipelines"
        assert "docs-agent-mcp/pipelines/issues_utils.py" in result["context_files"]
        assert "docs-agent-mcp/pipelines/utils.py" in result["context_files"]

    def test_tei_alias_resolves_to_embeddings(self, selector, area_map):
        result = selector.select_context("bug(tei): 413 on dense YAML chunks", area_map)
        assert result["valid"] is True
        assert result["issue_area"] == "embeddings"
        assert result["area_label"] == "area/embeddings"
        assert "docs-agent-mcp/mcp-server/embeddings_client.py" in result["context_files"]

    def test_invalid_title_does_not_run_area_lookup(self, selector, area_map):
        result = selector.select_context("Broken chatbot JWT", area_map)
        assert result["valid"] is False
        assert result["issue_area"] == ""
        assert result["context_files"] == area_map["always_read"]

    def test_unknown_area_stays_valid_with_architecture_only(self, selector, area_map):
        result = selector.select_context("chore(widgets): rename overlay class", area_map)
        assert result["valid"] is True
        assert result["issue_area"] == "widgets"
        assert result["area_label"] == ""
        assert result["context_files"] == area_map["always_read"]


class TestSelectorCli:
    def test_check_paths_succeeds_for_mcp(self, selector, area_map):
        result = selector.select_context("bug(mcp): tool handshake drops session", area_map)
        assert selector.missing_paths(result, REPO_ROOT) == []


class TestWorkflowDocs:
    def test_docs_agent_workflow_requires_architecture_and_area_mcp(self):
        source = (REPO_ROOT / ".github/workflows/issue-triage.md").read_text(encoding="utf-8")
        assert "docs/agents/architecture.md" in source
        assert "area/mcp" in source
        assert "safe-outputs:" in source

    def test_compiled_lockfile_allows_area_mcp(self):
        lock = (REPO_ROOT / ".github/workflows/issue-triage.lock.yml").read_text(encoding="utf-8")
        assert "area/mcp" in lock
        assert "runtime-import .github/workflows/issue-triage.md" in lock
        assert "docs-agent-mcp/mcp-server/" in lock

    def test_kfp_pack_tells_shristi_to_checkout_architecture(self):
        pack = (REPO_ROOT / "docs/agents/kfp-issue-triage.md").read_text(encoding="utf-8")
        assert "checkout: false" in pack
        assert "docs/agents/architecture.md" in pack
        assert "area/backend" in pack
        assert "area/sdk" in pack
        assert "sparse-checkout" in pack
