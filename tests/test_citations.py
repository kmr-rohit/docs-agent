"""Tests for citation extraction helpers."""

import importlib.util
import sys
from pathlib import Path

CITATIONS_PATH = Path(__file__).parent.parent / "docs-agent-mcp" / "mcp-server" / "citations.py"
spec = importlib.util.spec_from_file_location("citations", CITATIONS_PATH)
citations = importlib.util.module_from_spec(spec)
sys.modules["citations"] = citations
spec.loader.exec_module(citations)


class TestExtractCitationUrls:
    def test_extracts_source_lines(self):
        text = (
            "### Result 1\n**Source:** https://www.kubeflow.org/docs/kserve/\n"
            "### Result 2\n**Source:** https://github.com/kubeflow/kubeflow/issues/1\n"
        )
        urls = citations.extract_citation_urls(text)
        assert urls == [
            "https://www.kubeflow.org/docs/kserve",
            "https://github.com/kubeflow/kubeflow/issues/1",
        ]

    def test_extracts_embedded_block(self):
        text = 'Answer text\n\n<!--KUBEFLOW_CITATIONS:["https://a.example","https://b.example"]-->'
        urls = citations.extract_citation_urls(text)
        assert urls == ["https://a.example", "https://b.example"]

    def test_dedupes_urls(self):
        text = (
            "**Source:** https://x.test\n"
            "<!--KUBEFLOW_CITATIONS:[\"https://x.test/\"]-->"
        )
        urls = citations.extract_citation_urls(text)
        assert urls == ["https://x.test"]


class TestFilePathToUrl:
    def test_maps_content_en_docs_path(self):
        url = citations.file_path_to_url("content/en/docs/components/pipelines/introduction.md")
        assert url == "https://www.kubeflow.org/docs/components/pipelines/introduction"

    def test_extracts_markdown_links(self):
        text = "See [pipelines](https://www.kubeflow.org/docs/components/pipelines/) for details."
        urls = citations.extract_citation_urls(text)
        assert "https://www.kubeflow.org/docs/components/pipelines" in urls


class TestAppendCitationsBlock:
    def test_appends_block_when_urls_present(self):
        result = citations.append_citations_block("body", ["https://x.test"])
        assert "body" in result
        assert "<!--KUBEFLOW_CITATIONS:" in result
        assert citations.extract_citation_urls(result) == ["https://x.test"]

    def test_no_block_when_empty(self):
        assert citations.append_citations_block("body", []) == "body"
