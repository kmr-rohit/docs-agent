"""Citation extraction and structured metadata for RAG tool results."""

from __future__ import annotations

import json
import re

SOURCE_LINE_RE = re.compile(r"\*\*Source:\*\*\s*(\S+)")
CITATIONS_BLOCK_RE = re.compile(r"<!--KUBEFLOW_CITATIONS:(\[.*?\])-->")


def _normalize_url(url: str) -> str:
    cleaned = url.rstrip(")>.,;")
    if cleaned.endswith("/") and "://" in cleaned:
        # Treat https://host/path and https://host/path/ as the same citation.
        cleaned = cleaned.rstrip("/")
    return cleaned


def dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        normalized = _normalize_url(url.strip())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def extract_citation_urls(text: str) -> list[str]:
    """Extract citation URLs from MCP markdown or embedded metadata blocks."""
    if not text:
        return []

    urls: list[str] = []
    urls.extend(_normalize_url(match.group(1)) for match in SOURCE_LINE_RE.finditer(text))

    block_match = CITATIONS_BLOCK_RE.search(text)
    if block_match:
        try:
            block_urls = json.loads(block_match.group(1))
        except json.JSONDecodeError:
            block_urls = []
        if isinstance(block_urls, list):
            urls.extend(str(item).strip() for item in block_urls if item)

    return dedupe_urls(urls)


def append_citations_block(text: str, urls: list[str]) -> str:
    """Append machine-readable citation metadata for frontend extraction."""
    unique_urls = dedupe_urls(urls)
    if not unique_urls:
        return text
    payload = json.dumps(unique_urls, separators=(",", ":"))
    return f"{text}\n\n<!--KUBEFLOW_CITATIONS:{payload}-->"
