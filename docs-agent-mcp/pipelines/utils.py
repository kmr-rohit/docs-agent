"""Shared utility functions for pipeline components."""

import os
import re


def resolve_github_token(github_token: str = "") -> str:
    """Resolve a GitHub PAT from the pipeline parameter or environment.

    Checks ``github_token`` first, then ``Github_Pat`` (repo/OKE secret name),
    then ``GITHUB_TOKEN`` for compatibility with other tooling.
    """
    for candidate in (
        github_token,
        os.environ.get("Github_Pat", ""),
        os.environ.get("GITHUB_TOKEN", ""),
    ):
        if candidate and candidate.strip():
            return candidate.strip()
    return ""


def clean_content(content: str) -> str:
    """Clean raw document content for better embeddings.

    Removes Hugo frontmatter, template syntax, HTML tags, navigation
    artifacts, URLs, and normalizes whitespace.

    Args:
        content: Raw document content (markdown/HTML).

    Returns:
        Cleaned text suitable for embedding.
    """
    # Remove Hugo frontmatter (both --- and +++ styles)
    # \A anchors to absolute start of string; backreference ensures matching delimiters
    content = re.sub(
        r'\A\s*(?P<delim>-{3,}|\+{3,}).*?(?P=delim)\s*', '', content,
        flags=re.DOTALL
    )

    # Remove Hugo template syntax
    content = re.sub(r'\{\{.*?\}\}', '', content, flags=re.DOTALL)

    # Remove HTML comments and tags
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    content = re.sub(r'<[^>]+>', ' ', content)

    # Remove navigation/menu artifacts
    content = re.sub(
        r'\b(Get Started|Contribute|GenAI|Home|Menu|Navigation)\b', '',
        content, flags=re.IGNORECASE
    )

    # Clean up URLs and links
    content = re.sub(r'https?://[^\s]+', '', content)
    content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)

    # Remove excessive whitespace and normalize
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    content = content.strip()

    return content
