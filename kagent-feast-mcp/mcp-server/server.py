import os

from fastmcp import FastMCP
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_USER = os.getenv("MILVUS_USER", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "Milvus")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "kubeflow_docs_docs_rag")
CODE_COLLECTION_NAME = os.getenv("CODE_COLLECTION_NAME", "code_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
PORT = int(os.getenv("PORT", "8000"))

mcp = FastMCP("Kubeflow Docs MCP Server")

model: SentenceTransformer = None
client: MilvusClient = None


def _init():
    global model, client
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)
    if client is None:
        client = MilvusClient(uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)


def _search_collection(collection_name: str, query: str, top_k: int, output_fields: list[str]) -> list[dict]:
    """Shared helper: encode query and search a Milvus collection."""
    _init()
    embedding = model.encode(query).tolist()
    hits = client.search(
        collection_name=collection_name,
        data=[embedding],
        limit=top_k,
        output_fields=output_fields,
    )[0]
    return hits


@mcp.tool()
def search_kubeflow_docs(query: str, top_k: int = 5) -> str:
    """Search Kubeflow documentation using semantic similarity.

    Args:
        query: The search query about Kubeflow.
        top_k: Number of results to return (default 5).

    Returns:
        Formatted search results with content and citation URLs.
    """
    hits = _search_collection(
        COLLECTION_NAME, query, top_k,
        ["content_text", "citation_url", "file_path"],
    )

    if not hits:
        return "No results found for your query."

    results = []
    for i, hit in enumerate(hits, 1):
        entity = hit["entity"]
        entry = f"### Result {i} (score: {hit['distance']:.4f})"
        entry += f"\n**Source:** {entity.get('citation_url', '')}"
        entry += f"\n**File:** {entity.get('file_path', '')}"
        entry += f"\n\n{entity.get('content_text', '')}\n"
        results.append(entry)

    return "\n---\n".join(results)


@mcp.tool()
def search_kubeflow_code(query: str, top_k: int = 5) -> str:
    """Search Kubeflow code and YAML manifests using semantic similarity.

    Use this tool when the user asks about Kubernetes manifests, YAML
    configurations, Deployments, Services, ConfigMaps, Python source code,
    or infrastructure definitions in the Kubeflow project.

    Args:
        query: The search query about Kubeflow code or manifests.
        top_k: Number of results to return (default 5).

    Returns:
        Formatted search results with code content, resource metadata,
        and citation URLs.
    """
    hits = _search_collection(
        CODE_COLLECTION_NAME, query, top_k,
        ["content_text", "citation_url", "file_path", "resource_kind",
         "resource_name", "resource_namespace", "file_type"],
    )

    if not hits:
        return "No code results found for your query."

    results = []
    for i, hit in enumerate(hits, 1):
        entity = hit["entity"]
        entry = f"### Result {i} (score: {hit['distance']:.4f})"
        entry += f"\n**Source:** {entity.get('citation_url', '')}"
        entry += f"\n**File:** {entity.get('file_path', '')}"

        # Show resource metadata when available
        kind = entity.get("resource_kind", "")
        name = entity.get("resource_name", "")
        ns = entity.get("resource_namespace", "")
        ftype = entity.get("file_type", "")
        if kind or name:
            entry += f"\n**Resource:** {kind}"
            if name:
                entry += f" `{name}`"
            if ns:
                entry += f" (namespace: {ns})"
        if ftype:
            entry += f"\n**Type:** {ftype}"

        entry += f"\n\n```\n{entity.get('content_text', '')}\n```\n"
        results.append(entry)

    return "\n---\n".join(results)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORT)
