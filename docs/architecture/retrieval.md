# Retrieval

The MCP server turns a search query into evidence the model can quote and
citations the widget can show. It searches three Milvus collections, one per
source.

| Tool | Collection | Use it for |
|---|---|---|
| `search_kubeflow_docs` | `kubeflow_docs` | How-to and conceptual content, versions and releases |
| `search_github_issues` | `issues_rag` | Error messages, symptoms, "has anyone hit this" |
| `search_kubeflow_code` | `code_rag` | Defaults, CRD fields, manifests, controller behaviour |

The agent chooses the tool. Each tool's description says what it is for and what
it is not for, so the routing decision is explicit.

## Hybrid search and the intent router

Embeddings are good at meaning and weak at exact tokens such as version numbers,
config keys and error strings. The docs collection therefore stores both a dense
vector and a native BM25 sparse vector, and a rule-based intent router picks the
search mode for each query. No model is involved.

| Intent | Example | Mode | After search |
|---|---|---|---|
| `comparison` | "compare 1.9 and 1.10" | hybrid, deeper candidate pool | – |
| `release_date` | "when was 1.10 released" | BM25 | prefer chunks for that version |
| `temporal` | "what is the latest release" | BM25 on release notes | sort by release date |
| `exact` | a version, a dotted config key, an error string | BM25 | – |
| `conceptual` | "how", "why", "explain" | hybrid | – |
| `general` | anything else | hybrid | – |

Hybrid search fuses the dense and BM25 results with reciprocal rank fusion
(`hybridRanker: rrf`; a weighted ranker is available). A collection without the
BM25 field falls back to dense search. The chosen intent, mode and reason are
returned with every result and recorded in the trace.

## Collection schema (docs)

| Field | Purpose |
|---|---|
| `content_text` | Chunk text; also the BM25 input |
| `vector` | 768-dimension dense embedding (MPNet, from the TEI service) |
| `sparse_vector` | BM25 output, computed by Milvus |
| `document_id`, `chunk_index` | Stable document key and chunk position |
| `title`, `section_path` | Page title and heading breadcrumb |
| `citation_url`, `file_path` | Public URL and source path |
| `doc_type`, `version`, `release_date` | Release notes vs docs, the Kubeflow version, and its release date |

Indexes: a flat cosine index on `vector` and an inverted BM25 index on
`sparse_vector`.

## The citation contract

| Part | Responsibility |
|---|---|
| MCP evidence | Chunk text with URLs removed, labelled `[c1]`, `[c2]`, … |
| MCP structured result | `citations`: a list of `{id, url, score, section, version, …}` |
| Agent | Answers from the evidence and never prints URLs or source lists |
| Widget | Renders the Sources panel from the structured citations |

Because the model never sees a URL, it cannot invent one. Links always come from
the index.

## Code

- `docs-agent-mcp/mcp-server/server.py`: the three MCP tools
- `docs-agent-mcp/mcp-server/intent_router.py`: intent rules
- `docs-agent-mcp/mcp-server/milvus_search.py`: dense, BM25 and hybrid search
- `docs-agent-mcp/mcp-server/citations.py`: evidence and citation formatting
