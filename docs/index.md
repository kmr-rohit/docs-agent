# Flo, the Kubeflow docs assistant

Flo answers questions about Kubeflow from the official documentation, GitHub
issues and source code, and shows the sources it used. It is an agentic RAG
system that runs entirely on Kubernetes: the agent decides which search tool to
call, the tools search a vector database, and a self-hosted model writes the
answer.

The project is `kubeflow/docs-agent`. It started as a Google Summer of Code 2026
project and is becoming a subproject of the Kubeflow Agents working group.

<div class="grid cards" markdown>

-   :material-sitemap: **Architecture**

    ---

    How a question becomes an answer: agents, MCP tools, hybrid retrieval,
    models, and the public gateway.

    [:octicons-arrow-right-24: Architecture](architecture/index.md)

-   :material-source-pull: **Contributing**

    ---

    How to make a change, how it reaches users, and how to see what it did to
    answer quality.

    [:octicons-arrow-right-24: Contributing](contributing/index.md)

-   :material-chart-line: **Evaluation and feedback**

    ---

    The eval that runs on every merge, and how thumbs-down from real users turn
    into issues.

    [:octicons-arrow-right-24: Evaluation](evaluation/index.md)

-   :material-kubernetes: **Operations**

    ---

    Environments and promotion, Argo CD, tracing and data retention, ingestion.

    [:octicons-arrow-right-24: Operations](operations/index.md)

</div>

## At a glance

| Part | What it is |
|---|---|
| Agents | [kagent](https://kagent.dev) `Agent` resources: a docs agent (thinking off and on) and a debug agent |
| Tools | An MCP server with `search_kubeflow_docs`, `search_github_issues`, `search_kubeflow_code` |
| Retrieval | Milvus, hybrid dense + BM25 search chosen per query by an intent router |
| Models | Gemma 4 E4B answers; Qwen2.5-7B judges evaluations. Both served by KServe/vLLM, one GPU each |
| Edge | Istio gateway with TLS, rate limits, CORS allow-list and anonymous session tokens |
| Delivery | Argo CD; staging deploys every commit on `main`, production is promoted to a tested commit every week or two |
| Quality | A 58-question end-to-end eval on every merge; user feedback with nightly triage |
| Privacy | Anonymous sessions; conversations and feedback kept 4 weeks ([what we collect](privacy.md)) |
| Tracing | OpenTelemetry and Langfuse, with live and evaluation traces kept apart |

## The path of one question

```mermaid
sequenceDiagram
    autonumber
    participant U as User (docs site widget)
    participant G as Istio gateway
    participant A as kagent docs agent
    participant L as Gemma (KServe)
    participant M as MCP server
    participant V as Milvus
    U->>G: POST /api/session (anonymous token)
    U->>G: A2A message/stream + token
    G->>A: route (rate-limited, token checked)
    A->>L: question + tool schemas
    L-->>A: call search_kubeflow_docs(query)
    A->>M: tool call
    M->>V: intent router: BM25, hybrid or dense
    V-->>M: chunks
    M-->>A: evidence + structured citations
    A->>L: evidence
    L-->>A: grounded answer
    A-->>U: streamed answer, Sources panel, 👍 / 👎
```
