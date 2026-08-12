# docs-agent architecture — current + planned improvements

**Legend**

| Style | Meaning |
|-------|---------|
| **Solid boxes / solid arrows** | Shipped today (post-#210) |
| **Dotted boxes / dotted arrows** | Planned improvements (Phase 3 / roadmap) |

Related: [gsoc2026_agentic_rag.md](../../gsoc2026_agentic_rag.md) · [CONTRIBUTION_MAP.md](./CONTRIBUTION_MAP.md)

---

## Combined diagram (current + future overlay)

Copy into slides, GitHub, or any Mermaid renderer.

```mermaid
flowchart TB
  %% ─── Users ───
  U_WEB["🌐 Website user<br/>frontend/chatbot.js"]
  U_IDE["💻 IDE user<br/>Cursor / MCP client"]

  %% ─── Planned: edge security ───
  OAUTH["🔒 OAuth2 Proxy<br/>login + session"]
  RATE["⏱️ Rate limit / quotas<br/>LiteLLM or kgateway"]

  %% ─── docs-agent namespace (current) ───
  subgraph NS_DA["docs-agent namespace — CURRENT"]
    KAG["Kagent UI + Runner"]
    AGENT["Agent CRD"]
    MC["ModelConfig"]
    MCP["MCP server :8000<br/>3 search tools"]
    RMS["RemoteMCPServer"]
  end

  %% ─── Planned: agent hardening ───
  ROUTER["🧭 Semantic router<br/>LangGraph / ADK"]
  MCP_AUTH["🔑 MCP API key / JWT gate"]

  %% ─── ml-infra (current) ───
  subgraph NS_ML["ml-infra namespace — CURRENT"]
    QWEN["KServe Qwen-2.5-14B"]
    TEI["KServe TEI embeddings<br/>all-mpnet-base-v2"]
    MILV["Milvus standalone"]
  end

  %% ─── Planned: LLM safety ───
  GUARD_IN["🛡️ Input classifier<br/>prompt injection"]
  GUARD_OUT["🛡️ Llama-Guard KServe<br/>output filter"]

  %% ─── kubeflow (current) ───
  subgraph NS_KF["kubeflow namespace — CURRENT"]
    KFP["Kubeflow Pipelines"]
    P_DOCS["docs pipeline"]
    P_ISS["issues pipeline"]
    P_CODE["code pipeline"]
  end

  %% ─── Planned: pipeline hardening ───
  P_INC["📦 incremental pipeline → TEI"]
  P_SIG["✅ repo signature verify"]
  P_SCHED["🕐 scheduled / incremental ingest"]

  %% ─── Collections ───
  C_DOCS[("kubeflow_docs")]
  C_ISS[("issues_rag")]
  C_CODE[("code_rag<br/>⚠ may be empty")]

  %% ─── Planned: data / eval ───
  EVAL["📊 RAGAS / golden dataset"]
  FEED["👍 Feedback webhook<br/>→ eval store"]

  %% ─── Planned: platform ───
  ARGO["🔄 Argo CD GitOps"]
  HELM["📋 Helm umbrella chart"]
  OTEL["📡 OpenTelemetry traces"]
  ESO["🔐 External Secrets"]
  SCAN["🔍 Trivy / SonarQube CI"]

  %% ─── CI today ───
  GHA["GitHub Actions<br/>test + build image"]
  KUBECTL["kubectl apply<br/>operator forks only"]

  %% ═══ CURRENT PATHS (solid) ═══
  U_WEB ==>|A2A JSON-RPC| KAG
  U_IDE ==>|streamable HTTP| MCP
  KAG ==> AGENT
  AGENT ==> RMS
  RMS ==> MCP
  AGENT ==> MC
  MC ==>|OpenAI API| QWEN
  MCP ==>|embed query| TEI
  MCP ==>|vector search| MILV
  KFP ==> P_DOCS & P_ISS & P_CODE
  P_DOCS ==> C_DOCS
  P_ISS ==> C_ISS
  P_CODE ==> C_CODE
  C_DOCS & C_ISS & C_CODE -.-> MILV
  GHA ==> KUBECTL
  KUBECTL ==> MCP

  %% ═══ PLANNED PATHS (dotted) ═══
  U_WEB -.->|future| OAUTH
  U_IDE -.->|future| OAUTH
  OAUTH -.-> RATE
  RATE -.-> KAG
  RATE -.-> MCP
  MC -.->|via gateway| RATE
  RATE -.-> GUARD_IN
  GUARD_IN -.-> QWEN
  QWEN -.-> GUARD_OUT
  GUARD_OUT -.-> KAG
  AGENT -.-> ROUTER
  ROUTER -.-> RMS
  MCP_AUTH -.-> MCP
  P_INC -.-> KFP
  P_SIG -.-> P_DOCS & P_CODE
  P_SCHED -.-> KFP
  U_WEB -.-> FEED
  FEED -.-> EVAL
  MCP -.-> EVAL
  ARGO -.->|sync| MCP & KAG
  HELM -.->|package| MCP & KAG
  OTEL -.-> KAG & MCP & QWEN
  ESO -.-> MCP
  GHA -.-> SCAN
  GHA -.->|build only| ARGO

  %% ═══ STYLES ═══
  classDef current fill:#e8f4fc,stroke:#1a73e8,stroke-width:2px,color:#111
  classDef future fill:#fafafa,stroke:#888,stroke-width:2px,stroke-dasharray:6 4,color:#444
  classDef data fill:#fff3e0,stroke:#e65100,stroke-width:2px
  classDef warn fill:#fff8e1,stroke:#f9a825,stroke-width:2px,stroke-dasharray:6 4

  class KAG,AGENT,MC,MCP,RMS,QWEN,TEI,MILV,KFP,P_DOCS,P_ISS,P_CODE,GHA,KUBECTL,U_WEB,U_IDE current
  class OAUTH,RATE,ROUTER,MCP_AUTH,GUARD_IN,GUARD_OUT,P_INC,P_SIG,P_SCHED,EVAL,FEED,ARGO,HELM,OTEL,ESO,SCAN future
  class C_DOCS,C_ISS,C_CODE data
  class C_CODE warn
```

---

## Simplified slide version (one glance)

```mermaid
flowchart LR
  subgraph users["Users"]
    WEB[Website]
    IDE[IDE / MCP]
  end

  subgraph edge_future["🔲 PLANNED: Edge"]
    AUTH[OAuth2]
    GW[LLM Gateway<br/>rate limits]
  end

  subgraph app["✅ CURRENT: docs-agent"]
    KA[Kagent + Agent]
    M[MCP 3 tools]
  end

  subgraph ml["✅ CURRENT: ml-infra"]
    L[Qwen LLM]
    E[TEI embed]
    V[Milvus]
  end

  subgraph ingest["✅ CURRENT: ingest"]
    KFP[KFP pipelines]
  end

  subgraph ingest_future["🔲 PLANNED: pipelines"]
    H[TEI incremental<br/>sig verify<br/>code_rag fill]
  end

  subgraph ops_future["🔲 PLANNED: ops"]
    AC[Argo CD]
    HM[Helm charts]
    OB[OTel + evals]
  end

  WEB --> KA
  IDE --> M
  KA --> M
  KA --> L
  M --> E
  M --> V
  KFP --> V

  WEB -.-> AUTH
  IDE -.-> AUTH
  AUTH -.-> GW
  GW -.-> KA
  GW -.-> L
  KFP -.-> H
  M -.-> AC
  KA -.-> OB

  classDef solid fill:#dbeafe,stroke:#2563eb
  classDef dash fill:#f3f4f6,stroke:#6b7280,stroke-dasharray:6 4
  class WEB,IDE,KA,M,L,E,V,KFP solid
  class AUTH,GW,H,AC,HM,OB dash
```

---

## Layer view (current vs planned)

```mermaid
flowchart TB
  subgraph L1["Layer 1 — Access"]
    direction LR
    L1C["✅ Frontend widget + in-cluster MCP"]
    L1F["🔲 OAuth2 · rate limits · MCP API keys"]
  end

  subgraph L2["Layer 2 — Agent"]
    direction LR
    L2C["✅ Kagent Agent · systemMessage routing"]
    L2F["🔲 Semantic router · guardrails · conversation DB"]
  end

  subgraph L3["Layer 3 — Tools & LLM"]
    direction LR
    L3C["✅ MCP retrieval · direct KServe Qwen"]
    L3F["🔲 LLM gateway · input/output filters · retries"]
  end

  subgraph L4["Layer 4 — Data"]
    direction LR
    L4C["✅ Milvus · TEI · 3 collections"]
    L4F["🔲 code_rag population · collection RBAC · eval store"]
  end

  subgraph L5["Layer 5 — Ingestion"]
    direction LR
    L5C["✅ docs / issues / code pipelines"]
    L5F["🔲 incremental TEI · signed sources · schedules"]
  end

  subgraph L6["Layer 6 — Delivery"]
    direction LR
    L6C["✅ GHA test/build · kubectl deploy forks"]
    L6F["🔲 Argo CD · Helm · Trivy · terraform/kubeconform CI"]
  end

  L1C --> L2C --> L3C --> L4C
  L5C --> L4C
  L6C --> L1C

  L1F -.-> L1C
  L2F -.-> L2C
  L3F -.-> L3C
  L4F -.-> L4C
  L5F -.-> L5C
  L6F -.-> L6C

  classDef c fill:#e0f2fe,stroke:#0284c7
  classDef f fill:#f9fafb,stroke:#9ca3af,stroke-dasharray:6 4
  class L1C,L2C,L3C,L4C,L5C,L6C c
  class L1F,L2F,L3F,L4F,L5F,L6F f
```

---

## ASCII reference (no Mermaid needed)

### Current (solid)

```
 Website / IDE
      │
      ▼
 ┌─ docs-agent ─────────────────────────┐
 │ Kagent ──► Agent ──► MCP (:8000)    │
 │    │              search x3          │
 │    └── ModelConfig ─────────────┐    │
 └────────────────────────────────│────┘
                                  ▼
 ┌─ ml-infra ───────────────────────────┐
 │ Qwen LLM ◄──────────────────────────┘
 │ TEI embeddings ◄── MCP (query embed)
 │ Milvus ◄────────── MCP (search)
 │     ▲
 │     │ kubeflow_docs | issues_rag | code_rag
 └─────┼────────────────────────────────
       │
 ┌─ kubeflow ─ KFP pipelines (docs / issues / code)
 └────────────────────────────────────────

 CI: GHA → pytest → build image → kubectl apply (fork)
```

### Planned improvements (dotted)

```
 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 :  [OAuth2] ──► [LLM Gateway: rate limit · token quota · audit log]    :
 :       │                    │                                         :
 :       └──────────► Kagent / MCP (authenticated)                      :
 :                                                                      :
 :  [Input guard] ──► Qwen ──► [Llama-Guard out] ──► user               :
 :  [Semantic router] ──► tool choice (docs vs issues vs code)         :
 :  [MCP API key] on :8000 if exposed beyond mesh                        :
 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 :  Pipelines: incremental→TEI · signed GitHub clones · scheduled runs :
 :  Data: fill code_rag · RAGAS eval · thumbs-up golden dataset        :
 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 :  Ops: Argo CD sync · Helm charts · External Secrets · OTel traces    :
 :  CI: Trivy/SonarQube · kubeconform · terraform validate              :
 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

---

## Improvement backlog by area

| Area | Planned (dotted) | Owner | Community? |
|------|------------------|-------|------------|
| **Security / edge** | OAuth2 proxy, MCP API key, rate limits | Maintainer design | Implement after design issue |
| **LLM gateway** | LiteLLM or kgateway, quotas, logging | Maintainer design | Help with Helm deploy |
| **Guardrails** | Input classifier, Llama-Guard KServe | Maintainer | KServe manifest PRs |
| **Agent core** | Semantic router, prompt hardening | **Maintainer only** | — |
| **MCP** | TEI retry/backoff, structured errors | Maintainer review | ✅ small PRs |
| **Pipelines** | incremental→TEI, sig verify, `code_rag` run | Maintainer + ops | ✅ runbook, TEI migration |
| **Frontend** | Feedback UI, tool-step transparency | Maintainer UX direction | ✅ implementation |
| **Data / eval** | RAGAS, golden dataset from feedback | Maintainer | ✅ eval scripts |
| **GitOps** | Argo CD app-of-apps | Maintainer + Platform WG | ✅ Helm scaffold |
| **IaC** | Helm replaces app-layer TF; dedupe Istio YAML | Maintainer | ✅ packaging |
| **Observability** | OpenTelemetry end-to-end | Maintainer | ✅ instrumentation PRs |
| **CI** | Trivy, kubeconform, terraform validate | — | ✅ good first issues |

---

## This week (mapped to diagram)

| Diagram box | Action |
|-------------|--------|
| `code_rag` (warn) | Run code pipeline — turns dotted “empty” into solid data |
| `FEED` / `EVAL` | File community issue for thumbs up/down |
| `ARGO` / `HELM` | Maintainer spike doc only — stay dotted |
| `GW` / `OAUTH` | Open `maintainer-only` design issue — stay dotted |
| Tests on `MCP` / `TEI` | Community PRs — harden without changing arch |
