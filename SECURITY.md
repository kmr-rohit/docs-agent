# Security — docs-agent stack

This document describes known risks, mitigations, and operational checks for the **Kagent + MCP + Milvus + TEI + Qwen** deployment on OKE. It is intended for reviewers, mentors, and operators running break-the-system or penetration-style tests.

**Reporting issues:** Open a private security advisory on GitHub or contact the repo maintainers. Do not file public issues with live credentials, cluster IPs, or PATs.

---

## Architecture and exposure

| Component | Namespace | Typical exposure | Auth today |
|-----------|-----------|------------------|------------|
| Kagent UI / A2A | `docs-agent` | **Public** via `kagent-ui-lb` (LoadBalancer) or optional Istio ingress | **None** on A2A |
| MCP server (`/mcp`) | `docs-agent` | **ClusterIP only** (not on public internet) | **None** |
| Milvus | `ml-infra` | Internal (`19530`) | `root` + secret password |
| TEI embeddings | `ml-infra` | Internal | **None** |
| Qwen LLM (KServe) | `ml-infra` | Internal | Dummy `local-kserve` key in-cluster |
| Kubeflow Pipelines | `kubeflow` | KFP UI (cluster-specific) | Dex / cluster auth |

**Important:** The MCP service is not published directly, but **any caller who can reach Kagent’s public A2A endpoint** can cause the agent to invoke MCP tools (`search_kubeflow_docs`, `search_github_issues`, `search_kubeflow_code`) and thus read indexed RAG content and consume GPU/embeddings indirectly.

---

## Optional ingress (Terraform)

File: `docs-agent-mcp/terraform/kagent_ingress.tf`  
Example vars: `docs-agent-mcp/terraform/terraform.tfvars.example`

### Purpose

Alternative to (or in addition to) the raw **`kagent-ui-lb`** LoadBalancer:

- TLS via **cert-manager** + **Let’s Encrypt** (HTTP-01)
- Stable hostname (`kagent_domain_name`) instead of a bare IP
- **CORS** rules on the Istio `VirtualService` for browser clients (e.g. Vercel, kubeflow.org)

### Default: disabled

```hcl
enable_kagent_ingress = false   # default — no Gateway/Certificate/VS created
```

When `false`, only existing exposure paths apply (e.g. LoadBalancer). When `true`, you must set DNS, email, and tight CORS before `terraform apply`.

### Resources created when enabled

| Resource | Role |
|----------|------|
| `ClusterIssuer` (`letsencrypt-prod`) | ACME account for TLS certificates |
| `Certificate` (`kagent-ui-cert`) | TLS secret in `istio-system` |
| `Gateway` (`kagent-gateway`) | HTTP → HTTPS redirect, TLS on 443 |
| `VirtualService` (`kagent-ui-routing`) | Routes host → `kagent-ui:8080`, applies `corsPolicy` |

### Variables

| Variable | Description |
|----------|-------------|
| `enable_kagent_ingress` | Master switch (default `false`) |
| `kagent_domain_name` | FQDN on certificate and Gateway |
| `kagent_acme_email` | Let’s Encrypt registration email |
| `kagent_cors_allow_origin_regexes` | Istio `allowOrigins` regex list (default `[".*"]` — **insecure for production**) |

### Ingress security notes

1. **Ingress does not add authentication** — it only adds TLS, routing, and CORS.
2. **Default CORS `.*`** allows any website to trigger browser calls to your agent if users visit a malicious page while your agent URL is public — combine with **auth** or **do not use `.*` in production**.
3. **HTTP-01 challenges** expose temporary challenge paths on the ingress; use a dedicated hostname and monitor cert-manager logs.
4. **ClusterIssuer name** `letsencrypt-prod` is cluster-scoped — avoid name collisions in shared clusters.
5. MCP remains **ClusterIP**; ingress targets **Kagent UI only**, not `mcp-kubeflow-docs:8000`.

### Enabling safely (checklist)

1. Point `kagent_domain_name` DNS to the Istio ingress external IP.
2. Set `kagent_acme_email` to a monitored mailbox.
3. Replace default CORS with explicit regexes, e.g. only your Vercel app and `kubeflow.org`.
4. Add **OAuth2 proxy / API gateway / rate limits** in front of Kagent before wide announcement.
5. Prefer **HTTPS** for any `https://` frontend (Vercel) calling the agent (avoid mixed content).

---

## Known vulnerabilities and abuse scenarios

Severity is relative to a **public demo cluster** with LoadBalancer + no auth. Adjust if your deployment is private/VPN-only.

### Critical / high

#### 1. Unauthenticated public Kagent (A2A)

**Risk:** Anyone with the LoadBalancer IP or hostname can stream chat, force tool calls, exhaust **Qwen GPU**, and retrieve content from **docs_rag**, **issues_rag**, and **code_rag** via the agent.

**Attacker actions:** Automated `message/stream` requests, long contexts, parallel sessions, prompt injection to maximize tool usage.

**Mitigations:**

- Do not expose `kagent-ui-lb` to `0.0.0.0/0` for production; use VPN, IP allowlist, or OAuth2 proxy.
- Rate limiting at load balancer / Istio / API gateway.
- Per-user API keys or session auth (see GSoC architecture doc: `gsoc2026_agentic_rag.md`).

#### 2. Application-layer “DoS” (GPU / cost exhaustion)

**Risk:** Not classic network DDoS; attackers flood the **LLM** and **embeddings** paths through Kagent → MCP → TEI/Milvus.

**Mitigations:** Concurrency caps, rate limits, max tokens, request timeouts, quotas on OCI, autoscaling limits with hard max replicas.

#### 3. Milvus default or weak credentials

**Risk:** Milvus is configured with user `root` and password from secret `milvus-auth` / `mcp-server-secret`. If the password is still the default `Milvus`, any workload in an Istio-allowed namespace (`docs-agent`, `kubeflow`) can read, insert, or **drop collections**.

**Mitigations:**

- Strong random password; rotate with secrets and MCP deployment together.
- Kubernetes **NetworkPolicies** limiting sources to MCP and pipeline pods only.
- Backups before ingestion runs; least-privilege Milvus users if supported.

#### 4. Permissive CORS when ingress is enabled

**Risk:** `kagent_cors_allow_origin_regexes = [".*"]` lets arbitrary origins call your agent from users’ browsers.

**Mitigations:** Explicit allowlist regexes; never ship `.*` to production.

### Medium

#### 5. MCP server has no authentication

**Risk:** Any pod in `docs-agent` (or future misconfiguration exposing port 8000) can call MCP directly without Kagent.

**Mitigations:** mTLS or API key between Kagent and MCP; NetworkPolicy on MCP pods; keep Service type `ClusterIP`.

#### 6. Unbounded `top_k` on MCP tools

**Risk:** Large `top_k` values increase Milvus load and response size (availability / memory).

**Mitigations:** Cap `top_k` in `server.py` (recommended max 20); validate in code review.

#### 7. GitHub PAT exposure via pipelines

**Risk:** PATs passed as pipeline **parameters** or logged in KFP/Argo artifacts appear in UI and object storage.

**Mitigations:** Use K8s secret `github-pat` only; leave `github_token` pipeline param empty; rotate PAT if ever logged; use fine-scoped PATs.

#### 8. Dummy LLM API key (`local-kserve`)

**Risk:** `kagent-kserve` secret uses placeholder `OPENAI_API_KEY: local-kserve`. If Qwen’s OpenAI-compatible port is exposed outside the mesh, auth may be trivial.

**Mitigations:** Require real auth on inference even in-cluster; no public Services on Qwen predictor except via controlled ingress.

#### 9. Prompt injection and data exfiltration via tools

**Risk:** Users can steer the agent to query sensitive **indexed** content (issues, code, docs). Not Milvus SQL injection for filters (`repo`, `state`, `resource_kind` are validated); LLM policy bypass is still possible.

**Mitigations:** Tool-call budgets, logging, output policies, human review for production deployments.

### Lower / hygiene

- **Istio ALLOW policies** are namespace-scoped — any compromised pod in `kubeflow` can reach Milvus/embeddings per `istio_policies.tf`.
- **No NetworkPolicies** in repo — defense in depth relies on Istio only.
- **Public agent URL in frontend** (`window.KUBEFLOW_DOCS_AGENT_URL`) is visible in page source — security must be server-side (auth), not obscurity.
- **Vercel static hosting:** attacker uses LB IP directly; protect the agent endpoint, not the JS bundle.

---

## What we do right today

| Control | Location |
|---------|----------|
| Milvus password not in Git | `mcp-server-secret`, `milvus-auth` |
| MCP fails if `MILVUS_PASSWORD` unset | `docs-agent-mcp/mcp-server/server.py` |
| Milvus filter injection guards | `_safe_filter_value` for `repo`, `state`, `resource_kind` |
| MCP not public | `mcp-server.yaml` → `ClusterIP` |
| Explicit Istio ALLOW policies | `docs-agent-mcp/terraform/istio_policies.tf` |
| TEI truncation / batch limits | pipelines + `embeddings.tf` |
| CD secrets in GitHub Environment | `.github/workflows/oke-cicd.yaml` (`kubeflow` env) |
| Optional ingress off by default | `enable_kagent_ingress = false` |

---

## Milvus collections (expected names)

| MCP tool | Collection |
|----------|------------|
| `search_kubeflow_docs` | `docs_rag` |
| `search_github_issues` | `issues_rag` |
| `search_kubeflow_code` | `code_rag` |

Stale names (e.g. `kubeflow_docs_docs_rag`) should be dropped after migration to avoid confusion and wrong MCP config.

---

## Operations checklist

1. **Rotate** `MILVUS_PASSWORD` in `mcp-server-secret` and `milvus-auth` together; restart MCP and verify pipelines.
2. **Restrict** `kubectl` access to `docs-agent`, `ml-infra`, `kubeflow`.
3. **Confirm** MCP Service is `ClusterIP`: `kubectl get svc mcp-kubeflow-docs -n docs-agent`.
4. **Review** public Services: `kubectl get svc -n docs-agent | grep LoadBalancer`.
5. **Before enabling ingress:** DNS, CORS allowlist, auth/rate-limit plan.
6. **After ingestion:** verify collections and entity counts; do not expose Milvus port `19530` via LoadBalancer.
7. **Pipeline runs:** never paste PAT into KFP UI parameters; use secrets only.
8. **Break-test scope:** document allowed targets (your LB hostname only) and forbid shared control-plane namespaces.

---

## Suggested hardening roadmap

| Priority | Item |
|----------|------|
| P0 | Auth + rate limit on public Kagent (or firewall LB) |
| P0 | Rotate Milvus password off defaults |
| P0 | Tight CORS when `enable_kagent_ingress = true` |
| P1 | Cap MCP `top_k` and query length |
| P1 | NetworkPolicies for Milvus and MCP |
| P1 | mTLS or API key Kagent → MCP |
| P2 | OAuth for website + agent (GSoC direction) |
| P2 | OCI WAF / rate limiting on public endpoint |

---

## References

- Ingress Terraform: `docs-agent-mcp/terraform/kagent_ingress.tf`
- Istio policies: `docs-agent-mcp/terraform/istio_policies.tf`
- MCP server: `docs-agent-mcp/mcp-server/server.py`
- Kagent CRDs: `docs-agent-mcp/manifests/kagent/setup.yaml`
- Frontend agent URL config: `frontend/README.md`
- Agent development notes: `AGENTS.md`
