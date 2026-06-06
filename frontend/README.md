# Kubeflow website chatbot (Kagent A2A)

Static assets for the docs-site chat widget. The bot talks to **Kagent** over JSON-RPC `message/stream` (not the legacy REST API).

## Configure the agent URL

Set one of these before the widget loads:

```html
<script>
  window.KUBEFLOW_DOCS_AGENT_URL = 'https://YOUR_HOST/a2a/docs-agent/kubeflow-docs-agent';
</script>
<script src="/docs_scripts/chatbot.js" defer></script>
```

Or on the script tag:

```html
<script
  src="/docs_scripts/chatbot.js"
  data-agent-url="https://YOUR_HOST/a2a/docs-agent/kubeflow-docs-agent"
  defer
></script>
```

Or base URL only (path is appended automatically):

```html
<script
  src="/docs_scripts/chatbot.js"
  data-agent-base="http://YOUR_LOAD_BALANCER_IP"
  defer
></script>
```

## Vercel demo page

Host `docs_scripts/` and `docs_styles/` on Vercel and set `KUBEFLOW_DOCS_AGENT_URL` to your cluster endpoint:

- **LoadBalancer (current OKE setup):** `kagent-ui-lb` external IP + `/a2a/docs-agent/kubeflow-docs-agent` (use HTTPS ingress if the Vercel site is `https://` to avoid mixed-content blocking).
- **Istio ingress (optional Terraform):** set `enable_kagent_ingress = true` and `kagent_domain_name` in `docs-agent-mcp/terraform/`, then point DNS and use `https://your-domain/...`.

CORS: browser calls require the agent host to allow your Vercel origin. Default Istio ingress CORS uses configurable regexes (`kagent_cors_allow_origin_regexes`); the LB path depends on Kagent’s own CORS settings.

## Feedback API (golden dataset)

Each bot reply shows a **1–5 rating scale** plus optional comment. Ratings are stored by the feedback service for golden-dataset curation.

Configure the feedback endpoint (expose `docs-feedback` via ingress or LoadBalancer in cluster):

```html
<script>
  window.KUBEFLOW_DOCS_FEEDBACK_URL = 'https://YOUR_HOST/api/feedback';
</script>
```

Or:

```html
<script src="/docs_scripts/chatbot.js" data-feedback-url="https://YOUR_HOST/api/feedback" defer></script>
```

In-cluster default service: `http://docs-feedback.docs-agent.svc.cluster.local:8080/api/feedback` (ClusterIP — not reachable from browsers until exposed).

## Citations

Retrieved sources are shown under each answer. The MCP tools emit `**Source:**` URLs and a machine-readable `<!--KUBEFLOW_CITATIONS:[...]-->` block; the widget collects these from the Kagent stream and renders a collapsible **Sources** list.

## Milvus collections (MCP tools)

| Tool | Collection |
|------|------------|
| Docs | `docs_rag` |
| Issues | `issues_rag` |
| Code | `code_rag` |
