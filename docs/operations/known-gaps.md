# Known gaps

What is not done yet, so nobody is surprised and so new contributors can pick
these up.

| Gap | Impact | Plan |
|---|---|---|
| **In-cluster access control is not enforced.** The Istio AuthorizationPolicies for the models, Milvus and embeddings only apply to pods with a sidecar, and `docs-agent` and `ml-infra` have none. | Any pod in the cluster can call the models or read and write Milvus. The public gateway is enforced and unaffected. | Enable sidecar injection one namespace at a time, `ml-infra` first, testing chat after each. |
| **The judge is not calibrated.** Qwen2.5-7B's agreement with human labels has not been measured. | Eval and triage scores are good for comparing commits, not as absolute quality. | Label a small set of answers by hand and report agreement. |
| **Triage guesses the component from keywords.** It does not read the user's comment. | Some content gaps are routed to the wrong working group. | Let the judge pick the component from the question and comment, with keywords as fallback. |
| **Image promotion is a manual PR.** | Someone has to update the image tag for staging and production. | Argo CD Image Updater for staging and a promotion workflow are designed. |
| **`tests/test_mcp_server.py` is skipped.** It targets server internals that #249 replaced. | Less unit coverage of the MCP tools. | Rewrite against `milvus_search` and `intent_router`. Good first issue. |
| **Single replicas.** Milvus is standalone, each model has one GPU pod, and there are no backups. | A node failure takes the service down; data loss needs re-ingestion. | Milvus backups first, then HA where it is worth the cost. |
| **Issues and code are chunked by characters.** Only docs use the token-aware chunker. | Weaker retrieval on issues and code. | Move the issues and code pipelines to the canonical ingest path. |
