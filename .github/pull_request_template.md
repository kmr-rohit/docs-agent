## Summary

<!-- What changed and why (1–3 sentences) -->

## Related issues

Fixes #<!-- issue number -->

## Type of change

- [ ] Tests / CI
- [ ] Documentation
- [ ] Pipeline / ingestion
- [ ] MCP server
- [ ] Frontend
- [ ] Infra (Terraform / manifests / Helm)
- [ ] Agentic RAG core (routing, prompts, tool design) — **requires prior issue approval**

## Testing

```bash
pip install -r requirements-test.txt
pytest -v
ruff check docs-agent-mcp/mcp-server tests docs-agent-mcp/pipelines
```

- [ ] Unit tests added or updated
- [ ] Manually tested (describe if cluster/MCP):

## DCO

- [ ] I sign off my commits (`git commit -s`) per [Kubeflow DCO](https://www.kubeflow.org/docs/about/contributing/)
