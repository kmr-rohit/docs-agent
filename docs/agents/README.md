# Agent docs — issue triage (first merge)

This folder is the **instruction pack + label set** for GitHub Agentic
Workflows. Merge this first. Then compile a workflow and open a test PR.

| File | Who uses it |
| --- | --- |
| [`issue-triage.md`](./issue-triage.md) | **docs-agent** AW source — copy to `.github/workflows/issue-triage.md` |
| [`kfp-issue-triage.md`](./kfp-issue-triage.md) | **Kubeflow Pipelines** AW source for Shristi's test PR |
| [`triage-labels.md`](./triage-labels.md) | docs-agent `kind/*` and `area/*` labels (`area/mcp`, …) |
| [`architecture.md`](./architecture.md) | Trusted layer/file map the agent must read |
| [`area-map.json`](./area-map.json) | Same map as JSON |

## Create labels (docs-agent)

GitHub will not apply a label that does not exist. After merge:

```bash
./scripts/sync-github-labels.sh kubeflow/docs-agent
```

Titles: `<type>(<area>): <summary>` — example `bug(mcp): search_kubeflow_docs returns Search failed`.

## Shristi: run initial AW testing with a PR

### docs-agent

```bash
gh extension install github/gh-aw   # once
cp docs/agents/issue-triage.md .github/workflows/issue-triage.md
gh aw compile .github/workflows/issue-triage.md
git add .github/workflows/issue-triage.md .github/workflows/issue-triage.lock.yml
git commit -s -m "chore: compile docs-agent issue triage AW"
# open a PR with those two files
```

After that PR is on a repo with issues enabled, open a test issue:

- `bug(mcp): search_kubeflow_docs returns Search failed on empty collection`

Expect `kind/bug` + `area/mcp` and one triage comment.

### Kubeflow Pipelines

```bash
# in a kubeflow/pipelines checkout
cp /path/to/docs-agent/docs/agents/kfp-issue-triage.md .github/workflows/issue-triage.md
gh aw compile .github/workflows/issue-triage.md
# open a PR with the .md + .lock.yml
```

Test issue titles: `bug(backend): …`, `bug(frontend): …`, `feat(sdk): …`.

Pipelines already has `area/backend` / `area/frontend` / `area/sdk`. Do not
create docs-agent labels (`area/mcp`) there.
