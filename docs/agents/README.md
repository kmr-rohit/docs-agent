# Agent docs

Trusted context for GitHub Agentic Workflows and human maintainers.

| File | Role |
| --- | --- |
| [`architecture.md`](./architecture.md) | In-repo architecture map (from the implementation deep dive) |
| [`triage-labels.md`](./triage-labels.md) | `kind/*` and `area/*` vocabulary, including `area/mcp` |
| [`area-map.json`](./area-map.json) | Machine map: title area → files + labels |
| [`kfp-issue-triage.md`](./kfp-issue-triage.md) | Copy-paste pack for Kubeflow Pipelines (`kubeflow/pipelines`) |
| [`.github/workflows/issue-triage.md`](../../.github/workflows/issue-triage.md) | docs-agent issue triage workflow source |

## How triage uses source and architecture

1. A new issue title is parsed as `<type>(<area>): <summary>`.
2. `scripts/select-triage-context.py` resolves `area` through `area-map.json`.
3. The agent **must** read `architecture.md` plus the selected source files
   before commenting or labeling.
4. Labels are applied only through `safe-outputs` (`kind/*`, `area/mcp`, …).

Create labels before the first run:

```bash
./scripts/sync-github-labels.sh kubeflow/docs-agent
```

After editing the workflow source:

```bash
gh aw compile .github/workflows/issue-triage.md
```

Commit both the `.md` source and the generated `.lock.yml`.
