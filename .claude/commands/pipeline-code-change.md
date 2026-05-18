---
name: pipeline-code-change
description: Workflow command scaffold for pipeline-code-change in docs-agent.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /pipeline-code-change

Use this workflow when working on **pipeline-code-change** in `docs-agent`.

## Goal

Coordinated updates or fixes to multiple pipeline scripts (incremental, issues, kubeflow, code) to maintain consistency or apply a cross-cutting change.

## Common Files

- `pipelines/incremental-pipeline.py`
- `pipelines/issues-pipeline.py`
- `pipelines/kubeflow-pipeline.py`
- `pipelines/code-pipeline.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Identify the change needed across pipelines (e.g., dependency pinning, Docker image reference, Milvus logic).
- Edit pipelines/incremental-pipeline.py, pipelines/issues-pipeline.py, pipelines/kubeflow-pipeline.py, and sometimes pipelines/code-pipeline.py to apply the change.
- Update documentation if necessary.
- Commit all affected pipeline files together with a descriptive message.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.