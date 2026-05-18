---
name: pipeline-codegen-recompile
description: Workflow command scaffold for pipeline-codegen-recompile in docs-agent.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /pipeline-codegen-recompile

Use this workflow when working on **pipeline-codegen-recompile** in `docs-agent`.

## Goal

Regenerate pipeline YAML from updated Python pipeline definition.

## Common Files

- `pipelines/github_rag_pipeline.yaml`
- `pipelines/kubeflow-pipeline.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Update Python pipeline definition (e.g., kubeflow-pipeline.py).
- Regenerate the YAML file (e.g., github_rag_pipeline.yaml) from the updated Python script.
- Commit the regenerated YAML file with a chore: prefix.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.