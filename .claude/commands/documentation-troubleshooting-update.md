---
name: documentation-troubleshooting-update
description: Workflow command scaffold for documentation-troubleshooting-update in docs-agent.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /documentation-troubleshooting-update

Use this workflow when working on **documentation-troubleshooting-update** in `docs-agent`.

## Goal

Incremental updates to kube.md to document new troubleshooting tips, workarounds, or operational notes.

## Common Files

- `kube.md`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Discover or resolve a new operational issue or workaround.
- Edit kube.md to add a new section or update existing content.
- Commit only kube.md with a docs: prefix in the message.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.