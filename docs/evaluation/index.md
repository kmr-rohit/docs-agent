# Evaluation and feedback

Two loops tell us whether Flo is getting better:

| Loop | When | Question it answers | Results |
|---|---|---|---|
| [Eval on every merge](eval-on-merge.md) | After each merge, when staging syncs the new commit | Did this change break routing or grounding? | Langfuse evaluation project, Experiments |
| [Feedback and triage](feedback-triage.md) | Every user 👍/👎, triaged nightly | Where do real answers fail, and who should fix it? | Langfuse live project, scores and a report |

All judging uses `flo-judge` (Qwen2.5-7B), a different model from the one that
answers, on its own GPU. See [Models](../architecture/models.md).
