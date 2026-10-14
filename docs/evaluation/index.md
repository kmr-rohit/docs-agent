# Evaluation and feedback

Three loops tell us whether Flo is getting better:

| Loop | When | Question it answers | Results |
|---|---|---|---|
| [Eval on every merge](eval-on-merge.md) | After each staging sync | Did this change break routing or grounding? | Langfuse evaluation project, Experiments |
| [Feedback and triage](feedback-triage.md) | Every user 👍/👎, triaged nightly | Where do real answers fail, and who should fix it? | Langfuse live project, scores and a report |
| [Model tool-use evals](tool-use-evals.md) | When choosing or tuning a model | Does a model call the right tool with the right arguments? | `tests/eval/SLM_EVALS.md` |

All judging uses `flo-judge` (Qwen2.5-7B), a different model from the one that
answers, on its own GPU. See [Models and the judge](../architecture/models.md).
