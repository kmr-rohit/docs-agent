# Models and the judge

Two models run in the cluster, each on its own GPU, both from the same
`llm-runtime` Helm chart:

| Release | Model | Served as | GPU | Used by |
|---|---|---|---|---|
| `llm-runtime` | Gemma 4 E4B | `flo-llm` | 1 | The agents (answers) |
| `llm-judge` | Qwen2.5-7B-Instruct (AWQ) | `flo-judge` | 2 | Eval Job, feedback triage, Langfuse evaluators |

## Why a separate judge

An answer model that grades its own answers tends to rate them generously. The
judge is a different model family, so the eval does not grade Gemma with Gemma,
and judging never takes GPU time from user traffic.

Qwen2.5-7B is a small judge. Its scores are good for comparing two commits on
the same questions; they are not an absolute measure of quality. Measuring the
judge's agreement with human labels is listed under
[known gaps](../operations/known-gaps.md).

## How the answer model was chosen

Gemma 4 E4B replaced Qwen2.5-7B as the answer model in #261 after two evals:

| | Qwen2.5-7B | Gemma 4 E4B |
|---|---|---|
| Retrieval hit@5 (22 questions) | 22/22 | 22/22 |
| Faithfulness | 0.71 | 0.69 |
| Correctness | 0.75 | 0.75 |
| Tool use (291 runs, thinking on) | 168 | 270 |
| Mean latency | 17.3 s | 4.4 s |

Answer quality is on par, tool use is much better, and responses are about four
times faster. The details are in [Model tool-use evals](../evaluation/tool-use-evals.md).

## Switching models

The agents always call the served name `flo-llm`, so switching the answer model
is a change to one value, `model`, in the `llm-runtime` release. Qwen is kept in
the chart's catalog as the rollback. A switch restarts the GPU pod, so the
`llm-runtime` and `llm-judge` Argo CD applications are always synced by hand.

Chart values and commands: [`docs-agent-mcp/charts/llm-runtime/README.md`](https://github.com/kubeflow/docs-agent/blob/main/docs-agent-mcp/charts/llm-runtime/README.md). Design decisions: [LLM runtime decisions](../decisions/llm-runtime.md).
