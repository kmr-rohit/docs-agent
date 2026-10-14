# Models

Two models run in the cluster, each on its own GPU, both deployed with the
`llm-runtime` Helm chart (KServe with vLLM).

| Release | Model | Served as | Used by |
|---|---|---|---|
| `llm-runtime` | Gemma 4 E4B | `flo-llm` | The agents, to answer questions |
| `llm-judge` | Qwen2.5-7B-Instruct (AWQ) | `flo-judge` | The eval Job, feedback triage, Langfuse evaluators |

## The answer model

Gemma 4 E4B was chosen by evaluation against Qwen2.5-7B:

| | Qwen2.5-7B | Gemma 4 E4B |
|---|---|---|
| Retrieval hit@5 (22 questions) | 22/22 | 22/22 |
| Faithfulness | 0.71 | 0.69 |
| Correctness | 0.75 | 0.75 |
| Tool-use cases passed (of 291) | 168 | 270 |
| Mean latency | 17.3 s | 4.4 s |

Answer quality is level, tool use is much better, and answers are about four
times faster. Two ModelConfigs point at it: `flo-llm` (thinking off) and
`flo-llm-think` (thinking on, used by the docs agent's thinking variant).

## The judge

A model that grades its own answers rates them generously. The judge is a
different model family, so evaluations do not grade Gemma with Gemma, and judging
never takes GPU time from users. Its scores are used to compare commits on the
same questions.

## Switching models

The agents always call the served name `flo-llm`, so switching the answer model
is one value, `model`, in the `llm-runtime` release. Profiles for both models are
in the chart's catalog. A switch restarts the GPU pod and reloads weights, so the
`llm-runtime` and `llm-judge` applications are synced by hand in Argo CD.

Chart values and commands:
[`docs-agent-mcp/charts/llm-runtime/README.md`](https://github.com/kubeflow/docs-agent/blob/main/docs-agent-mcp/charts/llm-runtime/README.md).
