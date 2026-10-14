# Eval on every merge

After Argo CD syncs staging, a Job sends 58 real questions through the whole
deployed stack (agent, MCP tools, Milvus, model), grades the answers, and
records the run in Langfuse. If a gate fails, the Job fails and Argo CD marks the
sync as failed.

## What it checks

The cases are the single-turn questions in `tests/eval/flo_eval_dataset.json`:
docs routing, issues routing, debug routing, query fidelity, and small talk that
should not trigger a search.

| Score | Meaning | Gate |
|---|---|---|
| `routing` | The agent called an expected tool, or no tool for small talk | ≥ 0.85 |
| `query_fidelity` | The search query kept the key terms of the question | reported |
| `grounded` | The judge finds every claim supported by what the tools returned (`yes` / `partial` / `no`) | share of `yes` ≥ 0.65 |
| `answers` | The answer addresses the question | reported |
| errors | Any failed request | 0 |

Latest baseline (staging, October 2026): routing 1.00, query fidelity 1.00,
grounded 0.73, answers 0.78, median latency 2.6 s.

## Finding your run

1. Log in to Langfuse with the community account and open the **evaluation**
   project.
2. Go to **Datasets → `flo-e2e` → Experiments**.
3. Runs are named `<commit> pr-<number> @<author> <time>`. Find yours by PR
   number or GitHub username.
4. Open the run to see each question, the agent's answer, the expected tools and
   the four scores. Select two runs to compare them side by side.
5. Each item links to the agent's full trace (`agent_trace_id` in its metadata).

## How it works

- Chart: `docs-agent-mcp/charts/docs-agent/templates/eval-job.yaml`, enabled with
  `eval.enabled` (on in staging only). It is an Argo CD `PostSync` hook.
- The Job downloads `tests/eval/staging_eval.py` and the dataset at the synced
  commit, so the eval always matches the deployed code. Standard library only.
- It talks to the agents through the kagent controller's A2A API, and uses a
  unique A2A `contextId` per question, which becomes the Langfuse session id.
- Results are written as Langfuse v4 Experiments (spans with
  `langfuse.experiment.*` attributes) plus scores on each agent trace.

## Running it yourself

```bash
# against staging, from a laptop with port-forwards to the kagent controller and the judge
AGENT_BASE=http://localhost:8083 \
JUDGE_URL=http://localhost:8091/v1/chat/completions \
LIMIT=5 python tests/eval/staging_eval.py
```

Without Langfuse keys the results are printed but not recorded. `LIMIT` runs only
the first N cases; the chart value `eval.limit` does the same for a smoke run.
