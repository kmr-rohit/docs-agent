# Decisions

Short records of choices that shape the project, so later contributors know why
things are the way they are. Each entry: the decision, the reason, and what we
gave up.

| # | Decision | Why | Trade-off |
|---|---|---|---|
| 1 | **Argo CD is the only deployer**; CI never touches the cluster | Two writers (CI `helm upgrade` and Argo CD) overwrote each other's settings; GitOps makes drift visible | Releases need a sync step |
| 2 | **Staging and production are namespaces, not branches** | No merge-backs or drifting branches; contributors keep targeting `main` | Both share one cluster's GPUs and Milvus |
| 3 | **A separate judge model on its own GPU** (Qwen judges Gemma) | Avoids a model grading itself; judging never slows user traffic | A second GPU, and a small judge that needs calibration |
| 4 | **Gemma 4 E4B as the answer model** | Same answer quality as Qwen2.5-7B, much better tool use, about 4× faster | Small-talk behaviour needed the thinking variant; see [tool-use evals](../evaluation/tool-use-evals.md) |
| 5 | **Environment config in the public repo, secrets only in the cluster** | Reviewable, no private config repo to maintain | Hostnames and image names are public |
| 6 | **Route traces by namespace in the collector** | kagent's runtime ignores `OTEL_RESOURCE_ATTRIBUTES` | The collector needs read-only access to pods |
| 7 | **Feedback and triage write scores; humans file issues** | Avoids noisy or wrong auto-filed issues while the judge is uncalibrated | Someone has to read the report |
| 8 | **Small services run from ConfigMaps on stock images** (feedback, eval, triage) | No image to build or host for a few hundred lines of Python | Code is fetched or mounted at runtime |

Longer write-ups:

- [LLM runtime decisions](llm-runtime.md)
