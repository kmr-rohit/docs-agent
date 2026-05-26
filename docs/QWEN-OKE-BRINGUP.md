# Qwen on OKE via KServe — bring-up notes

## Prerequisites

- OKE GPU node pool (`VM.GPU.A10.1`, `nvidia.com/gpu` taint)
- KServe + Knative installed
- **Expand GPU node root disk** after each new GPU node (250GB boot volume ships with ~36GB root FS until expanded)

## Quick deploy

```bash
./scripts/deploy-qwen-kserve.sh
```

Or manually:

```bash
kubectl apply -f manifests/gpu-node-lvm-expand-job.yaml
kubectl wait --for=condition=complete job/gpu-node-lvm-expand -n kube-system --timeout=180s

kubectl apply -f manifests/serving-runtime.yaml
kubectl apply -f manifests/inference-service.yaml

kubectl wait --for=condition=Ready inferenceservice/qwen -n docs-agent --timeout=1800s
```

## Important image pin (CUDA)

**Do not use `latest-gpu`** on current OKE nodes (driver CUDA 13.1).  
`latest-gpu` requires CUDA **>= 13.2** and fails with:

```
nvidia-container-cli: unsatisfied condition: cuda>=13.2
```

Use pinned tag in `manifests/serving-runtime.yaml`:

```yaml
image: index.docker.io/kserve/huggingfaceserver:v0.17.0-gpu
```

## HF token (optional)

```bash
kubectl create secret generic huggingface-secret -n docs-agent \
  --from-literal=token="$HF_TOKEN" --dry-run=client -o yaml | kubectl apply -f -
```

## Endpoints

| Use | URL |
|-----|-----|
| OpenAI chat (in-cluster) | `http://qwen-predictor.docs-agent.svc.cluster.local/openai/v1` |
| Kagent ModelConfig baseUrl | same as above |

Model name: `qwen2.5-3b-instruct`

## Smoke test

```bash
kubectl exec -n docs-agent deploy/kagent-tools -- python3 -c "
import urllib.request, json
p=json.dumps({'model':'qwen2.5-3b-instruct','messages':[{'role':'user','content':'Hello'}],'max_tokens':32}).encode()
r=urllib.request.urlopen(urllib.request.Request('http://qwen-predictor.docs-agent.svc.cluster.local/openai/v1/chat/completions',data=p,headers={'Content-Type':'application/json'}))
print(r.read().decode())
"
```

## Stop / start (cost saving)

```bash
# stop predictor pods
kubectl annotate inferenceservice qwen -n docs-agent serving.kserve.io/stop=true --overwrite

# start again
kubectl annotate inferenceservice qwen -n docs-agent serving.kserve.io/stop-
```

## Registry for CD (MCP images)

OCIR is **not required**. Use **GHCR** with `GHCR_USERNAME` + `GHCR_TOKEN` in GitHub Actions.
