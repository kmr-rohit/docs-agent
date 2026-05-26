# Qwen on OKE — bring-up notes (validated 2026-05-26)

## Registry choice (GHCR vs OCIR)

**You do NOT need OCIR for CD.** GitHub Actions can build/push to **GHCR** using:

| Secret | Purpose |
|--------|---------|
| `GHCR_USERNAME` | GitHub username or `[org]` |
| `GHCR_TOKEN` | PAT with `write:packages` |

Image example: `ghcr.io/<org>/mcp-kubeflow-docs:<sha>`

Cluster pull: create `imagePullSecret` in `docs-agent` if the package is private.

OCIR is optional if you prefer Oracle-native registry later.

## LLM serving (current validated path)

`kserve/huggingfaceserver:latest-gpu` is **too large** for default OKE GPU node root (~36GB before LVM expand) and exhausts disk/inodes.

**Working setup:** direct vLLM deployment:

```bash
# 1. Expand GPU node disk (required once per new GPU node)
kubectl apply -f manifests/gpu-node-lvm-expand-job.yaml
kubectl wait --for=condition=complete job/gpu-node-lvm-expand -n kube-system --timeout=120s

# 2. Deploy Qwen vLLM
kubectl apply -f manifests/qwen-vllm-deployment.yaml

# 3. Validate
kubectl exec -n docs-agent deploy/qwen-vllm -- python3 -c "
import urllib.request, json
p=json.dumps({'model':'qwen2.5-3b-instruct','messages':[{'role':'user','content':'Hello'}],'max_tokens':32}).encode()
r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=p,headers={'Content-Type':'application/json'}))
print(r.read().decode())
"
```

OpenAI-compatible endpoint (in-cluster):

`http://qwen-vllm.docs-agent.svc.cluster.local:8000/v1`

## HF token

Create secret when `hf_token` is available:

```bash
kubectl create secret generic huggingface-secret -n docs-agent \
  --from-literal=token="$HF_TOKEN" --dry-run=client -o yaml | kubectl apply -f -
```

Qwen2.5-3B-Instruct works without token; token helps HF rate limits.

## Kagent cutover (next step)

Point `ModelConfig` at vLLM service instead of Groq:

```yaml
openAI:
  baseUrl: "http://qwen-vllm.docs-agent.svc.cluster.local:8000/v1"
model: qwen2.5-3b-instruct
```

## KServe InferenceService

Left stopped (`serving.kserve.io/stop=true`). Revisit after GPU nodes have expanded disk and a slimmer runtime image is chosen.
