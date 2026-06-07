# Feedback service (golden dataset)

Stores 1–5 ratings, optional comments, query/response text, and citation URLs from the docs chatbot.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness |
| `POST` | `/api/feedback` | Submit feedback JSON |
| `GET` | `/api/feedback/export` | List rows (`?min_rating=4&limit=100`) |

## Deploy (cluster)

```bash
# Build & push (from repo root)
docker build -t ghcr.io/OWNER/docs-feedback:latest \
  -f docs-agent-mcp/feedback-service/Dockerfile docs-agent-mcp/
docker push ghcr.io/OWNER/docs-feedback:latest

kubectl apply -f docs-agent-mcp/manifests/feedback-service/feedback-service.yaml
kubectl set image deployment/docs-feedback \
  feedback-api=ghcr.io/OWNER/docs-feedback:latest -n docs-agent
kubectl rollout status deployment/docs-feedback -n docs-agent
```

SQLite database path: `/data/feedback.db` on PVC `feedback-db`.

## View stored feedback

### 1. Export API (recommended)

```bash
kubectl port-forward -n docs-agent svc/docs-feedback 18081:8080
curl -s 'http://127.0.0.1:18081/api/feedback/export?limit=50' | jq .
curl -s 'http://127.0.0.1:18081/api/feedback/export?min_rating=4' | jq '.items[] | {rating, query, comment, created_at}'
```

### 2. SQLite inside the pod

```bash
kubectl exec -n docs-agent deploy/docs-feedback -- \
  python3 -c "
import sqlite3, json
conn = sqlite3.connect('/data/feedback.db')
for row in conn.execute('SELECT created_at, rating, query, comment, citations FROM feedback ORDER BY created_at DESC LIMIT 20'):
    print(row[0], 'rating=', row[1])
    print('  Q:', row[2][:80])
    if row[3]: print('  comment:', row[3])
    print('  citations:', json.loads(row[4] or '[]'))
"
```

### 3. Golden-dataset export to file

```bash
curl -s 'http://127.0.0.1:18081/api/feedback/export?min_rating=3' > golden-feedback.json
```

Use `min_rating=4` or `5` for high-quality examples; lower ratings flag responses to fix.

## Browser access

Expose via ingress/LB or port-forward, then set on the website:

```html
window.KUBEFLOW_DOCS_FEEDBACK_URL = 'https://YOUR_HOST/api/feedback';
```

Tighten `ALLOWED_ORIGINS` in the feedback ConfigMap for production.
