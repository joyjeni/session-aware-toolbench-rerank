# SATR HTTP microservice

> ACRS Objective 1. Proposal-stage: this service does not claim NDCG. Live
> Indian OGD execution belongs to MNCD; SATR only ranks.

```bash
python service/app.py 43131
curl -X POST http://127.0.0.1:43131/rerank -H 'Content-Type: application/json' \
  -d '{"query":"ragi pest Tumakuru","history_uids":[],"top_k":5}'
```

Uses the Karnataka catalog plus `CoactivationCache`. Gemma embeddings are
optional (HF Space); this service is CPU-only.
