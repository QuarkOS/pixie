# How the pieces connect

This is a generic map of the sanitized tree. It is not a live cutover runbook and has no hostnames.

1. **Web app** (`web/server.mjs` and `web/api/`) serves the site, A/B, chat queue, and admin. Data lives in `DATA_DIR`. Secrets come from env.
2. **Chat worker** (`worker/worker_v3.py`) polls the web app's worker API with `CHAT_WORKER_KEY`, loads a v3/s7 checkpoint, samples, and posts results.
3. **Training** (`train/train_v3.py`) writes checkpoints into a new run folder. Copy or point the worker config at a finished checkpoint when you want the worker to use it. Never overwrite an existing run folder.
4. **Data** flows Krea 2 sheets → `extract_v2` → 32×32 sprites → (separately) the 16×16 v3 training set. The 16×16 trainer and the 32×32 Krea pipeline are related but not the same model.

Do not deploy from this document. Ask the operator to apply any production change.
