#!/usr/bin/env bash
# Entry point for the Hugging Face Space.
set -euo pipefail

# 1. Fetch the trained model and sample images from the Hugging Face model repo
#    named in the MODEL_REPO Space variable.
python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo = os.environ.get("MODEL_REPO", "").strip()
if not repo:
    print("MODEL_REPO is not set: starting without a model.")
else:
    snapshot_download(repo_id=repo, local_dir="/models",
                      allow_patterns=["patchcore/*", "samples/*"])
    print(f"Downloaded model and samples from {repo}")
PY

# 2. Model API, reachable only inside the container
uvicorn app.api:app --host 127.0.0.1 --port 8000 &

# 3. Dashboard on the port Spaces expose. XSRF protection is off because Spaces
#    embed the app in an iframe, which otherwise breaks file uploads.
exec streamlit run dashboard/streamlit_app.py \
  --server.port 7860 --server.address 0.0.0.0 --server.headless true \
  --server.enableCORS false --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
