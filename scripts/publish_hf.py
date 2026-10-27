"""
Publish the trained artifacts and the API to Hugging Face.

    python scripts/publish_hf.py models   # upload api/artifacts/ to the model repo
    python scripts/publish_hf.py space    # deploy api/ as a Docker Space

Requires `pip install huggingface_hub` and `hf auth login` with a write token.
Run `python sync_models_to_api.py` first so api/artifacts/ is complete.
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
API_DIR = ROOT / "api"
ARTIFACTS_DIR = API_DIR / "artifacts"

sys.path.insert(0, str(API_DIR))
from fetch_artifacts import DEFAULT_MODEL_REPO  # noqa: E402

MODEL_REPO = DEFAULT_MODEL_REPO
SPACE_REPO = DEFAULT_MODEL_REPO.split("/")[0] + "/agroyield-api"

# Training splits are not needed for inference. The voting ensemble (~515 MB)
# and K-NN (~100 MB) are left out to keep the hosted API small and fast to boot.
EXCLUDE = ["X_*.pkl", "y_*.pkl", "voting_ensemble.pkl", "k-nn.pkl"]

MODEL_CARD = f"""---
license: mit
tags: [tabular-regression, scikit-learn, agriculture]
---

# AgroYield model artifacts

Trained models, encoders and scalers for
[AgroYield](https://github.com/Abdul-Wahab-17/AgroYield), a county-level US crop
yield predictor. They are loaded by the Flask API in that repo
(`api/fetch_artifacts.py` downloads them automatically).

Pinned library versions are in `api/requirements.txt` of the GitHub repo; the
pickles need scikit-learn 1.7.2 and xgboost 3.0.1.
"""

SPACE_README = f"""---
title: AgroYield API
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

Flask API for [AgroYield](https://github.com/Abdul-Wahab-17/AgroYield).
Models are downloaded from [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}) on start.
Try `GET /health` or `GET /metadata`.
"""


def publish_models(api: HfApi) -> None:
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=MODEL_CARD.encode(), path_in_repo="README.md", repo_id=MODEL_REPO)
    api.upload_folder(
        folder_path=ARTIFACTS_DIR,
        repo_id=MODEL_REPO,
        repo_type="model",
        ignore_patterns=EXCLUDE,
        commit_message="Upload AgroYield artifacts",
    )
    print(f"Models published: https://huggingface.co/{MODEL_REPO}")


def publish_space(api: HfApi) -> None:
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_file(path_or_fileobj=SPACE_README.encode(), path_in_repo="README.md",
                    repo_id=SPACE_REPO, repo_type="space")
    api.upload_folder(
        folder_path=API_DIR,
        repo_id=SPACE_REPO,
        repo_type="space",
        allow_patterns=["Dockerfile", "requirements.txt", "app.py", "fetch_artifacts.py"],
        commit_message="Deploy AgroYield API",
    )
    host = SPACE_REPO.replace("/", "-").replace("_", "-").lower()
    print(f"Space deployed: https://huggingface.co/spaces/{SPACE_REPO}")
    print(f"API URL:        https://{host}.hf.space")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else ""
    if target not in ("models", "space"):
        sys.exit(__doc__)
    api = HfApi()
    publish_models(api) if target == "models" else publish_space(api)
