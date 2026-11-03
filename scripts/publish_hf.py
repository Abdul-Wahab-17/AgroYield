"""
Publish the trained artifacts and the API to Hugging Face.

    python scripts/publish_hf.py models   # upload api/artifacts/ to the model repo
    python scripts/publish_hf.py space    # deploy the API as a Docker Space
    python scripts/publish_hf.py stage D  # assemble the deployable folder in D (for a local docker build)

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

# api/artifacts/ is built by `python -m agroyield.pipeline export`, which already leaves out models
# that are too large to host, so nothing else needs excluding.
EXCLUDE = []

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


def stage(dest: Path) -> Path:
    """Assemble the deployable API folder: app, downloader, pins, Dockerfile and the shared feature code."""
    import shutil
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "agroyield").mkdir(parents=True)
    for name in ("Dockerfile", "requirements.txt", "app.py", "fetch_artifacts.py"):
        shutil.copy(API_DIR / name, dest / name)
    for module in ("__init__.py", "features.py", "model.py"):
        shutil.copy(ROOT / "agroyield" / module, dest / "agroyield" / module)
    return dest


def publish_space(api: HfApi) -> None:
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_file(path_or_fileobj=SPACE_README.encode(), path_in_repo="README.md",
                    repo_id=SPACE_REPO, repo_type="space")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        staged = stage(Path(tmp) / "api")
        api.upload_folder(folder_path=staged, repo_id=SPACE_REPO, repo_type="space",
                          commit_message="Deploy AgroYield API")
    host = SPACE_REPO.replace("/", "-").replace("_", "-").lower()
    print(f"Space deployed: https://huggingface.co/spaces/{SPACE_REPO}")
    print(f"API URL:        https://{host}.hf.space")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else ""
    if target == "stage":
        print(f"staged at {stage(Path(sys.argv[2] if len(sys.argv) > 2 else 'build/api'))}")
        sys.exit(0)
    if target not in ("models", "space"):
        sys.exit(__doc__)
    api = HfApi()
    publish_models(api) if target == "models" else publish_space(api)
