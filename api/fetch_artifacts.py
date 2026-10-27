"""
Make sure the API's model artifacts are present, downloading them from the
Hugging Face Hub if they are not.

The trained models and preprocessors are too large for git, so they are
published to a Hugging Face model repo. The API calls ensure_artifacts() at
startup; you can also run this file directly:

    python api/fetch_artifacts.py            # download into api/artifacts/

Environment variables:
    AGROYIELD_MODEL_REPO   Hub repo to download from (default below)
    AGROYIELD_ARTIFACTS_DIR  where artifacts live (default: api/artifacts)
"""

import os
from pathlib import Path

DEFAULT_MODEL_REPO = "wahab17/agroyield-models"

# Files the API cannot start without.
REQUIRED_FILES = [
    "crop_encoder.pkl",
    "state_encoder.pkl",
    "county_encoder.pkl",
    "numerical_scaler.pkl",
    "weather_scaler.pkl",
    "soil_scaler.pkl",
    "geo_scaler.pkl",
    "feature_names.pkl",
    "preprocessing_metadata.json",
]


def missing_files(artifacts_dir: Path) -> list[str]:
    return [name for name in REQUIRED_FILES if not (artifacts_dir / name).exists()]


def ensure_artifacts(artifacts_dir: Path) -> None:
    artifacts_dir = Path(artifacts_dir)
    missing = missing_files(artifacts_dir)
    if not missing:
        return

    repo_id = os.environ.get("AGROYIELD_MODEL_REPO", DEFAULT_MODEL_REPO)
    print(f"Artifacts missing ({', '.join(missing)}); downloading from huggingface.co/{repo_id} ...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "Model artifacts are missing and huggingface_hub is not installed.\n"
            "Run `pip install -r api/requirements.txt`, or train locally "
            "(see README) and run `python sync_models_to_api.py`."
        ) from e

    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=artifacts_dir)

    still_missing = missing_files(artifacts_dir)
    if still_missing:
        raise SystemExit(f"Download finished but these files are still missing: {still_missing}")


if __name__ == "__main__":
    target = Path(os.environ.get("AGROYIELD_ARTIFACTS_DIR", Path(__file__).parent / "artifacts"))
    ensure_artifacts(target)
    print(f"Artifacts ready in {target}")
