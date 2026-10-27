FROM python:3.12-slim

# libgomp is needed by LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run containers as uid 1000
RUN useradd -m -u 1000 app
WORKDIR /app

COPY --chown=app requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

COPY --chown=app app.py fetch_artifacts.py ./
RUN mkdir -p artifacts && chown app artifacts
USER app

ENV PORT=7860 \
    PYTHONUNBUFFERED=1
EXPOSE 7860
# One worker: each worker holds every model in memory. Artifacts are fetched from
# the Hugging Face Hub on first start if they are not mounted into /app/artifacts.
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180 app:app
