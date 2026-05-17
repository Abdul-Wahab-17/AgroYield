# AgroYield Intelligence

AgroYield Intelligence is an end-to-end machine learning system for crop yield prediction across U.S. states and counties. It combines data preprocessing, model training, model serving through a Flask API, and an interactive React client for exploration and inference.

## Overview

The project supports:
- Multi-model training and evaluation for crop yield regression
- API-based single and regional prediction workflows
- Model selection and performance inspection endpoints
- Frontend dashboards for predictions and metrics

## Repository Structure

```text
.
├── api/                          # Flask prediction API
│   └── app.py
├── client/                       # React frontend
│   ├── public/
│   └── src/
├── models/
│   ├── error_analysis/           # Per-model error analysis outputs
│   └── feature_importance/       # Feature importance CSV outputs
├── src/
│   ├── download_data/            # Data acquisition scripts
│   ├── model/                    # Training and testing scripts
│   ├── merge_master_data.py
│   └── preprocess_data.py
├── requirements.txt
└── sync_models_to_api.py         # Copies model artifacts for API serving
```

## Technology Stack

- Python (data processing, modeling, API)
- Flask + Flask-CORS (serving inference endpoints)
- scikit-learn and gradient boosting libraries (modeling)
- React (frontend interface)

## Setup

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm

### Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### Install Frontend Dependencies

```bash
cd client
npm install
```

## Running the System

### 1. Preprocess Data

```bash
python src/preprocess_data.py
```

### 2. Train Models

```bash
python src/model/train_models.py
```

### 3. Sync Model Artifacts to API

```bash
python sync_models_to_api.py
```

### 4. Start the API

```bash
cd api
python app.py
```

Default API URL: `http://localhost:5000`

### 5. Start the Frontend

```bash
cd client
npm start
```

Default frontend URL: `http://localhost:3000`

## API Endpoints

Core endpoints exposed by the Flask app:
- `GET /health`
- `GET /metadata`
- `GET /counties/<state>`
- `GET /valid-states/<crop>`
- `GET /models`
- `POST /models/<model_name>`
- `POST /predict`
- `POST /predict-regional`
- `GET /metrics`
- `GET /feature-engineering`
- `GET /feature-importance`
- `GET /feature-importance/<model>`

## Example Requests

### Single Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Crop": "CORN",
    "State": "IOWA",
    "County": "POLK",
    "Year": 2024
  }'
```

### Regional Prediction

```bash
curl -X POST http://localhost:5000/predict-regional \
  -H "Content-Type: application/json" \
  -d '{
    "State": "IOWA",
    "County": "POLK",
    "Year": 2024
  }'
```

## Development Notes

Key scripts:
- `src/preprocess_data.py`: preprocessing pipeline
- `src/model/train_models.py`: model training pipeline
- `src/model/testing.py`: model testing utilities
- `sync_models_to_api.py`: artifact synchronization for API loading
- `api/app.py`: API entry point

## Troubleshooting

- If models fail to load in the API, run `python sync_models_to_api.py` and verify expected artifacts are available to the API.
- If the frontend cannot reach the API, ensure the API is running on port `5000` and review CORS/API URL configuration.
- If training fails, verify all Python dependencies installed from `requirements.txt` are available in the active environment.

## License

Add your project license information here (for example, MIT, Apache-2.0, or a custom academic/research license).
