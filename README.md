# AgroYield

County-level **crop yield prediction for the United States**. It joins 2.2 million USDA county yield observations (23 crops, 124 yield series, 1909–2025) with growing-season weather, drought indices and soil, trains 10 models, and scores them on years they never saw against simple baselines. A Flask API serves the models and a React dashboard explores them.

[![CI](https://github.com/Abdul-Wahab-17/AgroYield/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdul-Wahab-17/AgroYield/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![Predict page](docs/screenshots/predict.png)

Every number below comes from running this repository's scripts; nothing is hand-entered.

## What it does

- **Predict** one crop in one county for a year. Choose the *variety* too: NASS publishes separate yields for irrigated vs non-irrigated, winter vs spring wheat, grain vs silage, and so on, and each is its own series. Weather and soil default to the county's own 1991–2020 normals and can be overridden.
- **Regional view:** every crop a county grows, side by side. Crops with no yield records there are listed as absent, never extrapolated.
- **Evidence page:** each model's error on held-out years, against baselines, per crop.

| Regional | Evidence |
|---|---|
| ![Regional](docs/screenshots/regional.png) | ![Metrics](docs/screenshots/metrics.png) |

## Results

Test set: **12,739 county-year observations from 2023–2025**. No model saw these years in training or tuning (see [Method](#method)). Errors are percentages of the true yield, so bushels, pounds and tons compare fairly.

| Model | Mean error | Median error | Within 10 % | Error removed vs. best baseline |
|---|---:|---:|---:|---:|
| **Blend** (XGBoost + LightGBM + CatBoost) | **12.7 %** | 8.8 % | 55 % | 28 % |
| XGBoost *(default in the app)* | 12.9 % | 9.1 % | 54 % | 27 % |
| LightGBM | 13.0 % | 9.0 % | 54 % | 27 % |
| CatBoost | 13.0 % | 9.0 % | 54 % | 27 % |
| Hist Gradient Boosting | 13.3 % | 9.3 % | 53 % | 25 % |
| Random Forest | 13.9 % | 9.4 % | 53 % | 21 % |
| Decision tree | 15.7 % | 10.6 % | 48 % | 12 % |
| *Baseline: last observed yield in that county* | 17.7 % | 11.7 % | 44 % | – |
| SVR (25 k-row subsample) | 19.7 % | 12.4 % | 42 % | −11 % |
| Linear / Ridge regression | 22.3 % / 22.5 % | 16.6 % | 30 % | −26 % |
| *Baseline: linear trend per series* | 27.1 % | 17.5 % | 30 % | – |
| K-NN (60 k-row subsample) | 38.6 % | 25.1 % | 21 % | −117 % |
| *Baseline: county's historical mean* | 38.6 % | 41.2 % | 7 % | – |

**How to read this honestly**

- The strongest baseline is simply "same as the last observed yield". The gradient-boosted models remove about a quarter of its error, and the blend 28 %. That is a real gain, not a dramatic one: a county's own recent yield already explains most of next year's.
- **The models lose on some crops.** For rice (146 test rows, 10.7 % vs 6.1 % for the baseline) and peanuts (277 rows, 11.4 % vs 11.0 %) persistence is as good or better, because those yields barely move from year to year. The per-crop table on the Metrics page shows every crop, including these.
- The test years are thin. NASS stopped publishing most county estimates after about 2018 (about 5,000 county-years a year now, against about 21,000 in the 2000s and 50,000 in the 1980s), so 2023–2025 holds 12.7 k rows and rare crops have very few.
- A prediction is "expected yield under normal weather" unless you enter the year's weather. The models do respond to weather inputs (the tests check that a drought lowers corn), but the app has no live weather feed.

## Data

Every source is downloaded by a script in `agroyield/` with retries and resume, and recorded with its hash in `data/raw/MANIFEST.json`.

| Source | What | Script |
|---|---|---|
| [USDA NASS Quick Stats](https://quickstats.nass.usda.gov/) | County yields and harvested area, 1909–2025. Bulk snapshot `qs.crops_20261031` (1.1 GB, no API key). 2,215,608 observations after dropping suppressed `(D)` values and the "other counties" bucket. | `download_nass.py` |
| [NASA POWER](https://power.larc.nasa.gov/) | Daily max/min temperature, rainfall and humidity, 1981–2025, turned into April–October features: means, rainfall, growing degree days, days above 35 °C, frost days, June–August breakdowns. The 3,142 counties share 1,905 native 0.5° × 0.625° grid cells, each fetched once. | `download_weather.py` |
| [NOAA nClimDiv](https://www.ncei.noaa.gov/products/land-based-station/nclimdiv) | County monthly temperature and rainfall plus the **Palmer Drought Severity Index** and Z-index, 1895–2026. | `download_climdiv.py` |
| [USDA NRCS SSURGO](https://sdmdataaccess.nrcs.usda.gov/) | 0–30 cm clay, sand, silt, organic matter, pH, bulk density, CEC and water capacity, weighted by map-unit area, component share and layer thickness. 3,124 of 3,142 counties; the rest stay missing rather than being filled in. | `download_soil.py` |
| [Census Gazetteer 2020](https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html) | County FIPS codes, names, centroids. | `download_counties.py` |

`merge.py` joins everything on the 5-digit county FIPS code. (Joining on names loses counties such as "St. Clair" vs "ST CLAIR".) Of 2.2 M NASS rows, 1,455 had no county match. A **series** is commodity × class × production practice × utilization × unit; the 124 series with at least 1,000 county-years are modelled, and the series is a feature.

**Features (61).** Season weather, NOAA climate and drought indices, soil, location, and the county's own recent yields for that series: the last yield, the mean of the last five, the all-time mean, and the gap in years. History only ever uses *earlier* years, so it is known when a season starts and cannot leak the target. `agroyield/features.py` is imported by both training and the API, so serving computes exactly what training did (`tests/test_features.py` checks this).

## Method

1. **Temporal split.** Train ≤ 2020 (1,065,377 rows), validate 2021–2022, test 2023–2025.
2. Each model is fitted on the training years. Boosting rounds are chosen on the validation years, and the blend weights are learned on validation predictions.
3. Every model is then **refit on train + validation** and evaluated **once** on the test years.
4. The target is `log(yield)`: series differ in scale by up to 1000×, and errors are proportional.
5. K-NN and SVR train on random subsamples (60 k and 25 k rows) because they do not scale to a million rows; Random Forest is size-capped.

## Run it

### Use the trained models

The models (about 124 MB) are on the Hugging Face Hub and are downloaded on first start.

```bash
pip install -r api/requirements.txt
python api/app.py                      # http://localhost:5000

cd client && npm install && npm start  # http://localhost:3000
```

Or with Docker:

```bash
python scripts/publish_hf.py stage build/api     # assembles app.py + the shared feature code
docker build -t agroyield-api build/api
docker run -p 5000:7860 agroyield-api
```

To point the frontend at a different API, set `REACT_APP_API_URL`.

### Rebuild everything from the raw sources

```bash
pip install -r requirements.txt
python -m agroyield.pipeline all        # download -> merge -> train -> export
```

Each stage skips work already on disk, so an interrupted run resumes; run one stage with `python -m agroyield.pipeline download|merge|train|export`. Allow roughly **1–3 hours to download** (mostly NASA POWER and the 1.1 GB NASS file) and **2–4 hours to train** on 8 cores, with about 10 GB of free memory and 10 GB of disk.

```bash
pytest tests --ignore=tests/test_api.py   # offline: features, metrics, downloader parsing
python api/fetch_artifacts.py && pytest tests/test_api.py
```

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Loaded models, number of series |
| GET | `/metadata` | Crops with their varieties and states; models |
| GET | `/counties/<state>?series=` | Counties with yield records |
| POST | `/predict` | `{commodity or series, state, county, year, model?, <overrides>}` |
| POST | `/predict-regional` | `{state, county, year}`: every crop the county grows |
| GET | `/metrics`, `/metrics/commodities` | Test-set results; per-crop results |
| GET | `/feature-importance/<model>` | Top features of tree models |

```bash
curl -X POST localhost:5000/predict -H 'Content-Type: application/json' \
  -d '{"commodity": "CORN", "state": "IA", "county": "Polk", "year": 2026}'
```

Overrides: `tmax_season`, `tmin_season`, `prcp_season`, `gdd_season`, `heat_days`, `prcp_jun/jul/aug`, `cd_pdsi_*`, `clay_pct`, `sand_pct`, `organic_matter_pct`, `ph`, and more.

The API serves the 10 individual models. The blend is used for evaluation only, since it bundles its members.

## Project structure

```text
agroyield/    pipeline: download_*.py, merge.py, features.py, train.py, export.py, pipeline.py
api/          Flask API, Dockerfile, artifact downloader
client/       React dashboard
scripts/      publish to Hugging Face (models, Space, staging)
tests/        features, metrics, downloader parsing, API
```

## License

[MIT](LICENSE) © 2026 Abdul Wahab
