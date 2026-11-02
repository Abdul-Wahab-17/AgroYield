"""AgroYield API: county-level crop yield predictions.

Models and data come from the artifacts directory (see fetch_artifacts.py). Every prediction builds its
feature row with the same `agroyield.features` code the models were trained with.
"""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # Docker: agroyield/ sits next to this file
sys.path.insert(0, str(HERE.parent))   # repo checkout: agroyield/ is one level up

from agroyield.features import CATEGORICAL, FEATURES, SOIL, WEATHER, CLIMDIV, add_features, history_from_summary  # noqa: E402
from fetch_artifacts import ensure_artifacts  # noqa: E402

ARTIFACTS = Path(os.environ.get("AGROYIELD_ARTIFACTS_DIR", HERE / "artifacts"))
ensure_artifacts(ARTIFACTS)

app = Flask(__name__)
CORS(app)

MANIFEST = json.loads((ARTIFACTS / "manifest.json").read_text())
CATALOGUE = {s["series"]: s for s in json.loads((ARTIFACTS / "series_catalogue.json").read_text())}
DEFAULTS = json.loads((ARTIFACTS / "county_defaults.json").read_text())
COUNTIES = pd.read_csv(ARTIFACTS / "counties.csv", dtype={"fips": str})
COMPARISON = pd.read_csv(ARTIFACTS / "model_comparison.csv")
PER_COMMODITY = pd.read_csv(ARTIFACTS / "per_commodity.csv").set_index("commodity")
SUMMARY = json.loads((ARTIFACTS / "summary.json").read_text())

MODELS = {m["name"]: joblib.load(ARTIFACTS / "models" / m["file"]) for m in MANIFEST["models"]}
DEFAULT_MODEL = MANIFEST["default_model"]

_history = pd.read_parquet(ARTIFACTS / "history.parquet")
HISTORY = {(r.series, r.fips): (list(r.years), list(r.yields), int(r.n_all), float(r.sum_all))
           for r in _history.itertuples()}
del _history

FIPS_BY_NAME = {(r.state, r.county.upper()): r.fips for r in COUNTIES.itertuples()}
NAME_BY_FIPS = {r.fips: (r.state, r.county) for r in COUNTIES.itertuples()}

# series -> state -> sorted county names that have yield records (drives the UI drop-downs)
COUNTIES_BY_SERIES = {}
for (series, fips) in HISTORY:
    state, name = NAME_BY_FIPS.get(fips, (None, None))
    if state:
        COUNTIES_BY_SERIES.setdefault(series, {}).setdefault(state, []).append(name)
for by_state in COUNTIES_BY_SERIES.values():
    for names in by_state.values():
        names.sort()

def records(df):
    """DataFrame -> JSON-safe records (NaN becomes null; Flask would emit invalid JSON)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


OVERRIDABLE = WEATHER + CLIMDIV + SOIL + ["lat", "lon"]
_IS_NAN = lambda v: v is None or (isinstance(v, float) and np.isnan(v))  # noqa: E731


class ApiError(Exception):
    def __init__(self, message, status=400, **extra):
        super().__init__(message)
        self.status, self.extra = status, extra


@app.errorhandler(ApiError)
def _api_error(e):
    return jsonify({"error": str(e), **e.extra}), e.status


def resolve_series(data):
    series = data.get("series")
    if not series:
        commodity = (data.get("commodity") or data.get("Crop") or "").upper()
        heads = [s for s in CATALOGUE.values() if s["commodity"] == commodity and s["headline"]]
        if not heads:
            raise ApiError(f"Unknown crop: {commodity or '(none)'}", crops=sorted({s['commodity'] for s in CATALOGUE.values()}))
        series = heads[0]["series"]
    if series not in CATALOGUE:
        raise ApiError(f"Unknown series: {series}")
    return series


def resolve_fips(state, county):
    state, county = (state or "").upper(), (county or "").upper().strip()
    fips = FIPS_BY_NAME.get((state, county)) or FIPS_BY_NAME.get((state, f"{county} COUNTY"))
    if not fips:
        raise ApiError(f"Unknown county: {county.title()}, {state}")
    return fips


def build_row(series, fips, year, overrides):
    if (series, fips) not in HISTORY:
        state, name = NAME_BY_FIPS[fips]
        raise ApiError(f"No {CATALOGUE[series]['commodity'].title()} yield records for {name}, {state}",
                       valid_states=CATALOGUE[series]["states"][:12])
    info = CATALOGUE[series]
    years, yields, n_all, sum_all = HISTORY[(series, fips)]
    row = {c: np.nan for c in FEATURES}
    row.update({k: (np.nan if _IS_NAN(v) else v) for k, v in DEFAULTS.get(fips, {}).items()})
    for key, value in overrides.items():
        if key in OVERRIDABLE and value not in (None, ""):
            row[key] = float(value)
    row.update(history_from_summary(years, yields, n_all, sum_all, year))
    state, _ = NAME_BY_FIPS[fips]
    row.update({"year": float(year), "commodity": info["commodity"], "cls": info["class"],
                "practice": info["practice"], "state": state, "series": series})
    df = pd.DataFrame([row])
    df = add_features(df)
    for c in CATEGORICAL:
        df[c] = df[c].astype(str)
    return df[FEATURES], {"last_year": years[-1], "last_yield": yields[-1]}


def predict_one(series, fips, year, overrides, model_name):
    used = {k: v for k, v in overrides.items() if k in OVERRIDABLE and v not in (None, "")}
    X, context = build_row(series, fips, year, used)
    value = float(MODELS[model_name].predict(X)[0])
    info = CATALOGUE[series]
    stats = PER_COMMODITY.loc[info["commodity"]] if info["commodity"] in PER_COMMODITY.index else None
    return {
        "series": series, "commodity": info["commodity"], "class": info["class"], "practice": info["practice"],
        "prediction": round(value, 2), "unit": info["unit"], "model": model_name,
        "last_observed": {"year": int(context["last_year"]), "yield": round(context["last_yield"], 2)},
        "typical_range": [round(info["p01_yield"], 1), round(info["p99_yield"], 1)],
        "typical_range_years": info.get("range_years"),
        "expected_error_pct": None if stats is None else round(float(stats["MedAPE"]), 1),
        "evaluated_on": f"{SUMMARY['test_years'][0]}-{SUMMARY['test_years'][1]}",
        "history": "full" if year > context["last_year"] else "partial",
        "inputs": f"county normals with your overrides ({', '.join(sorted(used))})" if used
                  else "county normals (1991-2020 average season)",
    }


def parse_request():
    data = request.get_json(silent=True) or {}
    year = int(data.get("year") or data.get("Year") or MANIFEST.get("train_end", 2020) + 6)
    model = data.get("model") or DEFAULT_MODEL
    if model not in MODELS:
        model = DEFAULT_MODEL
    return data, year, model


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "default_model": DEFAULT_MODEL, "models": list(MODELS),
                    "series": len(CATALOGUE), "features": len(FEATURES)})


@app.route("/metadata")
def metadata():
    crops = {}
    for s in CATALOGUE.values():
        crops.setdefault(s["commodity"], {"commodity": s["commodity"], "series": [], "states": set()})
        crops[s["commodity"]]["series"].append(
            {"series": s["series"], "class": s["class"], "practice": s["practice"], "unit": s["unit"],
             "headline": s["headline"], "rows": s["rows"]})
        crops[s["commodity"]]["states"].update(s["states"])
    out = [{**c, "states": sorted(c["states"])} for c in sorted(crops.values(), key=lambda c: c["commodity"])]
    return jsonify({"crops": out, "models": list(MODELS), "default_model": DEFAULT_MODEL})


@app.route("/counties/<state>")
def counties(state):
    series = request.args.get("series")
    if series:
        return jsonify({"counties": COUNTIES_BY_SERIES.get(series, {}).get(state.upper(), [])})
    names = sorted(COUNTIES[COUNTIES["state"] == state.upper()]["county"])
    return jsonify({"counties": names})


@app.route("/predict", methods=["POST"])
def predict():
    data, year, model = parse_request()
    series = resolve_series(data)
    if not data.get("state") and not data.get("State"):
        raise ApiError("state and county are required")
    fips = resolve_fips(data.get("state") or data.get("State"), data.get("county") or data.get("County"))
    return jsonify(predict_one(series, fips, year, data, model))


@app.route("/predict-regional", methods=["POST"])
def predict_regional():
    data, year, model = parse_request()
    fips = resolve_fips(data.get("state") or data.get("State"), data.get("county") or data.get("County"))
    results, absent = [], []
    for s in sorted((s for s in CATALOGUE.values() if s["headline"]), key=lambda s: s["commodity"]):
        if (s["series"], fips) in HISTORY:
            results.append(predict_one(s["series"], fips, year, data, model))
        else:
            absent.append(s["commodity"])
    return jsonify({"model": model, "year": year, "results": results, "not_grown": absent})


@app.route("/metrics")
def metrics():
    return jsonify({"test_years": SUMMARY["test_years"], "rows": SUMMARY["rows"], "best_model": SUMMARY["best_model"],
                    "best_baseline": SUMMARY["best_baseline"], "models": records(COMPARISON)})


@app.route("/metrics/commodities")
def metrics_commodities():
    return jsonify(records(PER_COMMODITY.reset_index()))


@app.route("/feature-importance/<model_name>")
def feature_importance(model_name):
    path = ARTIFACTS / "feature_importance" / (model_name.lower().replace(" ", "_").replace("-", "_") + ".csv")
    if not path.exists():
        raise ApiError(f"No feature importances for {model_name}", status=404)
    df = pd.read_csv(path)
    return jsonify({"model": model_name, "features": records(df.head(25))})


@app.route("/feature-importance")
def feature_importance_list():
    return jsonify({"available": sorted(p.stem.replace("_", " ").title() for p in (ARTIFACTS / "feature_importance").glob("*.csv"))})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=port)
