"""API smoke tests. They need the model artifacts (run `python -m agroyield.pipeline export`,
or `python api/fetch_artifacts.py` to download the published ones) and are skipped without them."""
import sys
from pathlib import Path

import pytest

ART = Path(__file__).resolve().parent.parent / "api" / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "manifest.json").exists(), reason="model artifacts not available")


@pytest.fixture(scope="module")
def client():
    sys.path.insert(0, str(ART.parent))
    from app import app
    return app.test_client()


def post(client, path, **body):
    return client.post(path, json=body)


def test_health_lists_models(client):
    d = client.get("/health").get_json()
    assert d["status"] == "healthy" and d["models"] and d["series"] > 20


def test_prediction_for_a_known_county_is_plausible(client):
    r = post(client, "/predict", commodity="CORN", state="IA", county="POLK", year=2026)
    assert r.status_code == 200
    d = r.get_json()
    lo, hi = d["typical_range"]
    # a 2026 trend can sit slightly above the last decade's 99th percentile; it must stay in the same ballpark
    assert lo < d["prediction"] < 1.25 * hi and d["unit"] == "BU / ACRE"
    assert d["last_observed"]["year"] <= 2025 and d["history"] == "full"


def test_the_same_county_gets_different_predictions_for_different_crops_and_varieties(client):
    corn = post(client, "/predict", commodity="CORN", state="IA", county="POLK", year=2026).get_json()["prediction"]
    soy = post(client, "/predict", commodity="SOYBEANS", state="IA", county="POLK", year=2026).get_json()["prediction"]
    assert abs(corn - soy) > 20


def test_overriding_weather_changes_the_prediction(client):
    base = post(client, "/predict", commodity="CORN", state="IA", county="POLK", year=2026).get_json()["prediction"]
    drought = post(client, "/predict", commodity="CORN", state="IA", county="POLK", year=2026,
                   prcp_season=120, heat_days=40, tmax_season=31).get_json()["prediction"]
    assert drought < base


def test_crops_a_county_does_not_grow_are_refused_not_extrapolated(client):
    r = post(client, "/predict", commodity="PEANUTS", state="IA", county="POLK", year=2026)
    assert r.status_code == 400 and "No Peanuts yield records" in r.get_json()["error"]


def test_unknown_inputs_return_clear_errors(client):
    assert post(client, "/predict", commodity="UNOBTAINIUM", state="IA", county="POLK").status_code == 400
    assert post(client, "/predict", commodity="CORN", state="IA", county="ATLANTIS").status_code == 400


def test_regional_lists_each_crop_once_and_reports_absent_ones(client):
    d = post(client, "/predict-regional", state="IA", county="POLK", year=2026).get_json()
    crops = [r["commodity"] for r in d["results"]]
    assert "CORN" in crops and len(crops) == len(set(crops)) and "PEANUTS" in d["not_grown"]


def test_metrics_report_baselines_next_to_models(client):
    d = client.get("/metrics").get_json()
    types = {m["Type"] for m in d["models"]}
    assert types == {"Model", "Baseline"} and d["test_years"][0] >= 2021


def test_metrics_json_is_strictly_valid(client):
    import json
    for path in ("/metrics", "/metrics/commodities"):
        text = client.get(path).get_data(as_text=True)
        json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"{c} in {path}")))


def test_response_says_whether_inputs_were_overridden(client):
    plain = post(client, "/predict", commodity="CORN", state="IA", county="POLK", year=2026).get_json()
    assert "overrides" not in plain["inputs"]
    over = post(client, "/predict", commodity="CORN", state="IA", county="POLK", year=2026, prcp_season=300).get_json()
    assert "prcp_season" in over["inputs"]


def test_counties_not_observed_recently_are_refused_with_the_reason(client):
    r = post(client, "/predict", commodity="WHEAT", state="CO", county="Alamosa", year=2026)  # last recorded 2008
    assert r.status_code == 400 and "too long ago" in r.get_json()["error"]


def test_every_served_series_gives_plausible_predictions(client):
    import app as A
    for series, info in A.CATALOGUE.items():
        state = info["states"][0]
        county = A.COUNTIES_BY_SERIES[series][state][0]
        d = post(client, "/predict", series=series, state=state, county=county, year=2026).get_json()
        lo, hi = d["typical_range"]
        assert lo / 2 < d["prediction"] < 1.5 * hi, (series, county, d["prediction"], lo, hi)


def test_models_without_feature_importance_return_an_empty_list_not_an_error(client):
    r = client.get("/feature-importance/Ridge Regression")
    assert r.status_code == 200 and r.get_json()["features"] == []


def test_each_variety_lists_only_states_that_have_counties(client):
    meta = client.get("/metadata").get_json()
    for crop in meta["crops"]:
        for s in crop["series"]:
            for state in s["states"]:
                assert client.get(f"/counties/{state}", query_string={"series": s["series"]}).get_json()["counties"], (s["series"], state)
