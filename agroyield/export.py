"""Build the files the API serves from the trained models and the master table.

    models/variant_X/*.joblib  ->  api/artifacts/models/
    county climatology         ->  api/artifacts/county_defaults.json
    recent yields per county   ->  api/artifacts/history.parquet
    series list, metrics       ->  api/artifacts/*.json / *.csv
"""
import json
import shutil

import joblib
import numpy as np
import pandas as pd

from . import dataset, merge
from .config import ARTIFACTS, MODELS, PROCESSED, TRAIN_END
from .features import CLIMDIV, FEATURES, GEO, SOIL, WEATHER

NORMALS = (1991, 2020)  # the 30-year window used as "normal" weather for blank inputs


def county_defaults(master: pd.DataFrame) -> dict:
    """Per county: static soil/geography plus normal-season weather (means over 1991-2020)."""
    cols = GEO + SOIL
    static = master.groupby("fips", observed=True)[cols].first()
    seasonal = WEATHER + CLIMDIV
    window = master[(master["year"] >= NORMALS[0]) & (master["year"] <= NORMALS[1])]
    normals = window.groupby("fips", observed=True)[seasonal].mean()
    allyears = master.groupby("fips", observed=True)[seasonal].mean()
    normals = normals.combine_first(allyears)  # counties without data in the window
    out = static.join(normals, how="left")
    return {fips: {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in row.items()}
            for fips, row in out.iterrows()}


def history_table(master: pd.DataFrame) -> pd.DataFrame:
    """Last five observed yields per county-series (what the history features are computed from)."""
    m = master[["series", "fips", "year", "yield"]].sort_values(["series", "fips", "year"])
    last = m.groupby(["series", "fips"], observed=True).tail(5)
    g = last.groupby(["series", "fips"], observed=True)
    totals = m.groupby(["series", "fips"], observed=True)["yield"].agg(n_all="size", sum_all="sum")
    recent = pd.DataFrame({"years": g["year"].apply(list),
                           "yields": g["yield"].apply(lambda s: [float(x) for x in s])})
    return recent.join(totals).reset_index()


def feature_importance(model) -> pd.DataFrame | None:
    est = getattr(model, "estimator", None)
    imp = getattr(est, "feature_importances_", None)
    if imp is None:
        return None
    names = [n.split("__", 1)[-1] for n in model.prep.get_feature_names_out()]
    if len(names) != len(imp):
        return None
    return pd.DataFrame({"Feature": names, "Importance": np.asarray(imp, dtype="float64")}).sort_values(
        "Importance", ascending=False)


def run(variant: str = "A", ship=None, max_model_mb: float = 150.0):
    src = MODELS / f"variant_{variant}"
    out = ARTIFACTS
    if out.exists():
        shutil.rmtree(out)
    (out / "models").mkdir(parents=True)
    (out / "feature_importance").mkdir()

    master = pd.read_parquet(dataset.MASTER)
    (out / "county_defaults.json").write_text(json.dumps(county_defaults(master)))
    history_table(master).to_parquet(out / "history.parquet", index=False)
    shutil.copy(merge.CATALOGUE, out / "series_catalogue.json")
    counties = pd.read_csv(PROCESSED.parent / "raw" / "counties.csv", dtype={"fips": str})
    counties[["fips", "state", "county"]].to_csv(out / "counties.csv", index=False)

    summary = json.loads((src / "summary.json").read_text())
    for f in ("model_comparison.csv", "per_commodity.csv", "per_series.csv", "summary.json"):
        shutil.copy(src / f, out / f)

    comparison = pd.read_csv(src / "model_comparison.csv")
    wanted = ship or comparison.loc[comparison["Type"] == "Model", "Model"].tolist()
    shipped = []
    for path in sorted(src.glob("*.joblib")):
        model = joblib.load(path)
        size_mb = path.stat().st_size / 1e6
        if model.name not in wanted or size_mb > max_model_mb or model.name == "Blend":
            continue
        shutil.copy(path, out / "models" / path.name)
        shipped.append({"name": model.name, "file": path.name, "mb": round(size_mb, 1)})
        fi = feature_importance(model)
        if fi is not None:
            fi.to_csv(out / "feature_importance" / (path.stem + ".csv"), index=False)
    best = summary["best_model"]
    if best not in [m["name"] for m in shipped]:  # best is too large or a blend: fall back to the best shipped one
        best = next(r["Model"] for _, r in comparison.iterrows() if r["Model"] in [m["name"] for m in shipped])
    (out / "manifest.json").write_text(json.dumps({"variant": variant, "default_model": best, "models": shipped,
                                                  "features": FEATURES, "train_end": TRAIN_END}, indent=1))
    print(f"exported {len(shipped)} models ({sum(m['mb'] for m in shipped):.0f} MB), default {best}")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "A")
