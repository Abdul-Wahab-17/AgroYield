"""Join NASS yields, county geography, weather, climate and soil into one modelling table.

Everything is joined on the 5-digit county FIPS code; counties are never matched by name (the
old pipeline matched on names, silently dropping counties such as "ST CLAIR" / "St. Clair").

A *series* is one published yield definition: commodity x class x production practice x
utilization x unit (for example CORN | ALL CLASSES | IRRIGATED | BU / ACRE). Every series with
enough county-years is kept and modelled; the series becomes a categorical feature.
"""
import json

import numpy as np
import pandas as pd

from . import download_climdiv, download_counties, download_nass, download_soil, download_weather
from .config import MIN_SERIES_ROWS, PROCESSED

OUT = PROCESSED / "master.parquet"
CATALOGUE = PROCESSED / "series_catalogue.json"
REPORT = PROCESSED / "merge_report.json"


def build_series(nass: pd.DataFrame) -> pd.DataFrame:
    nass = nass.copy()
    nass["series"] = (nass["commodity"] + " | " + nass["class"] + " | " + nass["practice"] + " | "
                      + nass["utilization"] + " | " + nass["unit"])
    return nass


def run() -> pd.DataFrame:
    counties = pd.read_csv(download_counties.OUT, dtype={"fips": str})
    nass = pd.read_csv(download_nass.OUT)
    report = {"nass_rows": len(nass)}

    nass["fips"] = nass["state_fips"].astype(str).str.zfill(2) + nass["county_fips"].astype(str).str.zfill(3)
    nass = nass.merge(counties[["fips", "state", "county", "lat", "lon", "land_sqmi"]], on="fips", how="inner")
    report["rows_after_county_match"] = len(nass)
    report["rows_without_county"] = report["nass_rows"] - len(nass)

    nass = build_series(nass)
    sizes = nass.groupby("series").size()
    keep = sizes[sizes >= MIN_SERIES_ROWS].index
    report["series_total"], report["series_kept"] = int(len(sizes)), int(len(keep))
    nass = nass[nass["series"].isin(keep)]
    report["rows_in_kept_series"] = len(nass)

    weather = pd.read_csv(download_weather.OUT, dtype={"fips": str})
    climdiv = pd.read_csv(download_climdiv.OUT, dtype={"fips": str})
    soil = pd.read_csv(download_soil.OUT, dtype={"fips": str})
    df = (nass.merge(weather, on=["fips", "year"], how="left")
              .merge(climdiv, on=["fips", "year"], how="left")
              .merge(soil, on="fips", how="left"))
    report["rows_with_power_weather"] = int(df["tmax_season"].notna().sum())
    report["rows_with_climdiv"] = int(df["cd_prcp_season"].notna().sum())
    report["rows_with_soil"] = int(df["clay_pct"].notna().sum())
    # A year the weather source covers but where this county has no value means missing data, not 0
    df = df.drop(columns=["utilization"]).rename(columns={"class": "cls"})

    for c in ("commodity", "cls", "practice", "state", "series", "unit", "source"):
        df[c] = df[c].astype("category")
    num = df.select_dtypes("float64").columns
    df[num] = df[num].astype("float32")
    df = df.sort_values(["series", "fips", "year"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False)

    # Catalogue used by the API and UI: what can be predicted, in which units, where
    cat = []
    for series, g in df.groupby("series", observed=True):
        row = g.iloc[0]
        recent = g[g["year"] >= g["year"].max() - 9]
        cat.append({
            "series": series, "commodity": str(row["commodity"]), "class": str(row["cls"]),
            "practice": str(row["practice"]), "unit": str(row["unit"]), "rows": int(len(g)),
            "first_year": int(g["year"].min()), "last_year": int(g["year"].max()),
            "states": sorted(g["state"].astype(str).unique().tolist()),
            "median_yield": float(g["yield"].median()),
            # Typical range over the latest decade only: yields have risen steeply, so a 1909-2025 range is misleading
            "p01_yield": float(recent["yield"].quantile(0.01)), "p99_yield": float(recent["yield"].quantile(0.99)),
            "range_years": [int(recent["year"].min()), int(recent["year"].max())],
        })
    # The headline series of a commodity is its largest all-practices series
    best = {}
    for s in cat:
        if s["practice"] == "ALL PRODUCTION PRACTICES" and s["rows"] > best.get(s["commodity"], (None, 0))[1]:
            best[s["commodity"]] = (s["series"], s["rows"])
    for s in cat:
        s["headline"] = best.get(s["commodity"], (None,))[0] == s["series"]
    cat.sort(key=lambda s: -s["rows"])
    CATALOGUE.write_text(json.dumps(cat, indent=1))
    report.update({"rows": len(df), "counties": int(df["fips"].nunique()), "commodities": int(df["commodity"].nunique()),
                   "years": [int(df["year"].min()), int(df["year"].max())]})
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return df


if __name__ == "__main__":
    run()
