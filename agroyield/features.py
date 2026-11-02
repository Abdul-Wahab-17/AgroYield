"""Feature engineering shared by training and the API.

Every function here is a pure, deterministic transformation with **no fitted state** (no medians,
thresholds or scalers learned from data), so the API computes exactly what training computed.
Anything that must be learned from data lives inside the scikit-learn Pipeline that is saved
together with each model.

History features describe a county's own recent yields for the same series. They only ever use
observations from *strictly earlier years*, so they are available when a season starts and never
leak the target.
"""
import numpy as np
import pandas as pd

CATEGORICAL = ["commodity", "cls", "practice", "state", "series"]

WEATHER = ["tmax_season", "tmin_season", "tavg_season", "prcp_season", "rh_season", "gdd_season",
           "heat_days", "frost_days", "rain_days", "temp_range_season",
           "prcp_jun", "prcp_jul", "prcp_aug", "tmax_jun", "tmax_jul", "tmax_aug"]
CLIMDIV = ["cd_prcp_season", "cd_tmax_season", "cd_tmin_season", "cd_prcp_jun", "cd_prcp_jul", "cd_prcp_aug",
           "cd_tmax_jun", "cd_tmax_jul", "cd_tmax_aug", "cd_pdsi_jun", "cd_pdsi_jul", "cd_pdsi_aug",
           "cd_pdsi_season", "cd_z_jul", "cd_z_season", "cd_prcp_winter"]
SOIL = ["clay_pct", "sand_pct", "silt_pct", "organic_matter_pct", "ph", "bulk_density", "cec", "awc"]
GEO = ["lat", "lon", "land_sqmi"]
HISTORY = ["prev_yield", "prev5_mean", "hist_mean", "hist_n", "years_since_prev"]
DERIVED = ["year_c", "aridity", "heat_dry", "summer_prcp", "loam_index", "soil_quality", "prev_ratio"]

NUMERIC = ["year"] + GEO + WEATHER + CLIMDIV + SOIL + HISTORY + DERIVED
FEATURES = CATEGORICAL + NUMERIC


def history_features(prev_years, prev_yields, year):
    """History features for one county-series at `year`, from its earlier observations.

    `prev_years` / `prev_yields` are all observations before `year`, oldest first.
    Mirrors :func:`add_history` exactly (see tests/test_features.py).
    """
    y = np.asarray(prev_yields, dtype="float64")
    yrs = np.asarray(prev_years, dtype="float64")
    if len(y) == 0:
        return {"prev_yield": np.nan, "prev5_mean": np.nan, "hist_mean": np.nan, "hist_n": 0.0,
                "years_since_prev": np.nan}
    return {"prev_yield": y[-1], "prev5_mean": y[-5:].mean(), "hist_mean": y.mean(),
            "hist_n": float(len(y)), "years_since_prev": float(year - yrs[-1])}


def history_from_summary(years, yields, n_all, sum_all, year):
    """History features at serving time from what the API stores per county-series: the last five
    observations plus the count and sum of *all* observations.

    For a year after the last observation this equals training's :func:`add_history` exactly. For an
    earlier year only the stored window is available, so history is approximated from it.
    """
    years, yields = list(years), list(yields)
    if years and year > years[-1]:
        return {"prev_yield": float(yields[-1]), "prev5_mean": float(np.mean(yields[-5:])),
                "hist_mean": float(sum_all / n_all), "hist_n": float(n_all),
                "years_since_prev": float(year - years[-1])}
    past = [(y, v) for y, v in zip(years, yields) if y < year]
    return history_features([y for y, _ in past], [v for _, v in past], year)


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add history features to a frame with columns series, fips, year, yield (all years present)."""
    df = df.sort_values(["series", "fips", "year"]).copy()
    g = df.groupby(["series", "fips"], observed=True)
    y = df["yield"].astype("float64")
    prev = [g["yield"].shift(k).astype("float64") for k in range(1, 6)]
    df["prev_yield"] = prev[0]
    df["prev5_mean"] = pd.concat(prev, axis=1).mean(axis=1)  # mean of the last up-to-5 observations
    n = g.cumcount()
    df["hist_n"] = n.astype("float64")
    df["hist_mean"] = (g["yield"].cumsum().astype("float64") - y) / n.where(n > 0)
    df["years_since_prev"] = (df["year"] - g["year"].shift(1)).astype("float64")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features. Requires the history columns (see :func:`add_history`)."""
    df = df.copy()
    f = lambda c: df[c].astype("float64")
    df["year_c"] = f("year") - 2000
    df["aridity"] = f("prcp_season") / (f("gdd_season") + 1)
    df["summer_prcp"] = f("prcp_jun") + f("prcp_jul") + f("prcp_aug")
    df["heat_dry"] = f("heat_days") / (f("prcp_jul") + f("prcp_aug") + 1)
    df["loam_index"] = 100 - (abs(f("clay_pct") - 20) + abs(f("sand_pct") - 40) + abs(f("silt_pct") - 40))
    df["soil_quality"] = (f("organic_matter_pct") * 0.3 + (7 - abs(f("ph") - 6.5)) * 0.2
                          + f("awc") * 100 * 0.3 + f("cec") * 0.2)
    df["prev_ratio"] = f("prev_yield") / f("hist_mean")
    return df
