"""Growing-season weather per county from NASA POWER daily data (1981 onwards).

POWER's meteorology is on a native 0.5 x 0.625 degree grid, so counties whose centroids fall in the
same grid cell would get identical data. Each cell is requested once and shared by its counties.
For each cell we request daily T2M_MAX / T2M_MIN / PRECTOTCORR / RH2M, then compute season
features from the *daily* values:
April 1 to October 31 means and totals, growing degree days (base 10 C, daily temperatures
capped at 30 C), days above 35 C, and June / July / August monthly temperature and rainfall.
Each cell is cached on its own so an interrupted run resumes where it stopped.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import manifest
from .config import CACHE, FIRST_WEATHER_YEAR, LAST_YEAR, RAW
from .download_counties import OUT as COUNTIES
from .http import get

API = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M"]
LAT_STEP, LON_STEP = 0.5, 0.625  # MERRA-2 grid spacing
DIR = CACHE / "weather"
OUT = RAW / "weather_season.csv"
SEASON_START, SEASON_END = (4, 1), (10, 31)


def season_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily frame (DatetimeIndex, POWER columns) -> one row of features per year."""
    d = daily.copy()
    d["tmax"], d["tmin"] = d["T2M_MAX"], d["T2M_MIN"]
    d["tavg"] = (d["tmax"] + d["tmin"]) / 2
    d["prcp"], d["rh"] = d["PRECTOTCORR"], d["RH2M"]
    d["gdd"] = ((d["tmax"].clip(upper=30) + d["tmin"].clip(lower=10)) / 2 - 10).clip(lower=0)
    d["heat_day"] = (d["tmax"] > 35).astype(float).where(d["tmax"].notna())
    d["frost_day"] = (d["tmin"] < 0).astype(float).where(d["tmin"].notna())
    d["rain_day"] = (d["prcp"] >= 1).astype(float).where(d["prcp"].notna())
    m = d.index.month * 100 + d.index.day
    season = d[(m >= SEASON_START[0] * 100 + SEASON_START[1]) & (m <= SEASON_END[0] * 100 + SEASON_END[1])]
    g = season.groupby(season.index.year)
    out = pd.DataFrame({
        "tmax_season": g["tmax"].mean(), "tmin_season": g["tmin"].mean(), "tavg_season": g["tavg"].mean(),
        "prcp_season": g["prcp"].sum(min_count=150), "rh_season": g["rh"].mean(),
        "gdd_season": g["gdd"].sum(min_count=150),
        "heat_days": g["heat_day"].sum(min_count=150), "frost_days": g["frost_day"].sum(min_count=150),
        "rain_days": g["rain_day"].sum(min_count=150),
    })
    out["temp_range_season"] = out["tmax_season"] - out["tmin_season"]
    for month, name in ((6, "jun"), (7, "jul"), (8, "aug")):
        mm = season[season.index.month == month]
        gm = mm.groupby(mm.index.year)
        out[f"prcp_{name}"] = gm["prcp"].sum(min_count=25)
        out[f"tmax_{name}"] = gm["tmax"].mean()
    out.index.name = "year"
    return out.reset_index()


def cell_of(lat: float, lon: float):
    """Centre of the POWER grid cell containing a point (what the API snaps to anyway)."""
    return round(round(lat / LAT_STEP) * LAT_STEP, 3), round(round(lon / LON_STEP) * LON_STEP, 3)


def fetch_cell(lat: float, lon: float) -> pd.DataFrame:
    r = get(API, params={"parameters": ",".join(PARAMS), "community": "AG", "latitude": lat,
                         "longitude": lon, "start": f"{FIRST_WEATHER_YEAR}0101",
                         "end": f"{LAST_YEAR}1231", "format": "JSON"}, timeout=180)
    params = r.json()["properties"]["parameter"]
    daily = pd.DataFrame({p: pd.Series(v) for p, v in params.items()})
    daily.index = pd.to_datetime(daily.index, format="%Y%m%d")
    daily = daily.replace(-999, np.nan)  # POWER's missing-value flag
    return season_features(daily)


def _cell_path(cell):
    return DIR / f"{cell[0]:+07.3f}_{cell[1]:+08.3f}.csv"


def _job(cell):
    path = _cell_path(cell)
    if path.exists():
        return cell
    feats = fetch_cell(*cell)
    feats.to_csv(path, index=False)
    return cell


def run(workers: int = 6):
    if OUT.exists():
        print(f"weather: cached ({OUT})")
        return OUT
    DIR.mkdir(parents=True, exist_ok=True)
    counties = pd.read_csv(COUNTIES, dtype={"fips": str})
    counties["cell"] = [cell_of(a, b) for a, b in zip(counties["lat"], counties["lon"])]
    cells = sorted(set(counties["cell"]))
    todo = [c for c in cells if not _cell_path(c).exists()]
    print(f"weather: {len(counties)} counties share {len(cells)} grid cells; "
          f"{len(cells) - len(todo)} cached, {len(todo)} to download")
    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(_job, c): c for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as e:  # keep going; failed cells are retried on the next run
                failed.append(futures[fut])
                print(f"  {futures[fut]} failed: {e}")
            if i % 50 == 0:
                print(f"  weather {i}/{len(todo)}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} cells failed; re-run to retry them: {failed[:10]}")
    frames = []
    for cell, fips in counties.groupby("cell")["fips"]:
        f = pd.read_csv(_cell_path(cell))
        frames.extend(f.assign(fips=code) for code in fips)
    df = pd.concat(frames, ignore_index=True)
    df = df[["fips"] + [c for c in df.columns if c != "fips"]]
    df.to_csv(OUT, index=False)
    manifest.record("weather_season", source=API, path=OUT, rows=len(df),
                    note="NASA POWER daily (0.5x0.625 deg grid cells), season features per county-year")
    print(f"weather: {len(df):,} county-years")
    return OUT


if __name__ == "__main__":
    run()
