"""NOAA nClimDiv county monthly climate, 1895 to the present.

Provides season precipitation / temperature and the **Palmer Drought Severity Index (PDSI)** and
Palmer Z-index for every county back to 1895. These cover the years before NASA POWER begins
(1981) and give the model a direct measure of crop-relevant drought.

NCEI numbers states alphabetically (01 = Alabama ... 50 = Alaska), not by FIPS, so a mapping to
FIPS is applied before the county code is appended.
"""
import numpy as np
import pandas as pd

from . import manifest
from .config import CACHE, CLIMDIV_BASE, CLIMDIV_SNAPSHOT, RAW
from .http import download_file

OUT = RAW / "climdiv_season.csv"

# NCEI state code -> state FIPS (contiguous US sorted alphabetically, then Hawaii, Alaska)
_STATES = ("AL AZ AR CA CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY "
           "NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY HI AK").split()
_FIPS = {"AL": 1, "AK": 2, "AZ": 4, "AR": 5, "CA": 6, "CO": 8, "CT": 9, "DE": 10, "FL": 12, "GA": 13,
         "HI": 15, "ID": 16, "IL": 17, "IN": 18, "IA": 19, "KS": 20, "KY": 21, "LA": 22, "ME": 23,
         "MD": 24, "MA": 25, "MI": 26, "MN": 27, "MS": 28, "MO": 29, "MT": 30, "NE": 31, "NV": 32,
         "NH": 33, "NJ": 34, "NM": 35, "NY": 36, "NC": 37, "ND": 38, "OH": 39, "OK": 40, "OR": 41,
         "PA": 42, "RI": 44, "SC": 45, "SD": 46, "TN": 47, "TX": 48, "UT": 49, "VT": 50, "VA": 51,
         "WA": 53, "WV": 54, "WI": 55, "WY": 56}
NCEI_TO_FIPS = {i + 1: _FIPS[s] for i, s in enumerate(_STATES)}

FILES = {"pcpn": "pcpncy", "tmax": "tmaxcy", "tmin": "tmincy", "pdsi": "pdsicy", "z": "zndxcy"}


def parse(path, kind: str) -> pd.DataFrame:
    """Parse a fixed-width nClimDiv county file into (fips, year, m1..m12)."""
    rows = []
    with open(path) as f:
        for line in f:
            if len(line) < 95:
                continue
            ncei_state, county, year = int(line[0:2]), int(line[2:5]), int(line[7:11])
            if ncei_state not in NCEI_TO_FIPS:
                continue
            vals = [float(line[11 + 7 * m: 18 + 7 * m]) for m in range(12)]
            rows.append((f"{NCEI_TO_FIPS[ncei_state]:02d}{county:03d}", year, *vals))
    df = pd.DataFrame(rows, columns=["fips", "year"] + [f"m{m}" for m in range(1, 13)])
    months = [f"m{m}" for m in range(1, 13)]
    # Missing months in the latest year are flagged -99.99 (temperature, indices) or -9.99 (rain)
    df[months] = df[months].where(df[months] > (-9.0 if kind == "pcpn" else -99.0))
    return df


def season_features(frames: dict) -> pd.DataFrame:
    """Monthly frames -> one row per county-year of April-October summaries."""
    key = ["fips", "year"]
    f_to_c = lambda f: (f - 32) / 1.8
    p = frames["pcpn"].set_index(key) * 25.4  # inches -> mm
    tx = f_to_c(frames["tmax"].set_index(key))
    tn = f_to_c(frames["tmin"].set_index(key))
    pdsi = frames["pdsi"].set_index(key)
    z = frames["z"].set_index(key)
    apr_oct = [f"m{m}" for m in range(4, 11)]
    out = pd.DataFrame(index=p.index)
    out["cd_prcp_season"] = p[apr_oct].sum(axis=1, min_count=7)
    out["cd_tmax_season"] = tx[apr_oct].mean(axis=1)
    out["cd_tmin_season"] = tn[apr_oct].mean(axis=1)
    for m, name in ((6, "jun"), (7, "jul"), (8, "aug")):
        out[f"cd_prcp_{name}"] = p[f"m{m}"]
        out[f"cd_tmax_{name}"] = tx[f"m{m}"]
        out[f"cd_pdsi_{name}"] = pdsi[f"m{m}"]
    out["cd_pdsi_season"] = pdsi[[f"m{m}" for m in range(4, 10)]].mean(axis=1)
    out["cd_z_jul"] = z["m7"]
    out["cd_z_season"] = z[[f"m{m}" for m in range(4, 10)]].sum(axis=1, min_count=6)
    # Moisture carried into the season: previous October to March precipitation
    prev = p[[f"m{m}" for m in (10, 11, 12)]].sum(axis=1, min_count=3).groupby(level="fips").shift(1)
    cur = p[["m1", "m2", "m3"]].sum(axis=1, min_count=3)
    out["cd_prcp_winter"] = prev + cur
    return out.reset_index()


def run():
    if OUT.exists():
        print(f"climdiv: cached ({OUT})")
        return OUT
    frames = {}
    for kind, code in FILES.items():
        name = f"climdiv-{code}-v1.0.0-{CLIMDIV_SNAPSHOT}"
        path = CACHE / name
        download_file(f"{CLIMDIV_BASE}/{name}", path)
        frames[kind] = parse(path, kind)
        print(f"  climdiv {kind}: {len(frames[kind]):,} county-years")
    out = season_features(frames)
    out.to_csv(OUT, index=False)
    manifest.record("climdiv_season", source=f"{CLIMDIV_BASE} ({CLIMDIV_SNAPSHOT})", path=OUT, rows=len(out),
                    note="NOAA nClimDiv county monthly climate + Palmer drought indices")
    print(f"climdiv: {len(out):,} county-years, {out['year'].min()}-{out['year'].max()}")
    return OUT


if __name__ == "__main__":
    run()
