"""County soil properties from USDA NRCS SSURGO via the Soil Data Access web service.

For every county this computes the 0-30 cm average of eight horizon properties, weighted by
map-unit area within the county x component percentage x layer thickness. Counties SSURGO has no
data for stay missing (NaN) instead of being filled with regional placeholder values, which is
what the previous SoilGrids-fallback script did.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from . import manifest
from .config import CACHE, RAW
from .download_counties import OUT as COUNTIES
from .http import post

URL = "https://SDMDataAccess.nrcs.usda.gov/Tabular/SDMTabularService/post.rest"
PROPS = {  # SSURGO column -> our name
    "claytotal_r": "clay_pct", "sandtotal_r": "sand_pct", "silttotal_r": "silt_pct",
    "om_r": "organic_matter_pct", "ph1to1h2o_r": "ph", "dbthirdbar_r": "bulk_density",
    "cec7_r": "cec", "awc_r": "awc",
}
DIR = CACHE / "soil"
OUT = RAW / "soil_ssurgo.csv"

_W = ("CAST(mo.areaovacres AS float) * c.comppct_r * "
      "((CASE WHEN h.hzdepb_r > 30 THEN 30 ELSE h.hzdepb_r END) - h.hzdept_r)")


def query(areasymbols) -> str:
    cols = ", ".join(
        f"SUM(h.{c} * {_W}) / NULLIF(SUM(CASE WHEN h.{c} IS NOT NULL THEN {_W} END), 0) AS {c}" for c in PROPS)
    symbols = ", ".join(f"'{a}'" for a in areasymbols)
    return (f"SELECT l.areasymbol, {cols} FROM laoverlap l "
            "INNER JOIN muaoverlap mo ON mo.lareaovkey = l.lareaovkey "
            "INNER JOIN component c ON c.mukey = mo.mukey "
            "INNER JOIN chorizon h ON h.cokey = c.cokey "
            f"WHERE l.areatypename = 'County or Parish' AND l.areasymbol IN ({symbols}) "
            "AND h.hzdept_r < 30 GROUP BY l.areasymbol")


def parse(payload: dict) -> pd.DataFrame:
    table = payload.get("Table", [])
    if len(table) < 2:
        return pd.DataFrame(columns=["areasymbol"] + list(PROPS))
    return pd.DataFrame(table[1:], columns=[c.lower() for c in table[0]]).astype(
        {c: float for c in PROPS})


def fetch(areasymbols) -> pd.DataFrame:
    r = post(URL, data={"query": query(areasymbols), "format": "JSON+COLUMNNAME"}, timeout=240)
    return parse(r.json()) if r.text.strip() else parse({})


def _job(key, rows):
    path = DIR / f"{key}.csv"
    if path.exists():
        return key
    df = fetch([f"{r.state}{r.county_fips:03d}" for r in rows]).rename(columns=PROPS)
    df.to_csv(path, index=False)
    return key


def run(workers: int = 3, batch: int = 8):
    if OUT.exists():
        print(f"soil: cached ({OUT})")
        return OUT
    DIR.mkdir(parents=True, exist_ok=True)
    counties = pd.read_csv(COUNTIES, dtype={"fips": str})
    jobs = []  # small batches keep each SDA query fast; each batch is cached on its own
    for state, grp in counties.groupby("state"):
        rows = list(grp.itertuples())
        for i in range(0, len(rows), batch):
            jobs.append((f"{state}_{i // batch:02d}", rows[i:i + batch]))
    todo = [j for j in jobs if not (DIR / f"{j[0]}.csv").exists()]
    print(f"soil: {len(jobs) - len(todo)} batches cached, {len(todo)} to query")
    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(_job, k, rows): k for k, rows in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:
                failed.append(futs[f])
                print(f"  {futs[f]} failed: {e}")
            if i % 25 == 0:
                print(f"  soil {i}/{len(todo)}")
    if failed:
        raise SystemExit(f"{len(failed)} soil batches failed; re-run to retry: {failed[:10]}")
    df = pd.concat((pd.read_csv(p) for p in sorted(DIR.glob("*.csv"))), ignore_index=True)
    df = df.rename(columns={"areasymbol": "areasymbol"})
    df["fips"] = df["areasymbol"].str[:2].map(
        counties.drop_duplicates("state").set_index("state")["state_fips"]).astype(int).astype(str).str.zfill(2) \
        + df["areasymbol"].str[2:].str.zfill(3)
    df = df[["fips"] + list(PROPS.values())].round(3)
    df.to_csv(OUT, index=False)
    manifest.record("soil_ssurgo", source=URL, path=OUT, rows=len(df),
                    note="SSURGO 0-30 cm area/depth-weighted county means; counties without data are absent")
    print(f"soil: {len(df)} counties with SSURGO data of {len(counties)}")
    return OUT


if __name__ == "__main__":
    run()
