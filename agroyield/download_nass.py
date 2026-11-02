"""USDA NASS county-level crop yields from the public Quick Stats bulk file.

Downloads the pinned `qs.crops_<snapshot>.txt.gz` (about 1.1 GB, no API key needed) and streams
it in chunks, keeping county-level YIELD and AREA HARVESTED observations for every crop.
The output has one row per (commodity, class, practice, county, year) with a yield and its unit.
"""
import pandas as pd

from . import manifest
from .config import CACHE, NASS_SNAPSHOT, NASS_URL, RAW
from .http import download_file

GZ = CACHE / f"qs.crops_{NASS_SNAPSHOT}.txt.gz"
OUT = RAW / "nass_county.csv"

USECOLS = ["SOURCE_DESC", "COMMODITY_DESC", "CLASS_DESC", "PRODN_PRACTICE_DESC", "UTIL_PRACTICE_DESC",
           "STATISTICCAT_DESC", "UNIT_DESC", "DOMAIN_DESC", "AGG_LEVEL_DESC", "STATE_FIPS_CODE",
           "COUNTY_CODE", "YEAR", "FREQ_DESC", "REFERENCE_PERIOD_DESC", "VALUE"]
KEY = ["COMMODITY_DESC", "CLASS_DESC", "PRODN_PRACTICE_DESC", "UTIL_PRACTICE_DESC",
       "STATE_FIPS_CODE", "COUNTY_CODE", "YEAR"]


def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk[
        (chunk["AGG_LEVEL_DESC"] == "COUNTY")
        & (chunk["DOMAIN_DESC"] == "TOTAL")
        & (chunk["STATISTICCAT_DESC"].isin(["YIELD", "AREA HARVESTED"]))
        & (chunk["FREQ_DESC"] == "ANNUAL")
        & (chunk["REFERENCE_PERIOD_DESC"] == "YEAR")
    ].copy()
    # "(D)" withheld, "(Z)" rounds to zero, "(NA)": none of these are observations
    chunk["VALUE"] = pd.to_numeric(chunk["VALUE"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    chunk = chunk.dropna(subset=["VALUE", "COUNTY_CODE"])
    # county code 998 is NASS's "other (combined) counties" bucket, not a real county
    chunk = chunk[chunk["COUNTY_CODE"] < 998]
    return chunk


def run():
    if OUT.exists():
        print(f"nass: cached ({OUT})")
        return OUT
    download_file(NASS_URL, GZ)
    parts = []
    for i, chunk in enumerate(pd.read_csv(GZ, sep="\t", usecols=USECOLS, chunksize=500_000,
                                          dtype=str, low_memory=False)):
        for col in ("COUNTY_CODE", "STATE_FIPS_CODE", "YEAR"):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        parts.append(_clean_chunk(chunk))
        if i % 10 == 0:
            print(f"  nass chunk {i}: {sum(len(p) for p in parts):,} kept")
    df = pd.concat(parts, ignore_index=True)

    # A county-year can be published by both the annual survey and the five-yearly census.
    # Prefer the survey; the census value is used only when no survey value exists.
    df["_pref"] = (df["SOURCE_DESC"] != "SURVEY").astype(int)
    df = df.sort_values("_pref").drop_duplicates(KEY + ["STATISTICCAT_DESC", "UNIT_DESC"], keep="first")

    yld = df[df["STATISTICCAT_DESC"] == "YIELD"].rename(columns={"VALUE": "yield", "UNIT_DESC": "unit"})
    area = df[(df["STATISTICCAT_DESC"] == "AREA HARVESTED") & (df["UNIT_DESC"] == "ACRES")]
    area = area[KEY + ["VALUE"]].rename(columns={"VALUE": "area_harvested"})
    out = yld[KEY + ["unit", "yield", "SOURCE_DESC"]].merge(area, on=KEY, how="left")
    out = out.rename(columns={"COMMODITY_DESC": "commodity", "CLASS_DESC": "class",
                              "PRODN_PRACTICE_DESC": "practice", "UTIL_PRACTICE_DESC": "utilization",
                              "STATE_FIPS_CODE": "state_fips", "COUNTY_CODE": "county_fips",
                              "YEAR": "year", "SOURCE_DESC": "source"})
    out[["state_fips", "county_fips", "year"]] = out[["state_fips", "county_fips", "year"]].astype(int)
    out = out[out["yield"] > 0].sort_values(["commodity", "class", "practice", "state_fips", "county_fips", "year"])
    out.to_csv(OUT, index=False)
    manifest.record("nass_county", source=NASS_URL, path=OUT, rows=len(out),
                    note=f"Quick Stats crops snapshot {NASS_SNAPSHOT}; county-level annual yields")
    print(f"nass: {len(out):,} county yield observations, {out['commodity'].nunique()} commodities, "
          f"{out['year'].min()}-{out['year'].max()}")
    return OUT


if __name__ == "__main__":
    run()
