"""US counties (FIPS, name, centroid, land area) from the Census 2020 Gazetteer.

2020 is used rather than a later vintage because NASS still reports Connecticut by its
eight old counties, which the 2022+ planning regions replaced.
"""
import io
import zipfile

import pandas as pd

from . import manifest
from .config import CENSUS_COUNTIES_URL, RAW
from .http import get

OUT = RAW / "counties.csv"


def run():
    if OUT.exists():
        print(f"counties: cached ({OUT})")
        return OUT
    z = zipfile.ZipFile(io.BytesIO(get(CENSUS_COUNTIES_URL).content))
    df = pd.read_csv(z.open(z.namelist()[0]), sep="\t", dtype={"GEOID": str}, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({
        "fips": df["GEOID"].str.zfill(5),
        "state_fips": df["GEOID"].str.zfill(5).str[:2].astype(int),
        "county_fips": df["GEOID"].str.zfill(5).str[2:].astype(int),
        "state": df["USPS"],
        "county": df["NAME"],
        "lat": df["INTPTLAT"].astype(float),
        "lon": df["INTPTLONG"].astype(float),
        "land_sqmi": df["ALAND_SQMI"].astype(float),
    })
    # Contiguous US + Alaska/Hawaii are in NASS; territories are not modelled
    out = out[out["state"].isin(US_STATES)].reset_index(drop=True)
    out.to_csv(OUT, index=False)
    manifest.record("counties", source=CENSUS_COUNTIES_URL, path=OUT, rows=len(out))
    print(f"counties: {len(out)} written")
    return OUT


US_STATES = set(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM "
    "NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()
)

if __name__ == "__main__":
    run()
