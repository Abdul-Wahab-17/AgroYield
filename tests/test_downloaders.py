import pandas as pd

from agroyield import download_climdiv, download_nass, download_soil


def test_soil_query_uses_state_and_county_fips_symbols():
    q = download_soil.query(["IA153", "GA205"])
    assert "'IA153', 'GA205'" in q and "CAST(mo.areaovacres AS float)" in q  # float cast avoids int overflow


def test_soil_job_builds_symbols_from_each_county_not_the_batch_key(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(download_soil, "DIR", tmp_path)
    monkeypatch.setattr(download_soil, "fetch", lambda syms: seen.extend(syms) or download_soil.parse({}))
    row = type("R", (), {"state": "IA", "county_fips": 153})()
    download_soil._job("IA_00", [row])
    assert seen == ["IA153"]


def test_nass_cleaning_drops_suppressed_combined_and_non_county_rows():
    cols = download_nass.USECOLS
    base = ["SURVEY", "CORN", "ALL CLASSES", "ALL PRODUCTION PRACTICES", "ALL UTILIZATION PRACTICES",
            "YIELD", "BU / ACRE", "TOTAL", "COUNTY", 19, 153, 2020, "ANNUAL", "YEAR", "1,234.5"]
    rows = [base,
            base[:10] + [998] + base[11:],            # "other counties" bucket
            base[:14] + ["(D)"],                      # withheld
            base[:8] + ["STATE"] + base[9:],          # not county level
            base[:7] + ["FERTILIZER"] + base[8:]]     # not the total domain
    out = download_nass._clean_chunk(pd.DataFrame(rows, columns=cols))
    assert list(out["VALUE"]) == [1234.5]


def test_climdiv_maps_ncei_state_codes_to_fips(tmp_path):
    line = lambda st, yr, v: f"{st:02d}00105{yr}" + "".join(f"{x:7.2f}" for x in [v] * 12) + "\n"
    p = tmp_path / "f"
    p.write_text(line(13, 2000, 1.0) + line(13, 2001, -99.99) + line(99, 2000, 1.0))  # NCEI 13 = Iowa
    df = download_climdiv.parse(p, "pdsi")
    assert list(df.fips) == ["19001", "19001"]
    assert df.iloc[0].m1 == 1.0 and pd.isna(df.iloc[1].m1)
