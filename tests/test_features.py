import numpy as np
import pandas as pd

from agroyield import features as F


def _frame():
    rows = []
    for fips, base in (("19153", 150.0), ("19155", 120.0)):
        for year in (2000, 2001, 2002, 2005, 2006, 2007, 2008):  # note the gap
            rows.append({"series": "CORN", "fips": fips, "year": year, "yield": base + year % 7})
    df = pd.DataFrame(rows)
    df["series"] = df["series"].astype("category")
    return df


def test_history_never_uses_the_current_or_future_year():
    df = F.add_history(_frame())
    first = df[(df.fips == "19153")].iloc[0]
    assert np.isnan(first["prev_yield"]) and first["hist_n"] == 0
    row = df[(df.fips == "19153") & (df.year == 2005)].iloc[0]
    assert row["prev_yield"] == 150 + 2002 % 7  # last observation before 2005, not a calendar lag
    assert row["years_since_prev"] == 3


def test_training_and_serving_history_agree():
    df = F.add_history(_frame())
    for _, r in df.iterrows():
        past = df[(df.fips == r.fips) & (df.year < r.year)].sort_values("year")
        h = F.history_features(past.year, past["yield"], r.year)
        for k, v in h.items():
            assert (np.isnan(v) and np.isnan(r[k])) or abs(v - r[k]) < 1e-9, (k, r.fips, r.year, v, r[k])


def test_derived_features_are_deterministic_and_complete():
    df = F.add_history(_frame())
    for c in F.WEATHER + F.SOIL + F.GEO:
        df[c] = 1.0
    df["prcp_jun"] = df["prcp_jul"] = df["prcp_aug"] = 50.0
    a, b = F.add_features(df), F.add_features(df)
    pd.testing.assert_frame_equal(a, b)
    assert {"aridity", "heat_dry", "loam_index", "soil_quality", "prev_ratio"} <= set(a.columns)
    assert len(set(F.FEATURES)) == len(F.FEATURES)


def test_serving_history_matches_training_for_a_future_year():
    df = F.add_history(_frame())
    r = df[df.fips == "19153"]
    # Training: pretend one more row exists for 2009 and read its history features
    extra = pd.concat([_frame(), pd.DataFrame({"series": pd.Categorical(["CORN"], categories=["CORN"]),
                                              "fips": ["19153"], "year": [2009], "yield": [0.0]})])
    train_row = F.add_history(extra.assign(series=extra["series"].astype("category")))
    train_row = train_row[(train_row.fips == "19153") & (train_row.year == 2009)].iloc[0]
    last5 = r.sort_values("year").tail(5)
    served = F.history_from_summary(last5.year, last5["yield"], n_all=len(r), sum_all=r["yield"].sum(), year=2009)
    for k, v in served.items():
        assert abs(v - train_row[k]) < 1e-9, (k, v, train_row[k])
