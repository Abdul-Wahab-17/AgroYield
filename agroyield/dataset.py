"""Load the master table, add features and split by year."""
import numpy as np
import pandas as pd

from .config import FIRST_WEATHER_YEAR, PROCESSED, TRAIN_END, VAL_END
from .features import FEATURES, add_features, add_history

MASTER = PROCESSED / "master.parquet"


def load(variant: str = "A") -> pd.DataFrame:
    """Feature table for modelling.

    Variant "A" uses years with NASA POWER daily weather (1981+). Variant "B" also keeps earlier years,
    where only NOAA nClimDiv climate and drought indices exist (POWER features are missing there).
    History features are computed over *all* years first, so 1981 rows still see earlier yields.
    """
    df = pd.read_parquet(MASTER)
    df = add_features(add_history(df))
    if variant == "A":
        df = df[df["year"] >= FIRST_WEATHER_YEAR]
    num = df.select_dtypes("float64").columns
    df[num] = df[num].astype("float32")
    return df.reset_index(drop=True)


def split(df: pd.DataFrame):
    """Temporal split. Whole years go to one side, so a model is always tested on years it never saw."""
    train = df[df["year"] <= TRAIN_END]
    val = df[(df["year"] > TRAIN_END) & (df["year"] <= VAL_END)]
    test = df[df["year"] > VAL_END]
    return train, val, test


def xy(df: pd.DataFrame):
    return df[FEATURES], np.log(df["yield"].astype("float64"))
