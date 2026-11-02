"""Paths and pinned data-source settings shared by every pipeline stage."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "api" / "artifacts"

# NASS publishes a new full snapshot of the crops table every few days.
# Pinning the snapshot makes the dataset reproducible; bump it to refresh.
NASS_SNAPSHOT = "20261031"
NASS_URL = f"https://www.nass.usda.gov/datasets/qs.crops_{NASS_SNAPSHOT}.txt.gz"

CENSUS_COUNTIES_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2020_Gazetteer/2020_Gaz_counties_national.zip"
)

# Weather and yield years used for modelling
FIRST_WEATHER_YEAR = 1981
LAST_YEAR = 2025
# Temporal split: models never see a year later than the one they are evaluated on
TRAIN_END, VAL_END = 2020, 2022  # train <= 2020, val 2021-2022, test 2023-2025

# A (commodity, class, practice, unit) series needs this many county-years to be modelled
MIN_SERIES_ROWS = 1000

for d in (RAW, CACHE, PROCESSED, MODELS):
    d.mkdir(parents=True, exist_ok=True)

# NOAA nClimDiv county monthly climate (precipitation, temperature, drought indices) from 1895
CLIMDIV_SNAPSHOT = "20261006"
CLIMDIV_BASE = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv"
