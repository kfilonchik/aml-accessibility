"""
config.py — shared constants for the r5py-based accessibility travel-time pipeline.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

BASE      = Path(__file__).parent.parent   # repo root (src/ lives directly under it)
DATA      = BASE / "data"
GTFS_ZIPS = DATA / "gtfs"
MATRICES  = BASE / "output/travel_matrix"
MATRICES.mkdir(parents=True, exist_ok=True)

CRS_PT  = "EPSG:3763"   # PT-TM06, projected, metres — used for analysis/joins
CRS_WGS = "EPSG:4326"   # required by GTFS / r5py

# ── GTFS feeds — name -> zip filename under GTFS_ZIPS ──
FEEDS: dict[str, str] = {
    "metro":            "metro.gtfs.zip",
    "train":            "train.gtfs.zip",
    "fertagus":         "fertagus.gtfs.zip",
    "boat_2":           "boat_2.gtfs.zip",
    "cmet":             "cmet.gtfs.zip",
    "mts":              "mts.gtfs.zip",
    "cascais":          "cascais.gtfs.zip",
    "barreiro_bus":     "barreiro_bus.gtfs.zip",
    "carris_municipal": "carris_municipal.zip",
}

# ── Input geometries ──
BGRI_PATH = DATA / "polygons" / "BRGI_AML_with_Parishes_singlepolyg.geojson"

# ── OSM extract for r5py's TransportNetwork ──
OSM_PBF_AML_PATH = DATA / "polygons" / "aml.osm.pbf"

# ── Departure: representative weekday morning commute ──
DEPARTURE_DATE      = dt.date(2026, 4, 7)   # Tuesday, chosen for full GTFS calendar coverage
DEPARTURE_TIME      = dt.time(8, 0)
DEPARTURE_DATETIME  = dt.datetime.combine(DEPARTURE_DATE, DEPARTURE_TIME)

# Batch size for TravelTimeMatrix origin batches (memory/runtime control)
ORIGIN_BATCH_SIZE = 2_000
