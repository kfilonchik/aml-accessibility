"""
config.py — shared constants for the r5py-based accessibility travel-time pipeline.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

BASE      = Path(__file__).parent.parent   # repo root (src/ lives directly under it)
DATA      = Path("/NVMESTORAGE/DATA/aml_project_buffer")    ############################
GTFS_ZIPS = DATA / "gtfs"
MATRICES  = DATA / "travel_matrix_aml_buffer"
MATRICES.mkdir(parents=True, exist_ok=True)

CRS_PT  = "EPSG:3763"   # PT-TM06, projected, metres — used for analysis/joins
CRS_WGS = "EPSG:4326"   # required by GTFS / r5py

# ── GTFS feeds — name -> zip filename under GTFS_ZIPS ──
FEEDS: dict[str, str] = {
    "metro":            "metro.zip",
    "train":            "train.zip",
    "fertagus":         "Fertagus_GTFS.zip",
    "boat_2":           "transtejo_soflusa_boat.zip",
    "cmet":             "cmet_carris.zip",
    "mts":              "mts.zip",
    "cascais":          "cascais.zip",
    "barreiro_bus":     "barreiro.zip",
    "carris_municipal": "carris_municip.zip",
#    "Transportes Alentejo Litoral" : "TranspAlentejoLitoral.zip",
#    "Rodoviária do Tejo" : "RMTejoII.zip",
#   "Barraqueiro Rodoeste" : "RDO_reconstructed_20261110.zip",
}

'''# ── Input geometries ──
BGRI_PATH = DATA / "polygons" / "BRGI_AML_with_Parishes_singlepolyg.geojson"
ORIGINS_PATH = DATA / "polygons" / "origins_v2021_single_polygon.geojson"
PARISHES_PATH = DATA / "polygons" / "AML_TSI_parishes.geojson"

# ── OSM extract for r5py's TransportNetwork ──
OSM_PBF_AML_PATH = DATA / "polygons" / "aml.osm.pbf"'''

# ── Departure: representative weekday morning commute ──
DEPARTURE_DATE      = dt.date(2026, 11, 10)   # Tuesday, chosen for full GTFS calendar coverage; Easter week?
DEPARTURE_TIME      = dt.time(8, 0)
DEPARTURE_DATETIME  = dt.datetime.combine(DEPARTURE_DATE, DEPARTURE_TIME)

DEPARTURE_WINDOW = dt.timedelta(minutes=120)

MTS_PATCH_SERVICE = "DS_inverno"       # weekday "winter" schedule — matches early April
OUTPUT = BASE / "output"
STATS_DIR = OUTPUT / "statistics"

# Accessibility time decay: S(t) = 1 / (1 + (t / IMPEDANCE_X0) ** IMPEDANCE_K)
IMPEDANCE_X0 = 37.72  # minutes; travel time at which the weight is 0.5
IMPEDANCE_K = 3.13    # controls how sharply the weight decreases

# Batch size for TravelTimeMatrix origin batches (memory/runtime control)
ORIGIN_BATCH_SIZE = 2_000
# Cap origin x destination pairs per routing call to bound batch memory use.
# Script 03 reduces the origin batch size when destinations are numerous.
MAX_MATRIX_PAIRS_PER_BATCH = 1_000_000

# To 000_prepare_osm.py
# ── Input geometries ──
BGRI_PATH        = DATA / "BRGI_AML_with_Parishes_singlepolyg.geojson"
BGRI_BUFFER_PATH = DATA / "BGRI_BufferAML.geojson"
ORIGINS_PATH     = DATA / "origins_v2021_single_polygon.geojson"
PARISHES_PATH    = DATA / "AML_TSI_parishes.geojson"

LISBON_NAMES = {"Lisbon", "Lisboa"}

# ── OSM extract for r5py's TransportNetwork ──
OSM_PBF_URL_PT     = "https://download.geofabrik.de/europe/portugal-latest.osm.pbf"
OSM_PBF_FULL_PATH  = DATA / "portugal-latest.osm.pbf"
OSM_PBF_AML_PATH   = DATA / "aml_buffer_75km.osm.pbf"
OSM_BBOX_BUFFER_M  = 75000   # metres, applied in CRS_PT before clipping
