"""
02_build_network.py

Builds the r5py.TransportNetwork from the AML OSM extract and the zipped
GTFS feeds (config.FEEDS), then runs a tiny smoke-test travel-time query
(Alameda -> Marquês de Pombal, two Lisbon metro stations a few stops
apart) to confirm the JVM, network build, and routing all work before
committing to the full BGRI-to-BGRI matrix in 03_travel_time_matrices.py.
"""

from __future__ import annotations

import datetime

import geopandas as gpd
import r5py
from shapely.geometry import Point

from config import CRS_WGS, DEPARTURE_DATETIME, FEEDS, GTFS_ZIPS, OSM_PBF_AML_PATH


def build_network() -> r5py.TransportNetwork:
    gtfs_zip_paths = [GTFS_ZIPS / filename for filename in FEEDS.values()]
    missing = [p for p in gtfs_zip_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing GTFS zips: {missing}")
    print(f"  OSM PBF: {OSM_PBF_AML_PATH}")
    print(f"  GTFS zips ({len(gtfs_zip_paths)}): {[p.name for p in gtfs_zip_paths]}")
    network = r5py.TransportNetwork(str(OSM_PBF_AML_PATH), [str(p) for p in gtfs_zip_paths])
    print("  TransportNetwork built OK.")
    return network


def smoke_test(network: r5py.TransportNetwork) -> None:
    origin = gpd.GeoDataFrame(
        {"id": ["Alameda"]}, geometry=[Point(-9.134086, 38.737255)], crs=CRS_WGS
    )
    destination = gpd.GeoDataFrame(
        {"id": ["Marques_de_Pombal"]}, geometry=[Point(-9.150807, 38.72488)], crs=CRS_WGS
    )
    ttm = r5py.TravelTimeMatrix(
        network,
        origins=origin,
        destinations=destination,
        departure=DEPARTURE_DATETIME,
        transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
        max_time=datetime.timedelta(minutes=120),
        max_time_walking=datetime.timedelta(minutes=45),
    )
    print("\n  Smoke-test travel time matrix:")
    print(ttm.to_string())
    if ttm.empty or ttm["travel_time"].isna().all():
        raise RuntimeError("Smoke test produced no reachable route — investigate before proceeding.")
    print(f"\n  Alameda -> Marquês de Pombal median travel time: "
          f"{ttm['travel_time'].iloc[0]:.1f} min")


def main() -> None:
    print("=" * 78)
    print("02  Build r5py TransportNetwork and smoke-test routing")
    print("=" * 78)
    print(f"\n  Departure: {DEPARTURE_DATETIME}")

    print("\n[1] Building TransportNetwork …")
    network = build_network()

    print("\n[2] Smoke test …")
    smoke_test(network)

    print("\nDone — network builds and routes correctly.")


if __name__ == "__main__":
    main()
