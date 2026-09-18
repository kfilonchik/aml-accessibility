"""
03_travel_time_matrices.py

Computes a single batched r5py travel-time matrix between BGRI-cell
centroids across the whole AML (Área Metropolitana de Lisboa), including
Lisbon municipality on both the origin and destination side, for the
chosen weekday-morning-commute departure (config.DEPARTURE_DATETIME).

Every BGRI cell is both an origin and a destination, so any cell's travel
time to any other cell (not just to Lisbon) is available in the output.

The matrix is computed in ORIGIN_BATCH_SIZE-row batches and checkpointed
to MATRICES/aml_matrix_batch_*.parquet immediately, so a crash or restart
resumes rather than recomputing from scratch.
"""

from __future__ import annotations

import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import r5py

from config import (
    CRS_WGS, DEPARTURE_DATETIME, FEEDS, GTFS_ZIPS, MATRICES,
    ORIGIN_BATCH_SIZE, OSM_PBF_AML_PATH, BGRI_PATH,
)


def build_network() -> r5py.TransportNetwork:
    gtfs_zip_paths = [GTFS_ZIPS / filename for filename in FEEDS.values()]
    network = r5py.TransportNetwork(str(OSM_PBF_AML_PATH), [str(p) for p in gtfs_zip_paths])
    print("  TransportNetwork built OK.")
    return network


def load_bgri_centroids() -> gpd.GeoDataFrame:
    """One point per BGRI cell (representative_point(), AML-wide, Lisbon included),
    used as both origins and destinations."""
    bgri = gpd.read_file(BGRI_PATH)
    bgri = bgri.rename(columns={"BGRI2021": "id"})
    bgri["geometry"] = bgri.geometry.representative_point()
    keep_cols = ["id", "Municipality", "Dicofre", "N_INDIVIDUOS", "geometry"]
    print(f"  {len(bgri)} BGRI cells ({bgri['Municipality'].nunique()} municipalities)")
    return bgri[keep_cols].to_crs(CRS_WGS)


def run_matrix_batched(
    network: r5py.TransportNetwork,
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    out_prefix: str,
) -> pd.DataFrame:
    n_batches = int(np.ceil(len(origins) / ORIGIN_BATCH_SIZE))
    batch_frames = []
    for i in range(n_batches):
        out_path = MATRICES / f"{out_prefix}_batch_{i:03d}.parquet"
        if out_path.exists():
            print(f"    batch {i+1}/{n_batches}: already computed, loading from disk")
            batch_frames.append(pd.read_parquet(out_path))
            continue
        batch = origins.iloc[i * ORIGIN_BATCH_SIZE:(i + 1) * ORIGIN_BATCH_SIZE]
        print(f"    batch {i+1}/{n_batches}: {len(batch)} origins x {len(destinations)} destinations …")
        ttm = r5py.TravelTimeMatrix(
            network,
            origins=batch[["id", "geometry"]],
            destinations=destinations[["id", "geometry"]],
            departure=DEPARTURE_DATETIME,
            transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
            max_time=datetime.timedelta(minutes=120),
            max_time_walking=datetime.timedelta(minutes=45),
        )
        ttm = pd.DataFrame(ttm)
        ttm.to_parquet(out_path)
        batch_frames.append(ttm)
    return pd.concat(batch_frames, ignore_index=True)


def main() -> None:
    print("=" * 78)
    print("03  Batched r5py travel-time matrix (AML-wide, BGRI cell <-> BGRI cell)")
    print("=" * 78)

    print("\n[1] Building TransportNetwork …")
    network = build_network()

    print("\n[2] Loading BGRI centroids (origins = destinations, Lisbon included) …")
    bgri_centroids = load_bgri_centroids()

    print(f"\n[3] Computing AML-wide matrix ({len(bgri_centroids)} x {len(bgri_centroids)}) …")
    matrix = run_matrix_batched(network, bgri_centroids, bgri_centroids, "aml_matrix")
    matrix.to_parquet(MATRICES / "aml_matrix.parquet")
    print(f"    Saved {len(matrix)}-row aml_matrix.parquet")

    print("\nDone.")


if __name__ == "__main__":
    main()
