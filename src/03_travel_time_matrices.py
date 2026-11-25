"""
Compute AML -> AML and AML -> external 75 km buffer travel-time matrices.

All supplied BGRI cells are retained, including Lisbon and cells intersecting
the buffer boundary. Only AML cells are origins. Each result batch is saved
as a separate Parquet file; no full matrix is concatenated or loaded into RAM.

Outputs under config.MATRICES:
  aml_matrix_batch_*.parquet       (AML -> AML, compatible with script 04)
  aml_to_buffer_batch_*.parquet    (AML -> buffer)
  <prefix>_manifest.json          (settings and inputs required for resuming)

Existing batches without a matching manifest are rejected. Move older results
elsewhere or choose a fresh MATRICES directory before changing inputs/settings.

Select calculations with --matrix aml-aml, --matrix aml-buffer, or --matrix both
(default). Both runs AML -> AML first, then AML -> buffer.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import r5py

from config import (
    CRS_WGS, DEPARTURE_DATETIME, DEPARTURE_WINDOW, FEEDS, GTFS_ZIPS, MATRICES,
    ORIGIN_BATCH_SIZE, MAX_MATRIX_PAIRS_PER_BATCH, OSM_PBF_AML_PATH, BGRI_PATH, BGRI_BUFFER_PATH,
)


def build_network() -> r5py.TransportNetwork:
    gtfs_zip_paths = [GTFS_ZIPS / filename for filename in FEEDS.values()]
    network = r5py.TransportNetwork(str(OSM_PBF_AML_PATH), [str(p) for p in gtfs_zip_paths])
    print("  TransportNetwork built OK.")
    return network


def load_bgri_centroids(path: Path, region: str) -> gpd.GeoDataFrame:
    """Preserve all supplied cells and create one interior routing point per ID.

    Region membership comes from the source dataset. Cells intersecting the
    outer buffer boundary are retained, without clipping their polygons or
    filtering the resulting points by distance.
    """
    bgri = gpd.read_file(path)
    if bgri.crs is None:
        raise ValueError(f"{path.name}: geometry CRS is missing")
    if "BGRI2021" not in bgri or bgri["BGRI2021"].isna().any():
        raise ValueError(f"{path.name}: every cell must have a BGRI2021 ID")
    bgri["BGRI2021"] = bgri["BGRI2021"].astype(str).str.strip()
    if bgri["BGRI2021"].eq("").any() or bgri["BGRI2021"].duplicated().any():
        raise ValueError(f"{path.name}: BGRI2021 IDs must be nonempty and unique")
    if (bgri.geometry.isna() | bgri.geometry.is_empty).any():
        raise ValueError(f"{path.name}: every cell must have a nonempty geometry")
    if not bgri.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError(f"{path.name}: expected Polygon or MultiPolygon cells")
    bgri["id"] = bgri["BGRI2021"]
    bgri["region"] = region
    # The buffer dataset has administrative codes rather than parish names.
    if "Municipality" not in bgri:
        bgri["Municipality"] = pd.NA
    if "Dicofre" not in bgri:
        bgri["Dicofre"] = bgri["DTMNFR21"] if "DTMNFR21" in bgri else pd.NA
    if "N_INDIVIDUOS" not in bgri:
        bgri["N_INDIVIDUOS"] = pd.NA
    bgri["geometry"] = bgri.geometry.representative_point()
    if (bgri.geometry.isna() | bgri.geometry.is_empty).any():
        raise ValueError(f"{path.name}: could not create a routing point for every cell")
    keep_cols = [
        "id", "BGRI2021", "region", "Municipality", "Dicofre", "N_INDIVIDUOS", "geometry",
    ]
    print(f"  {region}: {len(bgri)} BGRI routing points from {path.name}")
    return bgri[keep_cols].to_crs(CRS_WGS).sort_values("id").reset_index(drop=True)


def load_bgri_datasets() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load AML (including Lisbon) and the supplied external buffer cells."""
    aml = load_bgri_centroids(BGRI_PATH, "AML")
    buffer = load_bgri_centroids(BGRI_BUFFER_PATH, "buffer")
    overlap = set(aml["id"]) & set(buffer["id"])
    if overlap:
        raise ValueError(f"AML and buffer datasets share {len(overlap)} BGRI2021 IDs")
    return aml, buffer


MAX_TRAVEL_TIME = datetime.timedelta(minutes=120)
MAX_WALKING_TIME = datetime.timedelta(minutes=45)


def file_signature(path: Path) -> dict:
    """Hash inputs once per run so replaced feeds invalidate old batches."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def point_signature(points: gpd.GeoDataFrame) -> str:
    digest = hashlib.sha256()
    for cell_id, geometry in zip(points["id"], points.geometry):
        digest.update(json.dumps([str(cell_id), geometry.wkb_hex]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_matrix_output(
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    out_prefix: str,
    inputs: dict,
) -> tuple[Path, int]:
    """Validate checkpoints before routing; bound the output size per call."""
    if ORIGIN_BATCH_SIZE <= 0 or MAX_MATRIX_PAIRS_PER_BATCH <= 0:
        raise ValueError("Batch limits must be positive integers")
    if origins.empty or destinations.empty:
        raise ValueError(f"{out_prefix}: origins and destinations must be nonempty")
    if len(destinations) > MAX_MATRIX_PAIRS_PER_BATCH:
        raise ValueError("MAX_MATRIX_PAIRS_PER_BATCH must allow at least one origin")
    batch_size = min(ORIGIN_BATCH_SIZE, MAX_MATRIX_PAIRS_PER_BATCH // len(destinations))
    metadata = {
        "schema_version": 1,
        "r5py_version": r5py.__version__,
        "departure": DEPARTURE_DATETIME.isoformat(),
        "departure_window_seconds": DEPARTURE_WINDOW.total_seconds(),
        "max_time_seconds": MAX_TRAVEL_TIME.total_seconds(),
        "max_walking_seconds": MAX_WALKING_TIME.total_seconds(),
        "transport_modes": ["TRANSIT", "WALK"],
        "percentiles": [50],
        "origin_count": len(origins),
        "destination_count": len(destinations),
        "origin_points_sha256": point_signature(origins),
        "destination_points_sha256": point_signature(destinations),
        "origin_batch_size": batch_size,
        "inputs": inputs,
    }
    MATRICES.mkdir(parents=True, exist_ok=True)
    manifest_path = MATRICES / f"{out_prefix}_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != metadata:
            raise ValueError(
                f"{out_prefix}: inputs or routing settings differ from {manifest_path}. "
                "Use a fresh MATRICES directory or move the old batches and manifest elsewhere."
            )
    elif any(MATRICES.glob(f"{out_prefix}_batch_*.parquet")):
        raise ValueError(
            f"{out_prefix}: existing batches have no manifest. "
            "Use a fresh MATRICES directory or move the old batches elsewhere."
        )
    else:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(manifest_path)
    return manifest_path, batch_size


def run_matrix_batched(
    network: r5py.TransportNetwork,
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    out_prefix: str,
    batch_size: int,
) -> None:
    """Write one origin batch at a time and resume completed files without reading them."""
    n_batches = (len(origins) + batch_size - 1) // batch_size
    print(f"  {out_prefix}: {len(origins):,} x {len(destinations):,} pairs; "
          f"{n_batches} batches, up to {batch_size} origins each")
    for i in range(n_batches):
        out_path = MATRICES / f"{out_prefix}_batch_{i:03d}.parquet"
        if out_path.exists():
            print(f"    batch {i+1}/{n_batches}: already saved, skipping")
            continue
        batch = origins.iloc[i * batch_size:(i + 1) * batch_size]
        print(f"    batch {i+1}/{n_batches}: {len(batch)} origins x {len(destinations)} destinations …")
        ttm = r5py.TravelTimeMatrix(
            network,
            origins=batch[["id", "geometry"]],
            destinations=destinations[["id", "geometry"]],
            departure=DEPARTURE_DATETIME,
            departure_time_window=DEPARTURE_WINDOW,
            percentiles=[50],
            transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
            max_time=MAX_TRAVEL_TIME,
            max_time_walking=MAX_WALKING_TIME,
        )
        # Only the completed filename is a checkpoint. An interrupted write is
        # overwritten on restart rather than mistaken for a completed batch.
        temporary = out_path.with_suffix(".parquet.tmp")
        pd.DataFrame(ttm).to_parquet(temporary, index=False)
        temporary.replace(out_path)
        print(f"    Saved {len(ttm):,} rows -> {out_path.name}")
        del ttm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", choices=["aml-aml", "aml-buffer", "both"], default="both",
        help="Calculation to run (default: both, sequentially).",
    )
    return parser.parse_args(argv)


def main(matrix: str = "both") -> None:
    if matrix not in {"aml-aml", "aml-buffer", "both"}:
        raise ValueError(f"Unknown matrix selection: {matrix}")
    print("=" * 78)
    print(f"03  Batched travel-time matrices: {matrix}")
    print("=" * 78)
    print(f"  Departure: {DEPARTURE_DATETIME}; window: {DEPARTURE_WINDOW}")

    print("\n[1] Loading AML and buffer BGRI routing points (Lisbon included) …")
    if matrix == "aml-aml":
        aml = load_bgri_centroids(BGRI_PATH, "AML")
        buffer = None
    else:
        aml, buffer = load_bgri_datasets()
    jobs = []
    if matrix in {"aml-aml", "both"}:
        jobs.append(("AML -> AML", aml, "aml_matrix"))
    if matrix in {"aml-buffer", "both"}:
        jobs.append(("AML -> buffer", buffer, "aml_to_buffer"))

    print("\n[2] Checking inputs and output manifests …")
    inputs = {
        "osm": file_signature(OSM_PBF_AML_PATH),
        "gtfs": [file_signature(GTFS_ZIPS / name) for name in FEEDS.values()],
    }
    # Validate only selected outputs before any expensive network build/routing.
    prepared_jobs = []
    for label, destinations, prefix in jobs:
        _, batch_size = prepare_matrix_output(aml, destinations, prefix, inputs)
        prepared_jobs.append((label, destinations, prefix, batch_size))

    print("\n[3] Building TransportNetwork …")
    network = build_network()

    for step, (label, destinations, prefix, batch_size) in enumerate(prepared_jobs, start=4):
        print(f"\n[{step}] Computing {label} …")
        run_matrix_batched(network, aml, destinations, prefix, batch_size)

    print(f"\nDone. Separate Parquet batches saved under {MATRICES}")


if __name__ == "__main__":
    main(parse_args().matrix)
