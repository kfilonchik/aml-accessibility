"""
00_prepare_osm.py

Fetch the Portugal OSM extract (Geofabrik) and clip it to the AML
(Área Metropolitana de Lisboa) bounding box, producing the .osm.pbf
r5py needs for its TransportNetwork's street/walk layer. osmnx (used by
02_calculate_walking_network.py) downloads a routable graph directly and
can't be fed into r5py — r5py requires a PBF file on disk.

Run once; both downloaded/clipped files are cached and reused on rerun.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request

import geopandas as gpd

from config import (
    CRS_PT, OSM_BBOX_BUFFER_M, OSM_PBF_AML_PATH, OSM_PBF_FULL_PATH,
    OSM_PBF_URL_PT, PARISHES_PATH,
)


def download_full_extract() -> None:
    if OSM_PBF_FULL_PATH.exists():
        print(f"  Already present: {OSM_PBF_FULL_PATH.name} "
              f"({OSM_PBF_FULL_PATH.stat().st_size / 1e6:.0f} MB) — skipping download")
        return
    print(f"  Downloading {OSM_PBF_URL_PT} -> {OSM_PBF_FULL_PATH} …")
    urllib.request.urlretrieve(OSM_PBF_URL_PT, OSM_PBF_FULL_PATH)
    print(f"  Downloaded {OSM_PBF_FULL_PATH.stat().st_size / 1e6:.0f} MB")


def compute_aml_bbox_wgs84() -> tuple[float, float, float, float]:
    parishes = gpd.read_file(PARISHES_PATH)
    bbox_buffered = (
        parishes.to_crs(CRS_PT).total_bounds
    )
    minx, miny, maxx, maxy = bbox_buffered
    buffered = gpd.GeoSeries(
        gpd.points_from_xy([minx, maxx], [miny, maxy]), crs=CRS_PT
    ).buffer(OSM_BBOX_BUFFER_M)
    minx2, miny2, maxx2, maxy2 = buffered.total_bounds
    bbox_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([minx2, maxx2], [miny2, maxy2]), crs=CRS_PT
    ).to_crs("EPSG:4326")
    lon_min, lat_min, lon_max, lat_max = bbox_gdf.total_bounds
    print(f"  AML bbox (WGS84, +{OSM_BBOX_BUFFER_M}m buffer): "
          f"[{lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}]")
    return lon_min, lat_min, lon_max, lat_max


def clip_to_aml(bbox_wgs84: tuple[float, float, float, float]) -> None:
    if OSM_PBF_AML_PATH.exists():
        print(f"  Already present: {OSM_PBF_AML_PATH.name} "
              f"({OSM_PBF_AML_PATH.stat().st_size / 1e6:.0f} MB) — skipping clip")
        return
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    bbox_str = f"{lon_min},{lat_min},{lon_max},{lat_max}"
    cmd = [
        "osmium", "extract", "--bbox", bbox_str,
        "--overwrite", "-o", str(OSM_PBF_AML_PATH), str(OSM_PBF_FULL_PATH),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit(f"osmium extract failed (exit {result.returncode})")
    print(f"  Clipped to {OSM_PBF_AML_PATH.stat().st_size / 1e6:.0f} MB")


def main() -> None:
    print("=" * 78)
    print("00  Prepare OSM PBF extract for r5py (AML)")
    print("=" * 78)
    print("\n[1] Downloading full Portugal extract …")
    download_full_extract()
    print("\n[2] Computing AML bounding box from parish geometries …")
    bbox = compute_aml_bbox_wgs84()
    print("\n[3] Clipping to AML bbox with osmium-tool …")
    clip_to_aml(bbox)
    print("\nDone.")


if __name__ == "__main__":
    main()
