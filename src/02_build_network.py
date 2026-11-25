"""
02_build_network.py

Prepares the OSM street extract that 03_travel_time_matrices.py needs:
downloads the Portugal extract (Geofabrik) and clips it to the AML bounding
box plus config.OSM_BBOX_BUFFER_M, writing config.OSM_PBF_AML_PATH.

Files that already exist are reused, so rerunning is safe. The routable
network itself is built (and cached) by r5py in script 03.

Clipping needs the `osmium` command:
  conda install -c conda-forge osmium-tool
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request

import geopandas as gpd

from config import (
    CRS_PT, CRS_WGS, OSM_BBOX_BUFFER_M, OSM_PBF_AML_PATH, OSM_PBF_FULL_PATH,
    OSM_PBF_URL_PT, PARISHES_PATH,
)


def download_full_extract() -> None:
    if OSM_PBF_FULL_PATH.exists():
        print(f"  Already present: {OSM_PBF_FULL_PATH.name} "
              f"({OSM_PBF_FULL_PATH.stat().st_size / 1e6:.0f} MB) — skipping download")
        return
    print(f"  Downloading {OSM_PBF_URL_PT} -> {OSM_PBF_FULL_PATH} …")
    # Rename only when complete, so an interrupted download is never reused.
    temporary = OSM_PBF_FULL_PATH.with_name(f"tmp_{OSM_PBF_FULL_PATH.name}")
    urllib.request.urlretrieve(OSM_PBF_URL_PT, temporary)
    temporary.replace(OSM_PBF_FULL_PATH)
    print(f"  Downloaded {OSM_PBF_FULL_PATH.stat().st_size / 1e6:.0f} MB")


def compute_aml_bbox_wgs84() -> tuple[float, float, float, float]:
    parishes = gpd.read_file(PARISHES_PATH)
    minx, miny, maxx, maxy = parishes.to_crs(CRS_PT).total_bounds
    buffered = gpd.GeoSeries(
        gpd.points_from_xy([minx, maxx], [miny, maxy]), crs=CRS_PT
    ).buffer(OSM_BBOX_BUFFER_M)
    minx2, miny2, maxx2, maxy2 = buffered.total_bounds
    bbox_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([minx2, maxx2], [miny2, maxy2]), crs=CRS_PT
    ).to_crs(CRS_WGS)
    lon_min, lat_min, lon_max, lat_max = bbox_gdf.total_bounds
    print(f"  AML bbox (WGS84, +{OSM_BBOX_BUFFER_M}m buffer): "
          f"[{lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}]")
    return lon_min, lat_min, lon_max, lat_max


def clip_to_aml(bbox_wgs84: tuple[float, float, float, float]) -> None:
    bbox_str = ",".join(str(value) for value in bbox_wgs84)
    # The temporary name must keep the .osm.pbf suffix: osmium infers the
    # output format from it.
    temporary = OSM_PBF_AML_PATH.with_name(f"tmp_{OSM_PBF_AML_PATH.name}")
    cmd = [
        "osmium", "extract", "--bbox", bbox_str,
        "--overwrite", "-o", str(temporary), str(OSM_PBF_FULL_PATH),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit(f"osmium extract failed (exit {result.returncode})")
    temporary.replace(OSM_PBF_AML_PATH)
    print(f"  Clipped to {OSM_PBF_AML_PATH.stat().st_size / 1e6:.0f} MB")


def main() -> None:
    print("=" * 78)
    print("02  Prepare OSM PBF extract for r5py (AML + buffer)")
    print("=" * 78)

    if OSM_PBF_AML_PATH.exists():
        print(f"  Already present: {OSM_PBF_AML_PATH} "
              f"({OSM_PBF_AML_PATH.stat().st_size / 1e6:.0f} MB) — nothing to do")
        return
    if shutil.which("osmium") is None:
        sys.exit("The 'osmium' command was not found. Install it with: "
                 "conda install -c conda-forge osmium-tool")

    print("\n[1] Downloading full Portugal extract …")
    download_full_extract()
    print("\n[2] Computing AML bounding box from parish geometries …")
    bbox = compute_aml_bbox_wgs84()
    print("\n[3] Clipping to AML bbox with osmium-tool …")
    clip_to_aml(bbox)
    print("\nDone.")


if __name__ == "__main__":
    main()
