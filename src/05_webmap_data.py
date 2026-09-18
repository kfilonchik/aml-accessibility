"""
05_webmap_data.py

Assembles the parish-level dataset behind the webmap
(output/webmap/aml_transit_atlas.html): joins the travel-sustainability
index (01), mode share (01), accessibility metric (04) and an added
travel-time-per-parish figure into a single parish.geojson, plus a
municipality boundary/label layer dissolved from the same source.

Travel time per parish: minutes needed to reach a given share of the AML's
population (10% / 25% / 50%), summing destination population in ascending
travel-time order until the threshold is crossed -- a standard cumulative-
accessibility statistic (same family as "jobs reachable within 30/60 min").

A plain unweighted mean travel time to every other BGRI cell was tried
first and rejected: AML covers ~3,000 km^2, so the average distance between
two random points in it is already ~25-30km, which is why that mean landed
around 65-100 min for every parish -- technically correct but a poor proxy
for "typical travel time", since it's dominated by far-flung cross-region
pairs rather than by where people actually live. The population-threshold
statistic answers a more meaningful question and stays directly comparable
to the accessibility metric (04), which is also population-weighted.

Read batch-by-batch like 04_accessibility_metric.py, since each batch
already contains a complete row-block for its origins.

Output (all under output/webmap/):
  parish.geojson
  municipalities.geojson
  municipality_labels.geojson
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from config import BGRI_PATH, BASE, MATRICES

WEBMAP_DIR = BASE / "output" / "webmap"
WEBMAP_DIR.mkdir(parents=True, exist_ok=True)

STATS_DIR = BASE / "data" / "statistics"
DATA = BASE / "data"

POP_THRESHOLDS = [0.10, 0.25, 0.50]  # fractions of total AML population


def compute_time_to_population(pop_by_id: pd.Series) -> pd.DataFrame:
    """For each BGRI origin, minutes needed to reach POP_THRESHOLDS shares of
    the AML's total population, summing destination population (including
    the origin's own cell, at travel_time=0) in ascending travel-time order.
    NaN where even all reachable destinations together (<=120 min cap) don't
    add up to that share -- i.e. "more than 120 min to reach that much
    population from here"."""
    total_pop = pop_by_id.sum()
    thresholds = {f"time_to_pop_{int(f*100):02d}": f * total_pop for f in POP_THRESHOLDS}

    batch_paths = sorted(MATRICES.glob("aml_matrix_batch_*.parquet"))
    print(f"  Processing {len(batch_paths)} batch files for time-to-population-share …")
    parts = {key: [] for key in thresholds}
    for i, path in enumerate(batch_paths):
        df = pd.read_parquet(path, columns=["from_id", "to_id", "travel_time"])
        df = df.dropna(subset=["travel_time"])
        df["pop_j"] = df["to_id"].map(pop_by_id).fillna(0.0)
        df = df.sort_values(["from_id", "travel_time"])
        df["cum_pop"] = df.groupby("from_id")["pop_j"].cumsum()
        for key, threshold in thresholds.items():
            reached = df[df["cum_pop"] >= threshold].groupby("from_id").first()["travel_time"]
            parts[key].append(reached)
        print(f"    batch {i+1}/{len(batch_paths)}: {path.name} done")

    out = pd.DataFrame({key: pd.concat(series_list) for key, series_list in parts.items()})
    assert out.index.duplicated().sum() == 0
    return out


def main() -> None:
    print("=" * 78)
    print("05  Assemble webmap parish dataset")
    print("=" * 78)

    print("\n[1] Computing time-to-population-share per BGRI cell …")
    bgri = gpd.read_file(BGRI_PATH).rename(columns={"BGRI2021": "id"})
    pop_by_id = bgri.set_index("id")["N_INDIVIDUOS"].astype(float)
    time_to_pop = compute_time_to_population(pop_by_id)
    tt_cols = list(time_to_pop.columns)

    for col in tt_cols:
        bgri[col] = bgri["id"].map(time_to_pop[col])
        n_unreached = bgri[col].isna().sum()
        print(f"    {col}: {n_unreached} / {len(bgri)} BGRI cells never reach that share within 120 min")

    print("\n[2] Rolling up travel time to parish (population-weighted mean) …")
    parish_src = bgri.dropna(subset=["Dicofre"]).copy()
    parish_src["Dicofre"] = parish_src["Dicofre"].astype(str).str.zfill(6)
    tt_by_parish_parts = {}
    for col in tt_cols:
        valid = parish_src.dropna(subset=[col])  # exclude from both numerator AND denominator
        g = valid.assign(weighted=valid[col] * valid["N_INDIVIDUOS"]).groupby("Dicofre")
        tt_by_parish_parts[col] = g["weighted"].sum() / g["N_INDIVIDUOS"].sum().replace(0, np.nan)
    tt_by_parish = pd.DataFrame(tt_by_parish_parts)

    print("\n[3] Joining sustainability, mode share, accessibility and travel time …")
    access = pd.read_csv(STATS_DIR / "AML_accessibility_parishes.csv", dtype={"Dicofre": str})
    for col in tt_cols:
        access[col] = access["Dicofre"].map(tt_by_parish[col])

    parish_geom = (
        bgri.dropna(subset=["Dicofre"])
        .assign(Dicofre=lambda d: d["Dicofre"].astype(str).str.zfill(6))[["Dicofre", "geometry"]]
        .dissolve("Dicofre")
        .reset_index()
    )
    parish = parish_geom.merge(access, on="Dicofre", how="inner")
    centroid_pt = parish.geometry.representative_point().to_crs("EPSG:4326")
    parish["lon"] = centroid_pt.x
    parish["lat"] = centroid_pt.y
    parish = parish.to_crs("EPSG:4326")
    parish["geometry"] = parish.geometry.simplify(0.0003, preserve_topology=True)
    print(f"    {len(parish)} parishes assembled")

    parish.to_file(WEBMAP_DIR / "parish.geojson", driver="GeoJSON")
    print(f"\n  Saved {WEBMAP_DIR / 'parish.geojson'}")

    print("\n[4] Building municipality boundary + label layers …")
    muni_src = bgri.to_crs("EPSG:3763")
    municipalities = muni_src[["Municipality", "geometry"]].dissolve("Municipality").reset_index()
    municipalities = municipalities.to_crs("EPSG:4326")
    municipalities["geometry"] = municipalities.geometry.simplify(0.0005, preserve_topology=True)
    municipalities.to_file(WEBMAP_DIR / "municipalities.geojson", driver="GeoJSON")

    labels = municipalities.copy()
    labels["geometry"] = labels.to_crs("EPSG:3763").geometry.representative_point().to_crs("EPSG:4326")
    labels[["Municipality", "geometry"]].to_file(WEBMAP_DIR / "municipality_labels.geojson", driver="GeoJSON")
    print(f"  Saved {WEBMAP_DIR / 'municipalities.geojson'}")
    print(f"  Saved {WEBMAP_DIR / 'municipality_labels.geojson'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
