"""
05_webmap_data.py

Assembles the parish-level dataset behind the webmap
(output/webmap/aml_transit_atlas.html): joins the travel-sustainability
index (01), mode share (01), accessibility metric (04) and an added
travel-time-per-parish figure into a single parish.geojson, plus a
municipality boundary/label layer dissolved from the same source.

Travel time per parish: minutes needed to reach a given share (10% / 25% /
50%) of the combined AML + 75 km buffer population, summing destination
population (AML -> AML and AML -> buffer matrices from script 03) in ascending
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

Read batch-by-batch like 04_accessibility_metric.py, so the full matrix is
never held in memory.

Finally publishes the page itself: copies templates/aml_transit_atlas.html
(which already contains the ABOUT / DATA / GitHub buttons) and the parish
CSV (the DATA download) next to the GeoJSON layers.

Output (all under output/webmap/):
  parish.geojson
  municipalities.geojson
  municipality_labels.geojson
  aml_transit_atlas.html
  AML_accessibility_parishes.csv
Expected to be placed there by hand (linked by the ABOUT button):
  AML_accessibility_methodological_report.pdf
"""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

import geopandas as gpd
import numpy as np
import pandas as pd

from config import BGRI_BUFFER_PATH, BGRI_PATH, BASE, DATA, MATRICES

WEBMAP_DIR = BASE / "output" / "webmap"
WEBMAP_DIR.mkdir(parents=True, exist_ok=True)

STATS_DIR = DATA / "ine_statistics"
PARISH_CONVERSION_PATH = DATA / "polygons" / "dtmnfr21_to_2024_parish_name_conversion.csv"
PARISH_2024_GEOM_PATH = STATS_DIR / "IndicatorsFreguesia.geojson"

HTML_TEMPLATE = Path(__file__).parent / "templates" / "aml_transit_atlas.html"
CSV_NAME = "AML_accessibility_parishes.csv"
REPORT_NAME = "AML_accessibility_methodological_report.pdf"

POP_THRESHOLDS = [0.10, 0.25, 0.50]  # fractions of total AML + buffer population


def load_population(path: Path) -> pd.Series:
    """Population per BGRI cell (same ID handling as script 04)."""
    bgri = gpd.read_file(path, columns=["BGRI2021", "N_INDIVIDUOS"], ignore_geometry=True)
    pop = bgri.set_index(bgri["BGRI2021"].str.strip())["N_INDIVIDUOS"].astype(float)
    if pop.isna().any():
        raise ValueError(f"{path.name}: missing population values")
    print(f"  {path.name}: {len(pop):,} BGRI cells, population {pop.sum():,.0f}")
    return pop


def matrix_batches(prefix: str, n_origins: int, n_destinations: int) -> list[Path]:
    """All batch files of a finished matrix; an unfinished one must not be read as unreachable."""
    manifest = json.loads((MATRICES / f"{prefix}_manifest.json").read_text())
    if (manifest["origin_count"], manifest["destination_count"]) != (n_origins, n_destinations):
        raise ValueError(f"{prefix}: population dataset sizes do not match the matrix manifest")
    size = manifest["origin_batch_size"]
    paths = [MATRICES / f"{prefix}_batch_{i:03d}.parquet" for i in range((n_origins + size - 1) // size)]
    if set(MATRICES.glob(f"{prefix}_batch_*.parquet")) != set(paths):
        raise ValueError(f"{prefix}: missing or unexpected batch files; finish script 03 first")
    return paths


def compute_time_to_population(aml_pop: pd.Series, buffer_pop: pd.Series) -> pd.DataFrame:
    """For each AML BGRI origin, minutes needed to reach POP_THRESHOLDS shares
    of the total AML + 75 km buffer population, summing destination
    population (AML and buffer cells, including the origin's own cell at
    travel_time=0) in ascending travel-time order. NaN where even all
    reachable destinations together (<=120 min cap) don't add up to that
    share -- i.e. "more than 120 min to reach that much population from here".

    The AML -> AML and AML -> buffer matrices are batched differently, so
    each batch is reduced to population reached per (origin, minute); those
    small tables are summed across both matrices before the cumulative sum.
    """
    total_pop = aml_pop.sum() + buffer_pop.sum()
    thresholds = {f"time_to_pop_{int(f*100):02d}": f * total_pop for f in POP_THRESHOLDS}

    per_minute = []
    for prefix, population in [("aml_matrix", aml_pop), ("aml_to_buffer", buffer_pop)]:
        paths = matrix_batches(prefix, len(aml_pop), len(population))
        print(f"  Processing {prefix}: {len(paths)} batches …")
        for i, path in enumerate(paths, start=1):
            df = pd.read_parquet(path, columns=["from_id", "to_id", "travel_time"])
            df = df.dropna(subset=["travel_time"])
            pop_j = df["to_id"].map(population)
            if pop_j.isna().any() or not df["from_id"].isin(aml_pop.index).all():
                raise ValueError(f"{path.name}: unknown origin or destination IDs")
            per_minute.append(pop_j.groupby([df["from_id"], df["travel_time"]]).sum())
            if i % 100 == 0 or i == len(paths):
                print(f"    {i}/{len(paths)} batches done")

    reached = pd.concat(per_minute).groupby(level=[0, 1]).sum().sort_index()
    cum_pop = reached.groupby(level=0).cumsum()
    out = {}
    for key, threshold in thresholds.items():
        hit = cum_pop[cum_pop >= threshold].reset_index(level=1)
        out[key] = hit.groupby(level=0)["travel_time"].first()
    return pd.DataFrame(out).reindex(aml_pop.index)


def main() -> None:
    print("=" * 78)
    print("05  Assemble webmap parish dataset")
    print("=" * 78)

    print("\n[1] Computing time-to-population-share per BGRI cell …")
    bgri = gpd.read_file(BGRI_PATH).rename(columns={"BGRI2021": "id"})
    bgri["id"] = bgri["id"].str.strip()
    time_to_pop = compute_time_to_population(load_population(BGRI_PATH), load_population(BGRI_BUFFER_PATH))
    tt_cols = list(time_to_pop.columns)

    for col in tt_cols:
        bgri[col] = bgri["id"].map(time_to_pop[col])
        n_unreached = bgri[col].isna().sum()
        print(f"    {col}: {n_unreached} / {len(bgri)} BGRI cells never reach that share within 120 min")

    print("\n[2] Rolling up travel time to parish (population-weighted mean, 2016→2024 parish conversion) …")
    parish_src = bgri.dropna(subset=["Dicofre"]).copy()
    parish_src["Dicofre"] = parish_src["Dicofre"].astype(str).str.zfill(6)

    # Same 2016 (dtmnfr16) -> 2024 (dtmnfr24) conversion as 04_accessibility_metric.py:
    # weight=1 for the 114 non-split parishes, fractional (summing to 1 per
    # old code) for the 4 that split into multiple 2024 parishes. Applied to
    # both the numerator and denominator of the population-weighted mean.
    conversion = pd.read_csv(PARISH_CONVERSION_PATH, dtype=str)
    conversion["weight"] = conversion["weight"].astype(float)
    conversion = conversion.rename(columns={"dtmnfr16": "Dicofre"})

    parish_src = parish_src.merge(
        conversion[["Dicofre", "dtmnfr24", "weight"]], on="Dicofre", how="left",
    )
    n_unmatched = parish_src["dtmnfr24"].isna().sum()
    if n_unmatched:
        print(f"    WARNING: {n_unmatched} BGRI rows have a Dicofre not found in the 2016->2024 "
              f"conversion table -- dropping them from the rollup")
        parish_src = parish_src.dropna(subset=["dtmnfr24"])

    tt_by_parish_parts = {}
    for col in tt_cols:
        valid = parish_src.dropna(subset=[col])  # exclude from both numerator AND denominator
        pop_w = valid["N_INDIVIDUOS"] * valid["weight"]
        numerator = (valid[col] * pop_w).groupby(valid["dtmnfr24"]).sum()
        denominator = pop_w.groupby(valid["dtmnfr24"]).sum()
        tt_by_parish_parts[col] = numerator / denominator.replace(0, np.nan)
    tt_by_parish = pd.DataFrame(tt_by_parish_parts)
    tt_by_parish.index.name = "Dicofre"

    print("\n[3] Joining sustainability, mode share, accessibility and travel time …")
    access = pd.read_csv(STATS_DIR / "AML_accessibility_parishes.csv", dtype={"Dicofre": str})
    for col in tt_cols:
        access[col] = access["Dicofre"].map(tt_by_parish[col])

    # Geometry: the real 2024 parish boundary layer (same one used by
    # 04_accessibility_metric.py), not BGRI polygons dissolved by the old
    # Dicofre code -- conversion weights are for statistic redistribution
    # only and must never be used to approximate geometry.
    parishes_2024 = gpd.read_file(PARISH_2024_GEOM_PATH)
    parishes_2024["Dicofre"] = parishes_2024["id"].astype(str).str.zfill(6)
    assert parishes_2024["Dicofre"].duplicated().sum() == 0, "duplicated Dicofre in 2024 geometry layer"
    assert access["Dicofre"].duplicated().sum() == 0, "duplicated Dicofre in accessibility/statistics table"
    unmatched_stats = set(access["Dicofre"]) - set(parishes_2024["Dicofre"])
    print(f"    Statistics codes with no match in 2024 geometry layer: {len(unmatched_stats)} "
          f"{sorted(unmatched_stats) if unmatched_stats else ''}")

    parish = parishes_2024[["Dicofre", "geometry"]].merge(access, on="Dicofre", how="left")
    assert len(parish) == len(parishes_2024), "row count changed across merge -- Dicofre not unique somewhere"
    n_no_access = parish["access"].isna().sum()
    print(f"    2024 geometry parishes with no accessibility value after join: {n_no_access}")

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

    print("\n[5] Publishing the atlas page and CSV download …")
    copyfile(HTML_TEMPLATE, WEBMAP_DIR / "aml_transit_atlas.html")
    copyfile(STATS_DIR / CSV_NAME, WEBMAP_DIR / CSV_NAME)
    print(f"  Saved {WEBMAP_DIR / 'aml_transit_atlas.html'}")
    print(f"  Saved {WEBMAP_DIR / CSV_NAME}")
    if not (WEBMAP_DIR / REPORT_NAME).is_file():
        print(f"  WARNING: {REPORT_NAME} not found in {WEBMAP_DIR} -- the ABOUT button will be a dead link")

    print("\nDone.")


if __name__ == "__main__":
    main()
