"""
04_accessibility_metric.py

Simple gravity-style public-transport accessibility metric per BGRI cell
(and rolled up to parish), following D.A. Smith's "simple accessibility to
population" approach:

    Access_i = sum_j  Population_j * S(travel_time_ij)

where S is a sigmoid impedance function converting travel time into a
0-1 weight (nearby opportunities count near-fully, distant ones decay
toward zero), and the sum runs over every BGRI cell j (including i itself,
at travel_time=0 -> S=1, i.e. a cell's own population counts in full).

Impedance function: S(x) = 1 / (1 + (x / x0)^k), the DfT Transport
Connectivity Metric's calibrated "PT Overall" curve (x0=32.50, k=2.81),
fitted on UK National Travel Survey travel-diary data. There is no
Portuguese/AML travel-diary microdata in this repo to fit a local curve
against, so these published parameters are used as-is (confirmed with the
user) rather than a locally "calibrated" fit. A single curve is used
(rather than separate per-mode curves) because r5py's TravelTimeMatrix
returns one blended door-to-door TRANSIT+WALK time per OD pair and does
not expose which vehicle mode(s) made up a given trip -- ferry (boat_2)
legs are not treated separately for the same reason; they're already
folded into the same blended travel_time as every other PT leg.

Travel time source: output/travel_matrix/aml_matrix_batch_*.parquet (see
03_travel_time_matrices.py) -- read batch-by-batch rather than the merged
671M-row aml_matrix.parquet, since each batch already contains a complete
row-block (its origins x all 25,912 destinations), so per-origin sums can
be computed one batch at a time without holding the full matrix in memory.

Output:
  data/AML_accessibility_bgri.geojson       (BGRI-cell level)
  data/statistics/AML_accessibility_bgri.csv
  data/AML_accessibility_parishes.geojson   (parish rollup, joined with TSI)
  data/statistics/AML_accessibility_parishes.csv
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from config import BGRI_PATH, DATA, MATRICES

STATS_DIR = DATA / "statistics"

# DfT Transport Connectivity Metric, "PT Overall" curve (NTS 2011-2023):
# S(x) = 1 / (1 + (x / IMPEDANCE_X0) ** IMPEDANCE_K)
IMPEDANCE_X0 = 32.50
IMPEDANCE_K  = 2.81


def impedance(travel_time_min: np.ndarray) -> np.ndarray:
    """Sigmoid decay weight in [0, 1]; NaN (unreachable within the matrix's
    120-min cap) maps to 0 weight, same as an infinitely long trip would."""
    weight = 1.0 / (1.0 + (travel_time_min / IMPEDANCE_X0) ** IMPEDANCE_K)
    return np.where(np.isnan(travel_time_min), 0.0, weight)


def load_population() -> pd.Series:
    bgri = gpd.read_file(BGRI_PATH)[["BGRI2021", "N_INDIVIDUOS"]]
    pop = bgri.set_index("BGRI2021")["N_INDIVIDUOS"].astype(float)
    print(f"  Population loaded for {len(pop)} BGRI cells, total {pop.sum():,.0f}")
    return pop


def compute_access(pop_by_id: pd.Series) -> pd.Series:
    batch_paths = sorted(MATRICES.glob("aml_matrix_batch_*.parquet"))
    print(f"  Processing {len(batch_paths)} batch files …")
    access_parts = []
    for i, path in enumerate(batch_paths):
        df = pd.read_parquet(path, columns=["from_id", "to_id", "travel_time"])
        df["pop_j"] = df["to_id"].map(pop_by_id).fillna(0.0)
        df["contrib"] = impedance(df["travel_time"].to_numpy()) * df["pop_j"]
        access_parts.append(df.groupby("from_id")["contrib"].sum())
        print(f"    batch {i+1}/{len(batch_paths)}: {path.name} done")
    access = pd.concat(access_parts)
    assert access.index.duplicated().sum() == 0, "origin appears in more than one batch"
    return access


def main() -> None:
    print("=" * 78)
    print("04  Simple PT accessibility-to-population metric (BGRI + parish)")
    print("=" * 78)

    print("\n[1] Loading population …")
    pop_by_id = load_population()
    total_pop = pop_by_id.sum()

    print("\n[2] Computing Access_i from travel-time batches …")
    access = compute_access(pop_by_id)

    print("\n[3] Assembling BGRI-level output …")
    bgri = gpd.read_file(BGRI_PATH)
    bgri = bgri.rename(columns={"BGRI2021": "id"})
    bgri["access_raw"] = bgri["id"].map(access)
    n_missing = bgri["access_raw"].isna().sum()
    if n_missing:
        print(f"    WARNING: {n_missing} BGRI cells missing from the travel-time matrix (not routed)")
    bgri["access_pct_pop"] = 100 * bgri["access_raw"] / total_pop

    stats = bgri["access_raw"].agg(["min", "mean", "median", "max"])
    print("\n  Access_raw summary statistics:")
    print(stats.to_string())

    bgri_cols = ["id", "Municipality", "Dicofre", "Freguesia", "N_INDIVIDUOS", "access_raw", "access_pct_pop"]
    bgri_out = bgri[bgri_cols + ["geometry"]].to_crs("EPSG:4326")
    bgri_out["geometry"] = bgri_out.geometry.simplify(0.0001, preserve_topology=True)
    bgri_out.to_file(DATA / "AML_accessibility_bgri.geojson", driver="GeoJSON")
    bgri[bgri_cols].to_csv(STATS_DIR / "AML_accessibility_bgri.csv", index=False)
    print(f"\n  Saved {DATA / 'AML_accessibility_bgri.geojson'}")
    print(f"  Saved {STATS_DIR / 'AML_accessibility_bgri.csv'}")

    print("\n[4] Rolling up to parish level (population-weighted mean) …")
    n_null_dicofre = bgri["Dicofre"].isna().sum()
    parish_src = bgri.dropna(subset=["Dicofre"]).copy()
    if n_null_dicofre:
        print(f"    Dropping {n_null_dicofre} BGRI rows with a null Dicofre before rollup")
    parish_src["Dicofre"] = parish_src["Dicofre"].astype(str).str.zfill(6)
    parish_src["weighted"] = parish_src["access_raw"] * parish_src["N_INDIVIDUOS"]

    grouped = parish_src.groupby("Dicofre").agg(
        N_INDIVIDUOS=("N_INDIVIDUOS", "sum"),
        weighted_sum=("weighted", "sum"),
        Municipality=("Municipality", "first"),
        Freguesia=("Freguesia", "first"),
    ).reset_index()
    grouped["access_raw"] = grouped["weighted_sum"] / grouped["N_INDIVIDUOS"].replace(0, np.nan)
    grouped["access_pct_pop"] = 100 * grouped["access_raw"] / total_pop
    grouped = grouped.drop(columns="weighted_sum")

    tsi = pd.read_csv(STATS_DIR / "AML_TSI_parishes.csv", dtype={"Dicofre": str})
    tsi_cols = ["Dicofre", "pct_active", "pct_pt", "pct_car", "pct_zero_car", "TSI"]
    parish = grouped.merge(tsi[tsi_cols], on="Dicofre", how="left")
    n_missing_tsi = parish["TSI"].isna().sum()
    print(f"    TSI missing for {n_missing_tsi} of {len(parish)} parishes after join")

    parish_geom = (
        bgri.dropna(subset=["Dicofre"])
        .assign(Dicofre=lambda d: d["Dicofre"].astype(str).str.zfill(6))[["Dicofre", "geometry"]]
        .dissolve("Dicofre")
        .reset_index()
    )
    parish_out = parish_geom.merge(parish, on="Dicofre", how="inner").to_crs("EPSG:4326")
    parish_out["geometry"] = parish_out.geometry.simplify(0.0003, preserve_topology=True)

    parish_stats = parish["access_raw"].agg(["min", "mean", "median", "max"])
    print("\n  Parish-level Access_raw summary statistics:")
    print(parish_stats.to_string())

    parish_cols = ["Dicofre", "Freguesia", "Municipality", "N_INDIVIDUOS",
                    "access_raw", "access_pct_pop", "pct_active", "pct_pt", "pct_car", "pct_zero_car", "TSI"]
    parish_out[parish_cols + ["geometry"]].to_file(DATA / "AML_accessibility_parishes.geojson", driver="GeoJSON")
    parish[["Dicofre", "Freguesia", "Municipality", "N_INDIVIDUOS", "access_raw", "access_pct_pop",
            "pct_active", "pct_pt", "pct_car", "pct_zero_car", "TSI"]].to_csv(
        STATS_DIR / "AML_accessibility_parishes.csv", index=False)
    print(f"\n  Saved {DATA / 'AML_accessibility_parishes.geojson'}")
    print(f"  Saved {STATS_DIR / 'AML_accessibility_parishes.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
