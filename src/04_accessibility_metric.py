"""
04_accessibility_metric.py

Simple gravity-style public-transport accessibility metric per BGRI cell
(and rolled up to parish), following D.A. Smith's "simple accessibility to
population" approach, normalized to a population-weighted average of the
impedance to every other BGRI from each origin:

    access_raw_i  = sum_{j != i}  Population_j * S(travel_time_ij)
    Population_other_i = sum_{j != i}  Population_j
    access_i      = access_raw_i / Population_other_i

where S is a sigmoid impedance function converting travel time into a 0-1
weight (nearby opportunities count near-fully, distant ones decay toward
zero). Every destination j != i is included in BOTH sums, whether or not
the travel-time matrix returned a finite time for that OD pair: a NaN
travel time (no route found within the matrix's 120-min cap) is treated as
an infinite travel time, so S(NaN) = 0 -- it contributes its population to
the denominator but zero weight to the numerator, correctly dragging
access_i down rather than being dropped from the average altogether.
Population_other_i is therefore the fixed AML-wide total minus the origin's
own population, the same for every origin. The origin itself (j = i,
travel_time = 0) is EXCLUDED from both sums: including it would let a
BGRI's own population dominate both numerator and denominator identically,
which for the least-connected BGRIs (where the origin has no real transit
reach at all) would inflate access_i instead of correctly pulling it toward
0 -- the metric is about reaching OTHER places, not the origin's own
residents. Population_j is always the destination BGRI's own population --
never a parish total. access_i is bounded in [0, 1] (a population-weighted
average impedance over every other BGRI, unreachable ones included at
weight 0); access_raw_i is kept alongside as an unnormalized diagnostic.
Parish aggregation (below) weights and averages access_i (not access_raw_i)
by each BGRI's own population, Population_i, as a share of the parish's
total population.

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

STATS_DIR = DATA / "ine_statistics"
PARISH_CONVERSION_PATH = DATA / "polygons" / "dtmnfr21_to_2024_parish_name_conversion.csv"
# Official 2024 parish boundary layer (124 freguesias, geometry keyed by "id"
# = dtmnfr24) -- NOT derived from BGRI polygons, so split parishes get their
# real 2024 boundaries rather than an approximation.
PARISH_2024_GEOM_PATH = STATS_DIR / "IndicatorsFreguesia.geojson"

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


def compute_access(pop_by_id: pd.Series) -> pd.DataFrame:
    """Per origin: access_raw (sum of impedance-weighted destination
    population, over EVERY destination j != i) and pop_other (sum of
    destination population over every j != i, regardless of whether that OD
    pair was reachable) -- the denominator for the normalized access_i,
    computed by the caller. A NaN travel_time (unreachable within the
    matrix's 120-min cap) is treated as an infinite travel time: it still
    counts its population in pop_other, but impedance() maps it to a 0
    contribution to access_raw, so it correctly pulls access_i down instead
    of being excluded from the average. Origin==destination pairs are
    excluded from both sums (see module docstring): the metric is about
    reaching OTHER BGRIs, not counting an origin's own population as part of
    its own accessibility."""
    batch_paths = sorted(MATRICES.glob("aml_matrix_batch_*.parquet"))
    print(f"  Processing {len(batch_paths)} batch files …")
    raw_parts = []
    other_pop_parts = []
    for i, path in enumerate(batch_paths):
        df = pd.read_parquet(path, columns=["from_id", "to_id", "travel_time"])
        df = df.loc[df["from_id"] != df["to_id"]]
        df["pop_j"] = df["to_id"].map(pop_by_id).fillna(0.0)
        df["contrib"] = impedance(df["travel_time"].to_numpy()) * df["pop_j"]
        raw_parts.append(df.groupby("from_id")["contrib"].sum())
        other_pop_parts.append(df.groupby("from_id")["pop_j"].sum())
        print(f"    batch {i+1}/{len(batch_paths)}: {path.name} done")
    access_raw = pd.concat(raw_parts)
    pop_other = pd.concat(other_pop_parts)
    assert access_raw.index.duplicated().sum() == 0, "origin appears in more than one batch"
    return pd.DataFrame({"access_raw": access_raw, "pop_other": pop_other})


def main() -> None:
    print("=" * 78)
    print("04  Simple PT accessibility-to-population metric (BGRI + parish)")
    print("=" * 78)

    print("\n[1] Loading population …")
    pop_by_id = load_population()

    print("\n[2] Computing Access_i from travel-time batches …")
    access = compute_access(pop_by_id)

    print("\n[3] Assembling BGRI-level output …")
    bgri = gpd.read_file(BGRI_PATH)
    bgri = bgri.rename(columns={"BGRI2021": "id"})
    bgri["access_raw"] = bgri["id"].map(access["access_raw"])
    bgri["pop_other"] = bgri["id"].map(access["pop_other"])
    n_missing = bgri["access_raw"].isna().sum()
    if n_missing:
        print(f"    WARNING: {n_missing} BGRI cells missing from the travel-time matrix (not routed)")
    # access_i = access_raw_i / Population_other_i -- a population-weighted
    # average impedance over every other BGRI (unreachable ones included at
    # weight 0, see module docstring). This, not access_raw, is what the
    # parish rollup below weights and averages.
    bgri["access"] = bgri["access_raw"] / bgri["pop_other"].replace(0, np.nan)
    bgri["access_pct"] = 100 * bgri["access"]

    stats = bgri[["access_raw", "access"]].agg(["min", "mean", "median", "max"])
    print("\n  access_raw_i and access_i (normalized) summary statistics:")
    print(stats.to_string())

    bgri_cols = ["id", "Municipality", "Dicofre", "Freguesia", "N_INDIVIDUOS",
                 "access_raw", "pop_other", "access", "access_pct"]
    bgri_out = bgri[bgri_cols + ["geometry"]].to_crs("EPSG:4326")
    bgri_out["geometry"] = bgri_out.geometry.simplify(0.0001, preserve_topology=True)
    bgri_out.to_file(DATA / "AML_accessibility_bgri.geojson", driver="GeoJSON")
    bgri[bgri_cols].to_csv(STATS_DIR / "AML_accessibility_bgri.csv", index=False)
    print(f"\n  Saved {DATA / 'AML_accessibility_bgri.geojson'}")
    print(f"  Saved {STATS_DIR / 'AML_accessibility_bgri.csv'}")

    print("\n[4] Rolling up to parish level (population-weighted mean, 2016→2024 parish conversion) …")
    n_null_dicofre = bgri["Dicofre"].isna().sum()
    parish_src = bgri.dropna(subset=["Dicofre"]).copy()
    if n_null_dicofre:
        print(f"    Dropping {n_null_dicofre} BGRI rows with a null Dicofre before rollup")
    parish_src["Dicofre"] = parish_src["Dicofre"].astype(str).str.zfill(6)

    # 2016 (dtmnfr16) -> 2024 (dtmnfr24) parish conversion. For most parishes
    # this is a 1:1 relabelling (weight=1); 4 parishes split across several
    # 2024 parishes, each with its own area-based weight (weights sum to 1
    # per dtmnfr16 code). Every BGRI's access/N_INDIVIDUOS gets fanned out
    # across its dtmnfr24 target(s) with that weight applied to BOTH the
    # numerator and denominator of the population-weighted mean, per the
    # user-specified formula:
    #   numerator term:   access_i * Pop_i * w
    #   denominator term: Pop_i * w
    conversion = pd.read_csv(PARISH_CONVERSION_PATH, dtype=str)
    conversion["weight"] = conversion["weight"].astype(float)
    conversion = conversion.rename(columns={"dtmnfr16": "Dicofre"})

    # --- Conversion-table validation -------------------------------------
    EXPECTED_SPLITS = {
        "111123": {"111129", "111131", "111132"},
        "111126": {"111130", "111133"},
        "111127": {"111134", "111135"},
        "151007": {"151008", "151009", "151010"},
    }
    n_unique_old = conversion["Dicofre"].nunique()
    print(f"    Conversion table: {len(conversion)} rows, {n_unique_old} unique dtmnfr16 codes")
    assert n_unique_old == 118, f"expected 118 unique dtmnfr16 codes, got {n_unique_old}"
    assert len(conversion) == 124, f"expected 124 conversion rows, got {len(conversion)}"

    targets_per_old = conversion.groupby("Dicofre")["dtmnfr24"].apply(set)
    split_old_codes = targets_per_old[targets_per_old.apply(len) > 1]
    print(f"    {len(split_old_codes)} dtmnfr16 codes split into multiple dtmnfr24 targets: "
          f"{sorted(split_old_codes.index)}")
    assert set(split_old_codes.index) == set(EXPECTED_SPLITS), \
        f"split codes {sorted(split_old_codes.index)} != expected {sorted(EXPECTED_SPLITS)}"
    for code, expected_targets in EXPECTED_SPLITS.items():
        actual_targets = split_old_codes[code]
        assert actual_targets == expected_targets, \
            f"{code}: targets {sorted(actual_targets)} != expected {sorted(expected_targets)}"

    weight_sums = conversion.groupby("Dicofre")["weight"].sum()
    assert np.allclose(weight_sums, 1.0), \
        f"weights don't sum to 1 for: {weight_sums[~np.isclose(weight_sums, 1.0)].to_dict()}"
    non_split = conversion[~conversion["Dicofre"].isin(EXPECTED_SPLITS)]
    assert (non_split["weight"] == 1.0).all(), \
        f"non-split parishes with weight != 1: {non_split.loc[non_split['weight'] != 1.0, 'Dicofre'].tolist()}"
    print("    Conversion table validated: 118 old codes, 124 rows, exactly the 4 expected "
          "split codes, weights sum to 1 per old code, weight=1 for all non-split codes")
    # -----------------------------------------------------------------------

    parish_src = parish_src.merge(
        conversion[["Dicofre", "dtmnfr24", "freguesia_2024_nome", "municipio_2024_nome_oficial", "weight"]],
        on="Dicofre", how="left",
    )
    n_unmatched = parish_src["dtmnfr24"].isna().sum()
    if n_unmatched:
        print(f"    WARNING: {n_unmatched} BGRI rows have a Dicofre not found in the 2016->2024 "
              f"conversion table -- dropping them from the parish rollup")
        parish_src = parish_src.dropna(subset=["dtmnfr24"])

    parish_src["pop_w"] = parish_src["N_INDIVIDUOS"] * parish_src["weight"]
    parish_src["numerator"] = parish_src["access"] * parish_src["pop_w"]
    # access_raw diagnostic, weighted-averaged the same way as access itself
    # (not summed -- access_raw_i values from different BGRIs aren't
    # additive into a meaningful parish total, only comparable as an average)
    parish_src["numerator_raw"] = parish_src["access_raw"] * parish_src["pop_w"]

    grouped = parish_src.groupby("dtmnfr24").agg(
        N_INDIVIDUOS=("pop_w", "sum"),
        numerator_sum=("numerator", "sum"),
        numerator_raw_sum=("numerator_raw", "sum"),
        Municipality=("municipio_2024_nome_oficial", "first"),
        Freguesia=("freguesia_2024_nome", "first"),
    ).reset_index().rename(columns={"dtmnfr24": "Dicofre"})
    grouped["access"] = grouped["numerator_sum"] / grouped["N_INDIVIDUOS"].replace(0, np.nan)
    grouped["access_pct"] = 100 * grouped["access"]
    grouped["access_raw"] = grouped["numerator_raw_sum"] / grouped["N_INDIVIDUOS"].replace(0, np.nan)
    grouped = grouped.drop(columns=["numerator_sum", "numerator_raw_sum"])

    # TSI (01_sustainability_index.py) is computed at the 118 dtmnfr16
    # parishes and hasn't itself been split onto 2024 boundaries, so for the
    # 4 split parishes the same dtmnfr16 TSI value is broadcast to every one
    # of its dtmnfr24 targets (no finer-grained TSI data exists to split it
    # further) -- via the dtmnfr16<->dtmnfr24 crosswalk, not recomputed.
    tsi = pd.read_csv(STATS_DIR / "AML_TSI_parishes.csv", dtype={"Dicofre": str})
    tsi_cols = ["Dicofre", "pct_active", "pct_pt", "pct_car", "pct_zero_car", "TSI"]
    tsi_2024 = conversion[["Dicofre", "dtmnfr24"]].merge(tsi[tsi_cols], on="Dicofre", how="left")
    tsi_2024 = tsi_2024.drop(columns="Dicofre").rename(columns={"dtmnfr24": "Dicofre"}).drop_duplicates(subset="Dicofre")

    parish = grouped.merge(tsi_2024, on="Dicofre", how="left")
    n_missing_tsi = parish["TSI"].isna().sum()
    print(f"    TSI missing for {n_missing_tsi} of {len(parish)} parishes after join")

    parish_stats = parish["access"].agg(["min", "mean", "median", "max"])
    print("\n  Parish-level access_i (normalized) summary statistics:")
    print(parish_stats.to_string())

    parish_cols = ["Dicofre", "Freguesia", "Municipality", "N_INDIVIDUOS",
                    "access", "access_pct", "access_raw", "pct_active", "pct_pt", "pct_car", "pct_zero_car", "TSI"]

    # Geometry: use the actual 2024 parish boundary layer -- conversion
    # weights are for statistic redistribution only and must never be used
    # to approximate geometry (a BGRI's single polygon can't be validly
    # sub-split across multiple 2024 parishes anyway).
    if not PARISH_2024_GEOM_PATH.exists():
        print(f"\n  STOPPING before geometry assembly: no 2024 parish boundary file at "
              f"{PARISH_2024_GEOM_PATH}. A real 2024 parish polygon dataset is required -- "
              f"BGRI geometry must not be used to approximate it.")
        parish[parish_cols].to_csv(STATS_DIR / "AML_accessibility_parishes.csv", index=False)
        print(f"  Saved statistics-only {STATS_DIR / 'AML_accessibility_parishes.csv'} "
              f"(no geojson written)")
        return

    parishes_2024 = gpd.read_file(PARISH_2024_GEOM_PATH)
    parishes_2024["Dicofre"] = parishes_2024["id"].astype(str).str.zfill(6)

    n_geom_codes = parishes_2024["Dicofre"].nunique()
    n_stat_codes = parish["Dicofre"].nunique()
    print(f"\n  [Geometry validation]")
    print(f"    Unique 2024 parish codes in statistics table: {n_stat_codes}")
    print(f"    Unique 2024 parish codes in geometry layer:    {n_geom_codes}")
    assert parishes_2024["Dicofre"].duplicated().sum() == 0, "duplicated Dicofre in 2024 geometry layer"
    assert parish["Dicofre"].duplicated().sum() == 0, "duplicated Dicofre in statistics table"

    unmatched_stats = set(parish["Dicofre"]) - set(parishes_2024["Dicofre"])
    print(f"    Statistics codes with no match in geometry layer: {len(unmatched_stats)} "
          f"{sorted(unmatched_stats) if unmatched_stats else ''}")

    parish_out = parishes_2024[["Dicofre", "geometry"]].merge(parish, on="Dicofre", how="left")
    n_no_access = parish_out["access"].isna().sum()
    print(f"    2024 geometry parishes with no accessibility value after join: {n_no_access}")
    assert len(parish_out) == n_geom_codes, "row count changed across merge -- Dicofre not unique somewhere"
    assert parish_out["Dicofre"].duplicated().sum() == 0, "each 2024 parish must appear exactly once in output"

    parish_out = parish_out.to_crs("EPSG:4326")
    parish_out["geometry"] = parish_out.geometry.simplify(0.0003, preserve_topology=True)

    parish_out[parish_cols + ["geometry"]].to_file(DATA / "AML_accessibility_parishes.geojson", driver="GeoJSON")
    parish[parish_cols].to_csv(STATS_DIR / "AML_accessibility_parishes.csv", index=False)
    print(f"\n  Saved {DATA / 'AML_accessibility_parishes.geojson'}")
    print(f"  Saved {STATS_DIR / 'AML_accessibility_parishes.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
