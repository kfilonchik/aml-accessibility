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
Population_other_i is the combined AML + buffer population minus the
origin's own population. Only AML cells are origins. The origin itself (j = i,
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

access_raw_normalized_i = 100 * (access_raw_i - min(access_raw)) /
(max(access_raw) - min(access_raw)), using all AML BGRI origins. Constant
raw scores produce zero normalized scores. Parish values are population-weighted averages
of these BGRI scores, with the existing parish conversion weights; parish
scores are not normalized again. This relative 0-100 score is not a
percentage of destination population.

Impedance function: S(x) = 1 / (1 + (x / x0)^k), with x0 (minutes) and k
configured by IMPEDANCE_X0 and IMPEDANCE_K in config.py. A single curve is used
(rather than separate per-mode curves) because r5py's TravelTimeMatrix
returns one blended door-to-door TRANSIT+WALK time per OD pair and does
not expose which vehicle mode(s) made up a given trip -- ferry (boat_2)
legs are not treated separately for the same reason; they're already
folded into the same blended travel_time as every other PT leg.

Travel time sources: config.MATRICES/aml_matrix_batch_*.parquet and
aml_to_buffer_batch_*.parquet (see 03_travel_time_matrices.py). Both complete
matrix sets are required, with matching run manifests. Contributions from AML
and buffer destinations are summed per AML origin, reading one batch at a time.
Unreachable pairs, including omitted rows, contribute zero; the denominator is
computed independently from all destination populations, excluding the origin.

Output:
  data/AML_accessibility_bgri.geojson       (BGRI-cell level)
  data/statistics/AML_accessibility_bgri.csv
  data/AML_accessibility_parishes.geojson   (parish rollup, joined with TSI)
  data/statistics/AML_accessibility_parishes.csv
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from config import BGRI_PATH, BGRI_BUFFER_PATH, DATA, MATRICES, IMPEDANCE_X0, IMPEDANCE_K

STATS_DIR = DATA / "ine_statistics"
PARISH_CONVERSION_PATH = DATA / "polygons" / "dtmnfr21_to_2024_parish_name_conversion.csv"
# Official 2024 parish boundary layer (124 freguesias, geometry keyed by "id"
# = dtmnfr24) -- NOT derived from BGRI polygons, so split parishes get their
# real 2024 boundaries rather than an approximation.
PARISH_2024_GEOM_PATH = STATS_DIR / "IndicatorsFreguesia.geojson"

def impedance(travel_time_min: np.ndarray) -> np.ndarray:
    """Sigmoid decay weight in [0, 1]; NaN (unreachable within the matrix's
    120-min cap) maps to 0 weight, same as an infinitely long trip would."""
    weight = 1.0 / (1.0 + (travel_time_min / IMPEDANCE_X0) ** IMPEDANCE_K)
    return np.where(np.isnan(travel_time_min), 0.0, weight)


def normalize_raw_accessibility(raw: pd.Series) -> pd.Series:
    """Min-max scale raw scores across AML BGRIs; preserve missing values."""
    if (raw.dropna() < 0).any() or np.isinf(raw.dropna()).any():
        raise ValueError("Raw accessibility must be finite and nonnegative")
    minimum, maximum = raw.min(), raw.max()
    if maximum > minimum:
        return 100.0 * (raw - minimum) / (maximum - minimum)
    # Constant scores have no range: use zero and preserve missing values.
    return raw * 0.0



def aggregate_parish_scores(parish_src: pd.DataFrame) -> pd.DataFrame:
    """Average BGRI scores using population and parish conversion weights."""
    parish_src = parish_src.copy()
    parish_src["pop_w"] = parish_src["N_INDIVIDUOS"] * parish_src["weight"]
    scores = ["access", "access_raw", "access_raw_normalized"]
    aggregations = {
        "N_INDIVIDUOS": ("pop_w", "sum"),
        "Municipality": ("municipio_2024_nome_oficial", "first"),
        "Freguesia": ("freguesia_2024_nome", "first"),
    }
    for score in scores:
        column = f"numerator_{score}"
        parish_src[column] = parish_src[score] * parish_src["pop_w"]
        aggregations[column] = (column, "sum")
    grouped = parish_src.groupby("dtmnfr24").agg(**aggregations).reset_index()
    grouped = grouped.rename(columns={"dtmnfr24": "Dicofre"})
    denominator = grouped["N_INDIVIDUOS"].replace(0, np.nan)
    for score in scores:
        grouped[score] = grouped.pop(f"numerator_{score}") / denominator
    grouped["access_pct"] = 100 * grouped["access"]
    return grouped


def load_population_file(path: Path) -> pd.Series:
    bgri = gpd.read_file(path)[["BGRI2021", "N_INDIVIDUOS"]]
    if bgri["BGRI2021"].isna().any():
        raise ValueError(f"{path.name}: missing BGRI2021 IDs")
    ids = bgri["BGRI2021"].astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError(f"{path.name}: IDs must be nonempty and unique")
    population = pd.to_numeric(bgri["N_INDIVIDUOS"], errors="raise").astype(float)
    if not np.isfinite(population).all() or (population < 0).any():
        raise ValueError(f"{path.name}: population must be finite and nonnegative")
    pop = pd.Series(population.to_numpy(), index=pd.Index(ids, name="BGRI2021"))
    print(f"  {path.name}: {len(pop):,} BGRI cells, population {pop.sum():,.0f}")
    return pop.sort_index()


def load_population() -> tuple[pd.Series, pd.Series]:
    """Keep origin population (AML) separate from external destination population."""
    aml = load_population_file(BGRI_PATH)
    buffer = load_population_file(BGRI_BUFFER_PATH)
    if not aml.index.intersection(buffer.index).empty:
        raise ValueError("AML and buffer datasets share BGRI2021 IDs")
    return aml, buffer


def matrix_batches(prefix: str, origin_count: int, destination_count: int) -> tuple[list[Path], dict]:
    """Require all checkpoints so unfinished routing is not treated as unreachable."""
    manifest_path = MATRICES / f"{prefix}_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}; finish script 03 for both matrices first")
    manifest = json.loads(manifest_path.read_text())
    if manifest["origin_count"] != origin_count or manifest["destination_count"] != destination_count:
        raise ValueError(f"{prefix}: population dataset sizes do not match the matrix manifest")
    size = manifest["origin_batch_size"]
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"{prefix}: invalid origin batch size")
    count = (origin_count + size - 1) // size
    paths = [MATRICES / f"{prefix}_batch_{i:03d}.parquet" for i in range(count)]
    if set(MATRICES.glob(f"{prefix}_batch_*.parquet")) != set(paths):
        raise ValueError(f"{prefix}: missing or unexpected batch files; finish script 03 before script 04")
    return paths, manifest


def compute_access(
    aml_pop: pd.Series, buffer_pop: pd.Series, *, allow_mixed_r5py_versions: bool = False,
) -> pd.DataFrame:
    """Sum both destination groups per AML origin, with a fixed population denominator.

    All destination populations are included even if a travel time is NaN or
    the pair is absent. Self-pairs contribute neither population nor impedance.
    Batch boundaries may differ between the two matrices.
    """
    if not aml_pop.index.is_unique or not buffer_pop.index.is_unique:
        raise ValueError("Population IDs must be unique")
    if not aml_pop.index.intersection(buffer_pop.index).empty:
        raise ValueError("AML and buffer population IDs must not overlap")
    aml_pop = aml_pop.sort_index()
    groups = []
    for prefix, population in [("aml_matrix", aml_pop), ("aml_to_buffer", buffer_pop)]:
        paths, manifest = matrix_batches(prefix, len(aml_pop), len(population))
        groups.append((prefix, population, paths, manifest))
    # Destination hashes/counts and batch sizes intentionally differ, but the
    # origin set, network, date and routing settings must be the same.
    shared_keys = [
        "schema_version", "r5py_version", "departure", "departure_window_seconds",
        "max_time_seconds", "max_walking_seconds", "transport_modes", "percentiles",
        "origin_count", "origin_points_sha256", "inputs",
    ]
    for key in shared_keys:
        if key not in groups[0][3] or key not in groups[1][3]:
            raise ValueError(f"Matrix manifests are missing {key}")
        if groups[0][3][key] != groups[1][3][key]:
            if key == "r5py_version" and allow_mixed_r5py_versions:
                warnings.warn(
                    "Combining matrices generated with different r5py versions: "
                    f"AML={groups[0][3][key]}, buffer={groups[1][3][key]}. "
                    "Matching inputs/settings do not guarantee equivalent routing behavior.",
                    RuntimeWarning,
                )
                continue
            if key == "r5py_version":
                raise ValueError(
                    "AML and buffer matrix settings differ: r5py_version "
                    f"(AML={groups[0][3][key]}, buffer={groups[1][3][key]}). "
                    "Regenerate with the same version, or explicitly accept mixed versions "
                    "with --allow-mixed-r5py-versions."
                )
            raise ValueError(f"AML and buffer matrix settings differ: {key}")
    if groups[0][3]["percentiles"] != [50]:
        raise ValueError("Accessibility requires percentile-50 matrices")

    access_raw = pd.Series(0.0, index=aml_pop.index)
    for prefix, population, paths, manifest in groups:
        size = manifest["origin_batch_size"]
        print(f"  Processing {prefix}: {len(paths)} batches …")
        for i, path in enumerate(paths):
            df = pd.read_parquet(path, columns=["from_id", "to_id", "travel_time"])
            if df[["from_id", "to_id"]].isna().any().any():
                raise ValueError(f"{path.name}: missing origin/destination IDs")
            df["from_id"] = df["from_id"].astype(str).str.strip()
            df["to_id"] = df["to_id"].astype(str).str.strip()
            expected_origins = aml_pop.index[i * size:(i + 1) * size]
            if not df["from_id"].isin(expected_origins).all():
                raise ValueError(f"{path.name}: unexpected origins for this batch")
            if not df["to_id"].isin(population.index).all():
                raise ValueError(f"{path.name}: unknown or wrong-group destination IDs")
            if df.duplicated(["from_id", "to_id"]).any():
                raise ValueError(f"{path.name}: duplicate origin/destination pairs")
            df = df.loc[df["from_id"] != df["to_id"]].copy()
            times = pd.to_numeric(df["travel_time"], errors="raise").to_numpy(dtype=float)
            if (times < 0).any():
                raise ValueError(f"{path.name}: negative travel times")
            df["contrib"] = impedance(times) * df["to_id"].map(population).to_numpy()
            sums = df.groupby("from_id")["contrib"].sum()
            access_raw.loc[sums.index] += sums
            print(f"    batch {i+1}/{len(paths)}: {path.name} done")
            del df
    total_population = aml_pop.sum() + buffer_pop.sum()
    return pd.DataFrame({"access_raw": access_raw, "pop_other": total_population - aml_pop})


def main(*, allow_mixed_r5py_versions: bool = False) -> None:
    print("=" * 78)
    print("04  Simple PT accessibility-to-population metric (BGRI + parish)")
    print("=" * 78)

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[1] Loading AML and buffer destination population …")
    aml_pop, buffer_pop = load_population()

    print("\n[2] Computing Access_i from travel-time batches …")
    access = compute_access(
        aml_pop, buffer_pop, allow_mixed_r5py_versions=allow_mixed_r5py_versions,
    )

    print("\n[3] Assembling BGRI-level output …")
    bgri = gpd.read_file(BGRI_PATH)
    bgri = bgri.rename(columns={"BGRI2021": "id"})
    bgri["id"] = bgri["id"].astype(str).str.strip()
    bgri["access_raw"] = bgri["id"].map(access["access_raw"])
    bgri["pop_other"] = bgri["id"].map(access["pop_other"])
    n_missing = bgri["access_raw"].isna().sum()
    if n_missing:
        print(f"    WARNING: {n_missing} BGRI cells missing from the travel-time matrix (not routed)")
    # access_i = access_raw_i / Population_other_i -- a population-weighted
    # average impedance over AML + buffer destinations (unreachable included at
    # weight 0, see module docstring). This, not access_raw, is what the
    # parish rollup below weights and averages.
    bgri["access"] = bgri["access_raw"] / bgri["pop_other"].replace(0, np.nan)
    bgri["access_pct"] = 100 * bgri["access"]
    bgri["access_raw_normalized"] = normalize_raw_accessibility(bgri["access_raw"])

    stats = bgri[["access_raw", "access", "access_raw_normalized"]].agg(["min", "mean", "median", "max"])
    print("\n  access_raw_i and access_i (normalized) summary statistics:")
    print(stats.to_string())

    bgri_cols = ["id", "Municipality", "Dicofre", "Freguesia", "N_INDIVIDUOS",
                 "access_raw", "pop_other", "access", "access_pct", "access_raw_normalized"]
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

    grouped = aggregate_parish_scores(parish_src)

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
                    "access", "access_pct", "access_raw", "access_raw_normalized",
                    "pct_active", "pct_pt", "pct_car", "pct_zero_car", "TSI"]

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
    parser = argparse.ArgumentParser(description="Combine AML and buffer accessibility contributions.")
    parser.add_argument(
        "--allow-mixed-r5py-versions", action="store_true",
        help="Accept matrices generated with different r5py versions; all other checks remain enabled.",
    )
    main(allow_mixed_r5py_versions=parser.parse_args().allow_mixed_r5py_versions)
