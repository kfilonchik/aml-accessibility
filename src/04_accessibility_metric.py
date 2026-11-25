"""
04_accessibility_metric.py

Gravity-style public-transport accessibility per AML BGRI cell, rolled up to
2024 parishes (D.A. Smith's "simple accessibility to population"):

    access_raw_i = sum_{j != i} Population_j * S(travel_time_ij)
    access_i     = access_raw_i / sum_{j != i} Population_j
    S(t)         = 1 / (1 + (t / IMPEDANCE_X0) ** IMPEDANCE_K)

Origins i are AML cells; destinations j are AML + buffer cells. The origin
itself is excluded from both sums, so its own residents cannot inflate the
score of poorly connected cells. Destinations that are unreachable within the
routing cap get weight 0 but stay in the denominator, so access_i is in [0, 1].

access_raw_normalized is access_raw min-max scaled to 0-100 over AML cells; it
is a relative score, not a percentage of population. Parish scores are averages
of the BGRI scores weighted by population and parish conversion weight.

Inputs (both matrices from 03_travel_time_matrices.py must be complete):
  config.MATRICES/aml_matrix_batch_*.parquet
  config.MATRICES/aml_to_buffer_batch_*.parquet

Outputs:
  config.DATA/AML_accessibility_bgri.geojson
  config.DATA/AML_accessibility_parishes.geojson
  config.DATA/ine_statistics/AML_accessibility_bgri.csv
  config.DATA/ine_statistics/AML_accessibility_parishes.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from config import (
    BGRI_BUFFER_PATH, BGRI_PATH, CRS_WGS, DATA, IMPEDANCE_K, IMPEDANCE_X0, MATRICES,
)

STATS_DIR = DATA / "ine_statistics"
PARISH_CONVERSION_PATH = DATA / "polygons" / "dtmnfr21_to_2024_parish_name_conversion.csv"
PARISH_2024_GEOM_PATH = STATS_DIR / "IndicatorsFreguesia.geojson"

SCORES = ["access", "access_raw", "access_raw_normalized"]
BGRI_COLS = ["id", "Municipality", "Dicofre", "Freguesia", "N_INDIVIDUOS",
             "access_raw", "pop_other", "access", "access_pct", "access_raw_normalized"]
TSI_COLS = ["pct_active", "pct_pt", "pct_car", "pct_zero_car", "TSI"]
PARISH_COLS = ["Dicofre", "Freguesia", "Municipality", "N_INDIVIDUOS",
               "access", "access_pct", "access_raw", "access_raw_normalized", *TSI_COLS]


def impedance(travel_time_min: np.ndarray) -> np.ndarray:
    """Decay weight in [0, 1]; NaN (unreachable) maps to 0."""
    weight = 1.0 / (1.0 + (travel_time_min / IMPEDANCE_X0) ** IMPEDANCE_K)
    return np.where(np.isnan(travel_time_min), 0.0, weight)


def load_population(path: Path) -> pd.Series:
    """Population per BGRI cell, in the same ID order as script 03."""
    bgri = gpd.read_file(path, columns=["BGRI2021", "N_INDIVIDUOS"], ignore_geometry=True)
    pop = bgri.set_index(bgri["BGRI2021"].str.strip())["N_INDIVIDUOS"].astype(float)
    if pop.isna().any():
        raise ValueError(f"{path.name}: missing population values")
    print(f"  {path.name}: {len(pop):,} BGRI cells, population {pop.sum():,.0f}")
    return pop.sort_index()


def matrix_batches(prefix: str, n_origins: int, n_destinations: int) -> tuple[list[Path], dict]:
    """List all batches; an unfinished matrix must not be read as unreachable."""
    manifest = json.loads((MATRICES / f"{prefix}_manifest.json").read_text())
    if (manifest["origin_count"], manifest["destination_count"]) != (n_origins, n_destinations):
        raise ValueError(f"{prefix}: population dataset sizes do not match the matrix manifest")
    size = manifest["origin_batch_size"]
    n_batches = (n_origins + size - 1) // size
    paths = [MATRICES / f"{prefix}_batch_{i:03d}.parquet" for i in range(n_batches)]
    if set(MATRICES.glob(f"{prefix}_batch_*.parquet")) != set(paths):
        raise ValueError(f"{prefix}: missing or unexpected batch files; finish script 03 first")
    return paths, manifest


def check_same_settings(aml: dict, buffer: dict) -> None:
    """Both matrices must share origins, network and routing settings."""
    expected_to_differ = {"destination_count", "destination_points_sha256", "origin_batch_size"}
    for key in (aml.keys() | buffer.keys()) - expected_to_differ:
        if aml.get(key) != buffer.get(key):
            raise ValueError(f"AML and buffer matrix settings differ: {key}")


def compute_access(aml_pop: pd.Series, buffer_pop: pd.Series) -> pd.DataFrame:
    """Sum AML and buffer contributions per AML origin, one batch at a time."""
    aml_paths, aml_manifest = matrix_batches("aml_matrix", len(aml_pop), len(aml_pop))
    buffer_paths, buffer_manifest = matrix_batches("aml_to_buffer", len(aml_pop), len(buffer_pop))
    check_same_settings(aml_manifest, buffer_manifest)

    access_raw = pd.Series(0.0, index=aml_pop.index)
    for label, paths, population in [
        ("AML -> AML", aml_paths, aml_pop), ("AML -> buffer", buffer_paths, buffer_pop),
    ]:
        print(f"  Processing {label}: {len(paths)} batches …")
        for i, path in enumerate(paths, start=1):
            df = pd.read_parquet(path, columns=["from_id", "to_id", "travel_time"])
            df = df[df["from_id"] != df["to_id"]]
            destination_pop = df["to_id"].map(population)
            if destination_pop.isna().any() or not df["from_id"].isin(access_raw.index).all():
                raise ValueError(f"{path.name}: unknown origin or destination IDs")
            contrib = impedance(df["travel_time"].to_numpy()) * destination_pop
            sums = contrib.groupby(df["from_id"]).sum()
            access_raw.loc[sums.index] += sums
            if i % 50 == 0 or i == len(paths):
                print(f"    {i}/{len(paths)} batches done")
    pop_other = aml_pop.sum() + buffer_pop.sum() - aml_pop
    return pd.DataFrame({"access_raw": access_raw, "pop_other": pop_other})


def build_bgri(access: pd.DataFrame) -> gpd.GeoDataFrame:
    bgri = gpd.read_file(BGRI_PATH).rename(columns={"BGRI2021": "id"})
    bgri["id"] = bgri["id"].str.strip()
    bgri["access_raw"] = bgri["id"].map(access["access_raw"])
    bgri["pop_other"] = bgri["id"].map(access["pop_other"])
    bgri["access"] = bgri["access_raw"] / bgri["pop_other"]
    bgri["access_pct"] = 100 * bgri["access"]
    minimum, maximum = bgri["access_raw"].min(), bgri["access_raw"].max()
    bgri["access_raw_normalized"] = 100 * (bgri["access_raw"] - minimum) / (maximum - minimum)
    return bgri


def load_parish_conversion() -> pd.DataFrame:
    """2016 -> 2024 parish codes. Split parishes have area weights summing to 1."""
    conversion = pd.read_csv(PARISH_CONVERSION_PATH, dtype=str).rename(columns={"dtmnfr16": "Dicofre"})
    conversion["weight"] = conversion["weight"].astype(float)
    if not np.allclose(conversion.groupby("Dicofre")["weight"].sum(), 1.0):
        raise ValueError(f"{PARISH_CONVERSION_PATH.name}: weights must sum to 1 per 2016 parish")
    return conversion


def aggregate_parishes(bgri: pd.DataFrame, conversion: pd.DataFrame) -> pd.DataFrame:
    """Average BGRI scores weighted by population * parish conversion weight."""
    columns = ["Dicofre", "dtmnfr24", "freguesia_2024_nome", "municipio_2024_nome_oficial", "weight"]
    cells = bgri[["Dicofre", "N_INDIVIDUOS", *SCORES]].merge(conversion[columns], on="Dicofre", how="inner")
    n_dropped = len(bgri) - bgri["Dicofre"].isin(conversion["Dicofre"]).sum()
    print(f"  {n_dropped} BGRI cells without a known parish code are left out of the parish averages")

    cells["pop_w"] = cells["N_INDIVIDUOS"] * cells["weight"]
    cells[SCORES] = cells[SCORES].mul(cells["pop_w"], axis=0)
    parish = cells.groupby("dtmnfr24").agg(
        N_INDIVIDUOS=("pop_w", "sum"),
        Municipality=("municipio_2024_nome_oficial", "first"),
        Freguesia=("freguesia_2024_nome", "first"),
        **{score: (score, "sum") for score in SCORES},
    )
    parish[SCORES] = parish[SCORES].div(parish["N_INDIVIDUOS"], axis=0)
    parish["access_pct"] = 100 * parish["access"]
    return parish.rename_axis("Dicofre").reset_index()


def add_tsi(parish: pd.DataFrame, conversion: pd.DataFrame) -> pd.DataFrame:
    """Join TSI, which exists for 2016 parishes; split parishes inherit its value."""
    tsi = pd.read_csv(STATS_DIR / "AML_TSI_parishes.csv", dtype={"Dicofre": str})
    tsi_2024 = (
        conversion[["Dicofre", "dtmnfr24"]]
        .merge(tsi[["Dicofre", *TSI_COLS]], on="Dicofre", how="left")
        .drop(columns="Dicofre")
        .rename(columns={"dtmnfr24": "Dicofre"})
        .drop_duplicates(subset="Dicofre")
    )
    return parish.merge(tsi_2024, on="Dicofre", how="left")


def main() -> None:
    print("=" * 78)
    print("04  PT accessibility to population (BGRI + parish)")
    print("=" * 78)

    print("\n[1] Loading AML and buffer population …")
    aml_pop = load_population(BGRI_PATH)
    buffer_pop = load_population(BGRI_BUFFER_PATH)

    print("\n[2] Computing accessibility from travel-time batches …")
    access = compute_access(aml_pop, buffer_pop)

    print("\n[3] Writing BGRI results …")
    bgri = build_bgri(access)
    print(bgri[SCORES].agg(["min", "mean", "median", "max"]).to_string())
    bgri_out = bgri[BGRI_COLS + ["geometry"]].to_crs(CRS_WGS)
    bgri_out["geometry"] = bgri_out.geometry.simplify(0.0001, preserve_topology=True)
    bgri_out.to_file(DATA / "AML_accessibility_bgri.geojson", driver="GeoJSON")
    bgri[BGRI_COLS].to_csv(STATS_DIR / "AML_accessibility_bgri.csv", index=False)

    print("\n[4] Writing parish results …")
    conversion = load_parish_conversion()
    parish = add_tsi(aggregate_parishes(bgri, conversion), conversion)
    print(parish["access"].agg(["min", "mean", "median", "max"]).to_string())
    parish[PARISH_COLS].to_csv(STATS_DIR / "AML_accessibility_parishes.csv", index=False)

    # Official 2024 boundaries: split parishes cannot be derived from BGRI cells.
    geometry = gpd.read_file(PARISH_2024_GEOM_PATH)
    geometry["Dicofre"] = geometry["id"].astype(str)
    parish_out = geometry[["Dicofre", "geometry"]].merge(parish, on="Dicofre", how="left", validate="1:1")
    parish_out = parish_out.to_crs(CRS_WGS)
    parish_out["geometry"] = parish_out.geometry.simplify(0.0003, preserve_topology=True)
    parish_out[PARISH_COLS + ["geometry"]].to_file(DATA / "AML_accessibility_parishes.geojson", driver="GeoJSON")

    print(f"\nDone. Results saved under {DATA} and {STATS_DIR}")


if __name__ == "__main__":
    main()
