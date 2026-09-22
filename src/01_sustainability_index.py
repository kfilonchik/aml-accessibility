"""
01_sustainability_index.py

Adapted Travel Sustainability Index (TSI) for the 118 freguesias of the
Lisbon Metropolitan Area (AML), based on journey-to-work/study modal-share
data from Censos 2021 (INE), now extended with a car-ownership component.

The original TSI formula combines car-ownership and modal-share data:

    TSI = 1.25 * ( (2*pct_zero_car + pct_active + pct_pt + (100 - pct_car)) / 5 )

INE's public 2021 census release does not provide household car-ownership
at freguesia geography for Portugal, so an earlier version of this script
dropped the car-ownership term and used a 3-component formula instead. This
version restores the full formula using a car-ownership figure from a
separate mobility-indicators dataset (data/statistics/IndicatorsFreguesia.
geojson, field `veh_ownership_pct_hh_no_vehicle`).

  pct_active   : % of commuters whose main mode is walking or cycling
  pct_pt       : % of commuters whose main mode is public transport
                 (bus, company/school transport excluded, metro, train, ferry)
  pct_car      : % of commuters whose main mode is car (driver or passenger)
                 or motorcycle
  pct_zero_car : % of HOUSEHOLDS (not commuters -- a different unit and
                 population from the three modal-share terms above) with no
                 motor vehicle. This is a proxy, not an exact match to "zero
                 CAR ownership": the source field is veh_ownership_pct_hh_
                 no_vehicle, i.e. zero vehicles of any kind (car, motorcycle,
                 etc.), not specifically zero cars -- IndicatorsFreguesia.
                 geojson has no car-specific ownership rate, only aggregate
                 vehicle stats (veh_ownership_avg_cars etc.) and this one
                 no-vehicle share. Used as the closest available proxy.

Coverage caveat: IndicatorsFreguesia.geojson uses different Dicofre codes
for 4 of our 118 freguesias -- all "Uniao de Freguesias" (2013 merged-
parish) codes in Sintra (111123, 111126, 111127) and Seixal (151007) --
while that dataset instead carries the pre-merger constituent-parish codes
(111129-111135, 151008-151010) under their old individual names. These are
reconciled via an explicit MERGED_PARISH_CROSSWALK (section 2b below), not
guessed: within each group, veh_ownership_pct_hh_no_vehicle and
veh_ownership_n_households are identical across every constituent row
(verified against the data), confirming the source was already computed at
the merged-parish level and only fanned out under the old names -- so the
merged parish's value is looked up exactly, with a check that logs a
warning if constituents in a group ever disagree.

Data caveat — "employed commuters" vs. "all commuters":
  INE indicator 0011704 was exported across 5 CSV files split by
  municipality. Two of them (Alcochete/Almada/Amadora/Barreiro/Cascais and
  Lisboa — 46 of 118 freguesias) only carry the "Total" condição-perante-o-
  trabalho breakdown (employed + student combined), not an employed-only
  cut like the other three files. To keep every parish on the same
  definition, this script uses the "Total" (employed + student) condition
  uniformly for all 118 freguesias, rather than mixing definitions across
  parishes.

Mode mapping (INE codes):
  active : 01 A pé, 09 Bicicleta
  pt     : 04 Autocarro, 06 Metropolitano, 07 Comboio, 10 Barco
  car    : 02 Automóvel ligeiro - condutor, 03 - passageiro, 08 Motociclo
  (taxi is not broken out separately by INE; it falls inside the residual
  "11 Outro" code together with other unclassified responses, so it cannot
  be cleanly reassigned to car without misclassifying that bucket)
  excluded from all three buckets : 05 Transporte coletivo da empresa/escola,
  11 Outro — these inflate neither numerator nor any other category, they
  just don't contribute to pct_active / pct_pt / pct_car.

Parish geometry: data/polygons/BRGI_AML_with_Parishes_singlepolyg.geojson
(the current single-polygon-per-BGRI2021 file; 19 BGRI rows have a null
Dicofre and are dropped before dissolving) dissolved by Dicofre, which
covers all 18 AML municipalities (118 freguesias).

Output: data/AML_TSI_parishes.geojson
"""

from __future__ import annotations

import re

import geopandas as gpd
import pandas as pd

from config import BGRI_PATH, CRS_PT, CRS_WGS, DATA

STATS_DIR = DATA / "ine_statistics"

VEH_OWNERSHIP_PATH = STATS_DIR / "IndicatorsFreguesia.geojson"
OUT_GEOJSON = DATA / "AML_TSI_parishes.geojson"
OUT_CSV     = STATS_DIR / "AML_TSI_parishes.csv"

CSV_FILES = sorted(STATS_DIR.glob("ine_indicador_0011704*.csv"))

COL_RE = re.compile(r"^S7A2021:2021-T:HM-([^-]+:[^-]+)-([^-]+:.+?)-(\d+:.+)$")

MODE_ACTIVE = {"01", "09"}                # A pé, Bicicleta
MODE_PT     = {"04", "06", "07", "10"}    # Autocarro, Metropolitano, Comboio, Barco
MODE_CAR    = {"02", "03", "08"}          # Auto-condutor, Auto-passageiro, Motociclo
MODE_TOTAL  = "T"

# ---------------------------------------------------------------------------
# 1. Parse the INE modal-share extracts into per-freguesia mode totals
# ---------------------------------------------------------------------------

records = []

for csv_path in CSV_FILES:
    with open(csv_path, encoding="utf-8-sig") as f:
        header = f.readline().strip().split(";")

    # column index -> (condition_code, mode_code) for every column except
    # the row-label column; destination breakdown is summed away below
    col_meta = []
    for h in header[1:]:
        m = COL_RE.match(h)
        cond_code = m.group(1).split(":")[0]
        mode_code = m.group(2).split(":")[0]
        col_meta.append((cond_code, mode_code))

    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", header=0)
    label_col = df.columns[0]

    for _, row in df.iterrows():
        dicofre = str(row[label_col]).split(":", 1)[0].strip()
        if len(dicofre) != 6:
            continue  # skip municipality (concelho) aggregate rows

        mode_sums = {}
        values = row.values[1:]
        for (cond_code, mode_code), value in zip(col_meta, values):
            if cond_code != MODE_TOTAL:
                continue  # use Total (employed + student) condition only
            mode_sums[mode_code] = mode_sums.get(mode_code, 0) + (value if pd.notna(value) else 0)

        records.append({"Dicofre": dicofre, **mode_sums})

modal = pd.DataFrame(records).fillna(0)
modal = modal.groupby("Dicofre", as_index=False).sum()

assert modal["Dicofre"].duplicated().sum() == 0
print(f"Parsed modal-share data for {len(modal)} freguesias")

# ---------------------------------------------------------------------------
# 2. Aggregate modes into active / pt / car and compute shares
# ---------------------------------------------------------------------------

mode_cols = [c for c in modal.columns if c != "Dicofre"]

modal["n_total"]  = modal[MODE_TOTAL]
modal["n_active"] = modal[[c for c in MODE_ACTIVE if c in modal.columns]].sum(axis=1)
modal["n_pt"]     = modal[[c for c in MODE_PT if c in modal.columns]].sum(axis=1)
modal["n_car"]    = modal[[c for c in MODE_CAR if c in modal.columns]].sum(axis=1)

modal["pct_active"] = 100 * modal["n_active"] / modal["n_total"]
modal["pct_pt"]     = 100 * modal["n_pt"] / modal["n_total"]
modal["pct_car"]    = 100 * modal["n_car"] / modal["n_total"]

# ---------------------------------------------------------------------------
# 2b. Zero-car(-vehicle) ownership, from a separate mobility-indicators
# dataset (INE's census release has no car-ownership at freguesia level).
# See module docstring for the no-vehicle-vs-no-car proxy caveat.
#
# 4 of our freguesias are 2013 parish mergers ("Uniao de Freguesias") that
# IndicatorsFreguesia.geojson instead lists under their old, pre-merger
# constituent-parish codes/names -- e.g. our 111123 "Uniao das freguesias
# de Almargem do Bispo, Pero Pinheiro e Montelavar" corresponds to that
# dataset's separate 111129/111131/111132 rows for those three names. This
# isn't a coincidence: within each such group, veh_ownership_n_households
# and veh_ownership_pct_hh_no_vehicle are IDENTICAL across every constituent
# row (verified directly against the data), meaning the source was already
# computed at the merged-parish level and fanned out under the old names,
# not independently re-measured per historical parish. So the merged
# parish's value is looked up exactly via this crosswalk, not approximated.
# ---------------------------------------------------------------------------

MERGED_PARISH_CROSSWALK = {
    "151007": ["151008", "151009", "151010"],  # Seixal, Arrentela e Aldeia de Paio Pires
    "111123": ["111129", "111131", "111132"],  # Almargem do Bispo, Pero Pinheiro e Montelavar
    "111126": ["111130", "111133"],            # Queluz e Belas
    "111127": ["111134", "111135"],            # Sao Joao das Lampas e Terrugem
}

veh = gpd.read_file(VEH_OWNERSHIP_PATH)[["id", "veh_ownership_pct_hh_no_vehicle"]]
veh = veh.rename(columns={"id": "Dicofre", "veh_ownership_pct_hh_no_vehicle": "pct_zero_car"})
veh["Dicofre"] = veh["Dicofre"].astype(str).str.zfill(6)
veh = veh.dropna(subset=["Dicofre"]).drop_duplicates(subset="Dicofre")
veh_by_code = veh.set_index("Dicofre")["pct_zero_car"]

crosswalk_rows = []
for merged_code, old_codes in MERGED_PARISH_CROSSWALK.items():
    values = veh_by_code.reindex(old_codes).dropna()
    if values.empty:
        continue
    if values.nunique() > 1:
        print(f"  WARNING: constituent parishes for {merged_code} disagree "
              f"({dict(values)}) -- skipping crosswalk fill for this code")
        continue
    crosswalk_rows.append({"Dicofre": merged_code, "pct_zero_car": values.iloc[0]})

veh = pd.concat([veh, pd.DataFrame(crosswalk_rows)], ignore_index=True)

modal = modal.merge(veh, on="Dicofre", how="left")
n_missing_veh = modal["pct_zero_car"].isna().sum()
print(f"pct_zero_car missing for {n_missing_veh} freguesias "
      f"({len(crosswalk_rows)} filled via the merged-parish crosswalk above)")

# ---------------------------------------------------------------------------
# 3. Compute TSI and its sub-components
# ---------------------------------------------------------------------------

modal["TSI_active_component"]    = modal["pct_active"]
modal["TSI_pt_component"]        = modal["pct_pt"]
modal["TSI_car_component"]       = 100 - modal["pct_car"]
modal["TSI_zero_car_component"]  = modal["pct_zero_car"]

modal["TSI"] = 1.25 * (
    (2 * modal["TSI_zero_car_component"]
     + modal["TSI_active_component"] + modal["TSI_pt_component"] + modal["TSI_car_component"]) / 5
)

stats = modal["TSI"].agg(["min", "mean", "median", "max"])
print("\nTSI summary statistics (raw):")
print(stats.to_string())

full_range = stats["max"] - stats["min"]
print(f"\nEmpirical range: {full_range:.1f} (min={stats['min']:.1f}, max={stats['max']:.1f})")

if full_range < 60:
    # Range is compressed well short of the full 0-100 span: rescale TSI
    # with min-max normalisation so it spans 0-100 for mapping/comparison.
    # The three sub-components are kept as raw percentages (un-normalised)
    # for inspection, since they are already bounded 0-100 by construction.
    print("Range is compressed — applying min-max normalisation to TSI (0-100).")
    modal["TSI_raw"] = modal["TSI"]
    modal["TSI"] = 100 * (modal["TSI_raw"] - stats["min"]) / full_range
else:
    print("Empirical range covers most of the 0-100 span — no normalisation applied.")

# ---------------------------------------------------------------------------
# 4. Attach parish geometry (full 118-freguesia AML, including Lisboa)
# ---------------------------------------------------------------------------

bgri = gpd.read_file(BGRI_PATH).to_crs(CRS_PT)
n_null_dicofre = bgri["Dicofre"].isna().sum()
if n_null_dicofre:
    print(f"Dropping {n_null_dicofre} BGRI rows with a null Dicofre before dissolving")
    bgri = bgri.dropna(subset=["Dicofre"])
bgri["Dicofre"] = bgri["Dicofre"].astype(str).str.zfill(6)
parish_geom = bgri[["Dicofre", "Municipality", "Freguesia", "geometry"]].dissolve("Dicofre", aggfunc="first").reset_index()

gdf = parish_geom.merge(modal, on="Dicofre", how="inner")
print(f"\nMerged geometry + modal data for {len(gdf)} of {len(parish_geom)} freguesias")

# ---------------------------------------------------------------------------
# 5. Export (reprojected to WGS84 for web mapping)
# ---------------------------------------------------------------------------

save_cols = [
    "Dicofre", "Freguesia", "Municipality",
    "n_total", "n_active", "n_pt", "n_car",
    "pct_active", "pct_pt", "pct_car", "pct_zero_car",
    "TSI_active_component", "TSI_pt_component", "TSI_car_component", "TSI_zero_car_component",
    "TSI",
]
if "TSI_raw" in gdf.columns:
    save_cols.append("TSI_raw")

gdf_out = gdf[save_cols + ["geometry"]].to_crs(CRS_WGS)
# Simplify boundaries (dissolved from BGRI census-block polygons, ~140k
# vertices total) for a lighter file — visually lossless at web-map zoom.
gdf_out["geometry"] = gdf_out.geometry.simplify(0.0003, preserve_topology=True)
gdf[save_cols].to_csv(OUT_CSV, index=False)
gdf_out.to_file(OUT_GEOJSON, driver="GeoJSON")

print(f"\nSaved {OUT_CSV}")
print(f"Saved {OUT_GEOJSON}")
