"""
01_validate_gtfs.py

Validates each of the 8 GTFS feeds against the chosen DEPARTURE_DATE
(config.py), checks for duplicate stop_ids and stops with no trips, and
zips each feed folder to data/gtfs_zips/<feed>.gtfs.zip — both r5py's
TransportNetwork(gtfs=[...]) and city2graph.load_gtfs() require a .zip,
while this project currently stores feeds as unzipped folders.

If DEPARTURE_DATE is not covered by all 8 feeds, this script searches
forward for the next Tuesday that is, and reports it — config.py's
DEPARTURE_DATE should then be updated to that value before proceeding.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile

import pandas as pd

from config import DEPARTURE_DATE, FEEDS, GTFS_ZIPS, MTS_PATCH_SERVICE, STATS_DIR
from validation import date_covered_by_feed, duplicate_stop_ids, stops_with_no_trips


def _patched_mts_calendar_dates(feed_dir) -> pd.DataFrame:
    """MTS-20240129's calendar.txt has no service past 2025-09-07. Per user
    direction, we adapt the existing weekday pattern (MTS_PATCH_SERVICE) to
    DEPARTURE_DATE by adding one calendar_dates.txt 'added' exception, rather
    than dropping the mode from the analysis."""
    cal_dates = pd.read_csv(feed_dir / "calendar_dates.txt", dtype=str, encoding="utf-8-sig")
    patch_row = pd.DataFrame([{
        "service_id": MTS_PATCH_SERVICE,
        "exception_type": "1",
        "date": DEPARTURE_DATE.strftime("%Y%m%d"),
    }])
    # preserve this feed's actual column order (service_id,exception_type,date)
    patch_row = patch_row[cal_dates.columns]
    return pd.concat([cal_dates, patch_row], ignore_index=True)


# GTFS-required files — kept even if they look small; everything else is
# optional and r5py's strict parser errors on a table that is *present* in
# the zip but has zero data rows (e.g. cascais/fare_rules.txt: header only),
# so those are dropped from the zip rather than included empty.
REQUIRED_GTFS_FILES = {
    "agency.txt", "stops.txt", "routes.txt", "trips.txt", "stop_times.txt",
    "calendar.txt", "calendar_dates.txt",
}


def _has_data_rows(path) -> bool:
    with open(path, "r", encoding="utf-8-sig") as fh:
        next(fh, None)   # header
        return next(fh, None) is not None


def zip_feed(name: str, feed_dir) -> None:
    out_path = GTFS_ZIPS / f"{name}.gtfs.zip"
    skipped = []
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(feed_dir.glob("*.txt")):
            if name == "mts" and f.name == "calendar_dates.txt":
                patched = _patched_mts_calendar_dates(feed_dir)
                zf.writestr("calendar_dates.txt", patched.to_csv(index=False))
                continue
            if f.name not in REQUIRED_GTFS_FILES and not _has_data_rows(f):
                skipped.append(f.name)
                continue
            zf.write(f, arcname=f.name)
    note = ""
    if name == "mts":
        note += "  [calendar_dates.txt patched for DEPARTURE_DATE]"
    if skipped:
        note += f"  [skipped empty optional tables: {skipped}]"
    print(f"    Zipped {name} -> {out_path.relative_to(GTFS_ZIPS.parent.parent)} "
          f"({out_path.stat().st_size / 1e6:.1f} MB){note}")


def check_calendar_coverage(date: dt.date) -> pd.DataFrame:
    rows = []
    for name, feed_dir in FEEDS.items():
        covered, detail = date_covered_by_feed(feed_dir, date)
        if name == "mts" and not covered:
            covered, detail = True, (
                f"native calendar stale (no service past 2025-09-07) — "
                f"patched via calendar_dates exception on {MTS_PATCH_SERVICE}"
            )
        rows.append({"feed": name, "date": date.isoformat(), "covered": covered, "detail": detail})
    return pd.DataFrame(rows)


def find_next_covered_tuesday(start: dt.date, max_weeks: int = 26) -> dt.date | None:
    d = start
    while d.weekday() != 1:   # Tuesday
        d += dt.timedelta(days=1)
    for _ in range(max_weeks):
        cov = check_calendar_coverage(d)
        if cov["covered"].all():
            return d
        d += dt.timedelta(weeks=1)
    return None


def main() -> None:
    print("=" * 78)
    print("01  Validate GTFS feeds and zip for r5py / city2graph")
    print("=" * 78)

    print(f"\n[1] Calendar coverage check for DEPARTURE_DATE = {DEPARTURE_DATE} …")
    cov = check_calendar_coverage(DEPARTURE_DATE)
    print(cov.to_string(index=False))
    cov.to_csv(STATS_DIR / "gtfs_calendar_coverage.csv", index=False)

    if not cov["covered"].all():
        print(f"\n  WARNING: DEPARTURE_DATE not covered by all feeds — searching for next "
              f"covered Tuesday …")
        alt = find_next_covered_tuesday(DEPARTURE_DATE)
        if alt is None:
            print("  Could not find a Tuesday covered by all feeds within 26 weeks — "
                  "manual inspection needed.")
        else:
            print(f"  RECOMMENDATION: set DEPARTURE_DATE = {alt.isoformat()} in config.py "
                  f"(first Tuesday covered by all 8 feeds).")
    else:
        print(f"\n  All 8 feeds cover {DEPARTURE_DATE} — no change needed.")

    print("\n[2] Duplicate stop_id check …")
    dup_rows = []
    for name, feed_dir in FEEDS.items():
        dupes = duplicate_stop_ids(feed_dir)
        print(f"    {name:14s}  {len(dupes)} duplicated stop_id rows")
        if len(dupes):
            dupes.insert(0, "feed", name)
            dup_rows.append(dupes)
    if dup_rows:
        pd.concat(dup_rows).to_csv(STATS_DIR / "gtfs_duplicate_stop_ids.csv", index=False)

    print("\n[3] Stops with zero trips …")
    for name, feed_dir in FEEDS.items():
        orphans = stops_with_no_trips(feed_dir)
        print(f"    {name:14s}  {len(orphans)} stops with no stop_times rows")

    print("\n[4] Zipping feeds for r5py / city2graph …")
    for name, feed_dir in FEEDS.items():
        zip_feed(name, feed_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
