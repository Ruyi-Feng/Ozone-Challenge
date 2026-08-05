"""Convert large events_data CSV to a single sorted Parquet file.

Reads CSV in chunks, sorts each chunk by Event_id, and appends as row groups
to a single Parquet file via ParquetWriter.  Predicate pushdown on Event_id
then skips irrelevant row groups efficiently.

Usage (from repo root):
  python data_processing/scripts/convert_csv_to_parquet.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── config ────────────────────────────────────────────────────────────────
CHUNKSIZE = 2_000_000  # rows per pandas chunk

FILES: list[dict[str, str]] = [
    {
        "csv": "data/processed/data/events_data_train.csv",
        "out": "data/processed/train_events.parquet",
    },
    {
        "csv": "data/processed/data/events_data_val.csv",
        "out": "data/processed/val_events.parquet",
    },
]


def convert_csv_to_parquet(csv_path: str, out_path: str) -> None:
    """Read *csv_path* in chunks, write sorted row groups to *out_path*."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    reader = pd.read_csv(csv_path, chunksize=CHUNKSIZE)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    n_chunks = 0

    for i, chunk in enumerate(reader):
        chunk = chunk.sort_values("Event_id")
        table = pa.Table.from_pandas(chunk)
        if writer is None:
            writer = pq.ParquetWriter(out, table.schema, compression="zstd")
        writer.write_table(table, row_group_size=len(table))
        total_rows += len(chunk)
        n_chunks += 1
        if (i + 1) % 10 == 0:
            print(f"  {i + 1} chunks, {total_rows:,} rows ...")

    if writer is not None:
        writer.close()

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  done: {n_chunks} chunks, {total_rows:,} rows → {out} ({size_mb:.0f} MB)")


def main() -> None:
    for entry in FILES:
        csv = entry["csv"]
        if not Path(csv).exists():
            print(f"SKIP: {csv} not found")
            continue
        print(f"\nConverting {csv} → {entry['out']} ...")
        convert_csv_to_parquet(csv, entry["out"])


if __name__ == "__main__":
    main()
