"""DLF 2.0 REST/PyPaimon catalog, schemas, reads, safe append and overwrite.

Verified workflow: Parquet append-only, unpartitioned tables without primary keys.
Never append a duplicate row to simulate an update. For backfill, backup first.
"""
from __future__ import annotations
import pyarrow as pa
from config import settings

RAW_SCHEMA = pa.schema([
    ("fileName", pa.string()),
    ("picture", pa.binary()),
])

ENRICHED_SCHEMA = pa.schema([
    ("id", pa.string()), ("fileName", pa.string()),
    ("picture", pa.binary()), ("caption", pa.string()),
    ("metadata", pa.string()), ("vector", pa.list_(pa.float32())),
    ("is_cloudy_candidate", pa.int32()), ("is_daytime", pa.int32()),
    ("is_nighttime", pa.int32()), ("weather", pa.string()),
    ("lighting", pa.string()), ("road_type", pa.string()),
    ("scene_type", pa.string()), ("dataset_tag", pa.int32())
])


def connect_catalog():
    from pypaimon import CatalogFactory
    return CatalogFactory.create(settings.catalog_options())


def ensure_table(catalog, name: str, arrow_schema: pa.Schema):
    from pypaimon import Schema
    database, short_name = name.split(".", 1)
    if short_name not in catalog.list_tables(database):
        schema = Schema.from_pyarrow_schema(
            arrow_schema, options={"bucket": "-1", "file.format": "parquet"}
        )
        catalog.create_table(name, schema, ignore_if_exists=False)
        print("Created:", name)
    return catalog.get_table(name)


def read_table(table):
    import pandas as pd
    rb = table.new_read_builder()
    splits = rb.new_scan().plan().splits()
    return rb.new_read().to_pandas(splits) if splits else pd.DataFrame()


def write_arrow(table, data: pa.Table, *, overwrite: bool = False) -> None:
    """Commit one Arrow batch. Overwrite is intentionally explicit."""
    builder = table.new_batch_write_builder()
    if overwrite:
        builder = builder.overwrite()
    writer, commit = builder.new_write(), builder.new_commit()
    try:
        writer.write_arrow(data)
        commit.commit(writer.prepare_commit())
    finally:
        writer.close()
        commit.close()


def insert_if_empty(table, data: pa.Table) -> bool:
    existing = read_table(table)
    if not existing.empty:
        print(f"Table not empty ({len(existing)} rows); skip append to avoid duplicates")
        return False
    write_arrow(table, data, overwrite=False)
    return True


def upload_raw_images(rows: list[dict[str, object]]) -> bool:
    catalog = connect_catalog()
    table = ensure_table(catalog, settings.raw_table, RAW_SCHEMA)
    written = insert_if_empty(table, pa.Table.from_pylist(rows, schema=RAW_SCHEMA))
    check = read_table(table)
    print(settings.raw_table, "row count:", len(check))
    if written and len(check) != len(rows):
        raise AssertionError("Raw table row count mismatch")
    return written


def main() -> None:
    import argparse
    from .prepare_images import list_camfront_images, select_adjacent, read_images
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "write-raw"])
    parser.add_argument("--source", choices=["sweeps", "samples"], default="sweeps")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.action == "check":
        cat = connect_catalog()
        print("databases:", cat.list_databases())
        print("default tables:", cat.list_tables("default"))
    else:
        paths = select_adjacent(list_camfront_images(settings.nu_scenes_dir, args.source), args.limit)
        upload_raw_images(read_images(paths))

if __name__ == "__main__":
    main()
