"""Create enhanced append-only table; backfill vectors by confirmed overwrite.

A Paimon append table has NO row-level updates. This module backs up the full
existing table and requires an explicit confirmation flag before overwrite.
Do not overwrite concurrently with other writers.
"""
from __future__ import annotations
import argparse
import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from config import settings
from .embedding import image_embedding


def make_enriched_records(raw_df: pd.DataFrame, labels: list[dict]) -> list[dict]:
    labels_by_name = {str(row["fileName"]): row for row in labels}
    if len(labels_by_name) != len(labels):
        raise ValueError("Duplicate caption filenames")
    if raw_df["fileName"].duplicated().any():
        raise ValueError("Duplicate source filenames")
    if set(raw_df["fileName"]) != set(labels_by_name):
        raise ValueError("Raw image and caption fileName sets differ")
    rows = []
    for row in raw_df.to_dict("records"):
        name = str(row["fileName"])
        meta = labels_by_name[name]
        weather, lighting = meta.get("weather", "unknown"), meta.get("lighting", "unknown")
        objects = meta.get("objects", [])
        risks = meta.get("risks", [])
        objects = json.loads(objects) if isinstance(objects, str) else objects
        risks = json.loads(risks) if isinstance(risks, str) else risks
        metadata = {
            "weather": weather, "lighting": lighting, "road_type": meta.get("road_type", "unknown"),
            "scene_category": meta.get("scene_type", "unknown"), "objects": objects, "risks": risks,
        }
        rows.append({
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, name)), "fileName": name,
            "picture": bytes(row["picture"]), "caption": str(meta["caption"]),
            "metadata": json.dumps(metadata, ensure_ascii=False), "vector": None,
            "is_cloudy_candidate": int(str(weather).lower() == "cloudy"),
            "is_daytime": int(str(lighting).lower() == "daytime"),
            "is_nighttime": int(str(lighting).lower() == "nighttime"),
            "weather": weather, "lighting": lighting, "road_type": meta.get("road_type", "unknown"),
            "scene_type": f"{lighting}_{weather}", "dataset_tag": 1,
        })
    return rows


def create_enriched_table() -> None:
    import pyarrow as pa
    from .paimon_io import (ENRICHED_SCHEMA, connect_catalog, ensure_table, read_table, insert_if_empty)
    cat = connect_catalog()
    source = read_table(cat.get_table(settings.raw_table))
    if source.empty:
        raise ValueError("Raw image table empty")
    with settings.caption_csv.open(encoding="utf-8", newline="") as f:
        labels = list(csv.DictReader(f))
    rows = make_enriched_records(source, labels)
    table = ensure_table(cat, settings.enriched_table, ENRICHED_SCHEMA)
    insert_if_empty(table, pa.Table.from_pylist(rows, schema=ENRICHED_SCHEMA))
    print("Enriched rows:", len(read_table(table)))


def _vector_present(value) -> bool:
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def backfill_vectors(*, confirm_overwrite: bool = False) -> Path | None:
    """Backup, vectorize all, and overwrite only after explicit confirmation."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .paimon_io import (ENRICHED_SCHEMA, connect_catalog, read_table, write_arrow)
    cat = connect_catalog()
    table = cat.get_table(settings.enriched_table)
    original = read_table(table)
    if original.empty or original["id"].duplicated().any():
        raise ValueError("Empty / duplicate-ID enriched table; inspect manually")
    has = [_vector_present(v) for v in original["vector"]]
    if all(has):
        for v in original["vector"]:
            if len(v) != settings.vector_dim:
                raise ValueError("Unexpected vector dimension")
        print("Already embedded; no overwrite needed")
        return None
    if any(has):
        raise ValueError("Partially embedded table; inspect manually before overwrite")
    if not confirm_overwrite:
        raise RuntimeError("Overwrite required; rerun with --confirm-overwrite after reviewing table/backup policy")
    backup_dir = settings.work_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"enriched_before_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.parquet"
    pq.write_table(pa.Table.from_pandas(original, schema=ENRICHED_SCHEMA, preserve_index=False), backup_path)
    print("Backup:", backup_path)
    updated = original.copy()
    updated["vector"] = [image_embedding(bytes(pic)).tolist() for pic in updated["picture"]]
    if any(len(v) != settings.vector_dim for v in updated["vector"]):
        raise RuntimeError("Vector dimension mismatch; aborting write")
    write_arrow(table, pa.Table.from_pandas(updated, schema=ENRICHED_SCHEMA, preserve_index=False), overwrite=True)
    result = read_table(table)
    if len(result) != len(original) or any(not _vector_present(v) or len(v) != settings.vector_dim for v in result["vector"]):
        raise RuntimeError("Post-overwrite verification failed; restore using backup")
    print("Backfill verified:", len(result), "rows x", settings.vector_dim)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "backfill"])
    parser.add_argument("--confirm-overwrite", action="store_true")
    args = parser.parse_args()
    if args.action == "create":
        create_enriched_table()
    else:
        backfill_vectors(confirm_overwrite=args.confirm_overwrite)

if __name__ == "__main__":
    main()
