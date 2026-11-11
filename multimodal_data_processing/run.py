"""Part-1 workflow entrypoint; no expensive external API call without a stage."""
from __future__ import annotations
import argparse
from .paimon_io import connect_catalog, upload_raw_images
from .prepare_images import list_camfront_images, select_adjacent, read_images
from .caption_metadata import annotate_to_csv
from .enrichment import create_enriched_table, backfill_vectors
from .retrieval import main as retrieval_main
from config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check-dlf", "raw", "caption", "enrich", "embedding"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--source", choices=["sweeps", "samples"], default="sweeps")
    parser.add_argument("--confirm-overwrite", action="store_true")
    args = parser.parse_args()
    if args.stage == "check-dlf":
        print("DLF databases:", connect_catalog().list_databases())
        return
    if args.stage == "raw":
        images = select_adjacent(list_camfront_images(settings.nu_scenes_dir, args.source), args.limit)
        upload_raw_images(read_images(images))
    elif args.stage == "caption":
        from .paimon_io import read_table
        df = read_table(connect_catalog().get_table(settings.raw_table))
        annotate_to_csv(df.head(args.limit).to_dict("records"), settings.caption_csv)
    elif args.stage == "enrich":
        create_enriched_table()
    elif args.stage == "embedding":
        backfill_vectors(confirm_overwrite=args.confirm_overwrite)

if __name__ == "__main__":
    main()
