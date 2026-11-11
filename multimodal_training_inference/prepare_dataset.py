"""Paimon enhanced table → stable 19/1 split → local JPEG and JSONL files.

Train/test from consecutive frames is a smoke test, not a generalization study.
Keep the split manifest so base and LoRA always see the SAME held-out image.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import uuid
from pathlib import Path
import pandas as pd
from PIL import Image
from config import settings


def select_valid_rows(df: pd.DataFrame, dataset_tag: int = 1) -> pd.DataFrame:
    required = {"id", "fileName", "picture", "caption", "dataset_tag"}
    if not required <= set(df.columns):
        raise ValueError(f"Missing Paimon columns: {sorted(required - set(df.columns))}")
    rows = df.loc[
        df["dataset_tag"].eq(dataset_tag)
        & df["picture"].notna()
        & df["caption"].notna()
        & df["caption"].astype(str).str.strip().ne("")
    ].copy()
    if rows.empty:
        raise ValueError("No labeled rows with dataset_tag=1")
    if rows["id"].duplicated().any() or rows["fileName"].duplicated().any():
        raise ValueError("Duplicate image id or filename")
    return rows.reset_index(drop=True)


def split_records(rows: pd.DataFrame, val_ratio: float, seed: int):
    """Stable selection by image ID before seeded shuffle; reproduces across read orders."""
    if not 0 < val_ratio < 1 or len(rows) < 2:
        raise ValueError("Need >=2 samples and 0<val_ratio<1")
    ordered = rows.sort_values("id").sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, min(len(rows)-1, round(len(rows)*val_ratio)))
    validation = ordered.iloc[:n_val].copy()
    training = ordered.iloc[n_val:].copy()
    assert set(training["id"]).isdisjoint(set(validation["id"]))
    return training, validation


def dataset_statistics(rows: pd.DataFrame) -> dict:
    """Dataset Statistics Overview from the tutorial: caption and image sizes."""
    captions = rows["caption"].astype(str).str.len()
    image_sizes = rows["picture"].map(lambda b: len(bytes(b)))
    result = {
        "samples": int(len(rows)),
        "caption_chars_min": int(captions.min()),
        "caption_chars_max": int(captions.max()),
        "caption_chars_mean": round(float(captions.mean()), 2),
        "caption_chars_median": round(float(captions.median()), 2),
        "image_size_kb_min": round(float(image_sizes.min()) / 1024, 2),
        "image_size_kb_max": round(float(image_sizes.max()) / 1024, 2),
        "image_size_kb_mean": round(float(image_sizes.mean()) / 1024, 2),
    }
    return result


def write_dataset(train: pd.DataFrame, val: pd.DataFrame, dest: Path, seed: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    image_dir = dest / "images"
    image_dir.mkdir(exist_ok=True)
    signature = hashlib.sha256(
        "\n".join(sorted([str(x) for x in list(train["id"]) + list(val["id"])] )).encode()
    ).hexdigest()
    manifest_file = dest / "split_manifest.json"
    manifest = {
        "seed": seed,
        "all_ids_sha256": signature,
        "train_ids": [str(x) for x in train["id"]],
        "validation_ids": [str(x) for x in val["id"]],
    }
    if not manifest_file.exists() and any((dest / name).exists() for name in ("train.jsonl", "validation.jsonl")):
        raise RuntimeError("Existing JSONL data has no split manifest; use a new --output-dir to protect the previous validation sample and baseline")
    if manifest_file.exists():
        old = json.loads(manifest_file.read_text(encoding="utf-8"))
        if old != manifest:
            raise RuntimeError("Existing split_manifest.json differs. Choose a new --output-dir to avoid overwriting a validated split.")
    else:
        manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    for subset_name, frame in (("train", train), ("validation", val)):
        out = []
        for row in frame.to_dict("records"):
            name = str(row["id"]) + ".jpg"
            target = image_dir / name
            data = bytes(row["picture"])
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError(f"Image file mismatch: {target}")
            else:
                target.write_bytes(data)
            with Image.open(target) as im:
                im.verify()
            out.append({"id": str(row["id"]), "image": f"images/{name}",
                        "caption": str(row["caption"]), "fileName": str(row["fileName"])})
        with (dest / f"{subset_name}.jsonl").open("w", encoding="utf-8") as f:
            for entry in out:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"{subset_name}: {len(out)} samples")


def load_records(path: Path) -> list[dict]:
    base = path.parent
    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    for row in records:
        if "id" not in row:
            if not row.get("fileName"):
                raise ValueError("Legacy JSONL requires fileName to reconstruct image ID")
            row["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, str(row["fileName"])))
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = base / image_path
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        row["image"] = str(image_path)
    return records


def verify_existing_dataset(dest: Path) -> dict:
    """Read earlier Notebook's 19/1 JSONL without changing its split or captions."""
    train = load_records(dest / "train.jsonl")
    val = load_records(dest / "validation.jsonl")
    if not train or not val:
        raise ValueError("Both training and validation sets must be non-empty")
    train_ids = [r["id"] for r in train]
    val_ids = [r["id"] for r in val]
    if len(train_ids + val_ids) != len(set(train_ids + val_ids)):
        raise ValueError("Duplicate IDs across the split")
    if (dest / "baseline_prediction.json").exists():
        baseline = json.loads((dest / "baseline_prediction.json").read_text(encoding="utf-8"))
        if Path(baseline["image"]).resolve() != Path(val[0]["image"]).resolve():
            raise ValueError("Existing baseline image is not the held-out validation image")
        if baseline.get("reference_caption") != val[0]["caption"]:
            raise ValueError("Baseline reference caption does not match validation JSONL")
    manifest_path = dest / "split_manifest.json"
    if not manifest_path.exists():
        info = {
            "legacy_imported": True,
            "note": "Original Notebook split preserved; do NOT regenerate from sorted Paimon IDs",
            "train_ids": train_ids,
            "validation_ids": val_ids,
        }
        manifest_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Verified original split: train={len(train)}, validation={len(val)}, manifest={manifest_path}")
    return {"train": len(train), "validation": len(val)}


def main() -> None:
    from multimodal_data_processing.paimon_io import connect_catalog, read_table
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=settings.dataset_dir)
    parser.add_argument("--val-ratio", type=float, default=settings.val_ratio)
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--verify-existing", action="store_true", help="Preserve prior Notebook JSONL/baseline; do not query Paimon")
    args = parser.parse_args()
    if args.verify_existing:
        verify_existing_dataset(args.output_dir)
        return
    df = read_table(connect_catalog().get_table(settings.enriched_table))
    good = select_valid_rows(df)
    train, val = split_records(good, args.val_ratio, args.seed)
    print("Raw rows:", len(df), "usable:", len(good))
    print("Training Dataset Statistics Overview:", dataset_statistics(train))
    print("Validation samples:", len(val))
    write_dataset(train, val, args.output_dir, args.seed)

if __name__ == "__main__":
    main()
