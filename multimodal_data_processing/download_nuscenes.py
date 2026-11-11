"""Download/extract nuScenes mini with explicit confirmation of dataset terms.

The public dataset is not redistributed by this repository. Respect nuScenes terms.
"""
from __future__ import annotations
import argparse
import tarfile
from pathlib import Path
import requests
from config import settings

URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"


def download_archive(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print("Archive exists, reuse:", target)
        return target
    partial = target.with_name(target.name + ".part")
    with requests.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        with partial.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    partial.replace(target)
    return target


def extract_safe(archive: Path, out_dir: Path) -> None:
    """Reject links, devices and archive traversal paths before extracting."""
    root = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        for member in members:
            target = (out_dir / member.name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Archive path traversal: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported tar member: {member.name}")
        tf.extractall(out_dir, members=members, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-dataset-terms", action="store_true", help="Confirm you reviewed nuScenes access terms")
    parser.add_argument("--archive", type=Path, default=settings.work_dir / "v1.0-mini.tgz")
    parser.add_argument("--extract-to", type=Path, default=settings.nu_scenes_dir)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()
    if not args.accept_dataset_terms:
        parser.error("Review https://www.nuscenes.org/ and pass --accept-dataset-terms")
    path = download_archive(URL, args.archive)
    print("Archive:", path)
    if not args.skip_extract:
        extract_safe(path, args.extract_to)
        print("Extracted to:", args.extract_to)

if __name__ == "__main__":
    main()
