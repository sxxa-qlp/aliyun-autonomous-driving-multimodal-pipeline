"""Select 20 adjacent CAM_FRONT images (same driving log) and preview them."""
from __future__ import annotations
import argparse
import io
import re
from pathlib import Path
from PIL import Image
from config import settings

_TS = re.compile(r"__CAM_FRONT__(\d+)\.(?:jpg|jpeg)$", re.IGNORECASE)


def list_camfront_images(dataset_dir: Path, source: str = "sweeps") -> list[Path]:
    """Try preferred nuScenes sweeps/CAM_FRONT, fall back to samples/CAM_FRONT."""
    preferred = dataset_dir / source / "CAM_FRONT"
    paths = sorted(preferred.glob("*.jpg")) if preferred.is_dir() else []
    if not paths:
        matches = sorted(dataset_dir.rglob(f"{source}/CAM_FRONT/*.jpg"))
        paths = matches
    if not paths:
        print(f"No {source}/CAM_FRONT images; check archive layout or try --source samples")
    return paths


def select_adjacent(paths: list[Path], limit: int = 20) -> list[Path]:
    """Stay within one filename log prefix and order by camera timestamp."""
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in paths:
        match = _TS.search(path.name)
        if match:
            prefix = path.name.split("__CAM_FRONT__", 1)[0]
            groups.setdefault(prefix, []).append((int(match.group(1)), path))
    candidates = sorted(groups.values(), key=lambda v: (-len(v), min(x[0] for x in v)))
    for group in candidates:
        if len(group) >= limit:
            return [p for _, p in sorted(group)[:limit]]
    raise ValueError(f"Need {limit} images from one CAM_FRONT log; found group lengths {[len(g) for g in candidates[:5]]}")


def read_images(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        blob = path.read_bytes()
        with Image.open(io.BytesIO(blob)) as image:
            image.verify()
        rows.append({"fileName": path.name, "picture": blob})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=settings.nu_scenes_dir)
    parser.add_argument("--source", choices=["sweeps", "samples"], default="sweeps")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    chosen = select_adjacent(list_camfront_images(args.dataset_dir, args.source), args.limit)
    for path in chosen:
        print(path.name, f"{path.stat().st_size/1024:.0f} KB")
    print("Selected", len(chosen), "images from one camera log")

if __name__ == "__main__":
    main()
