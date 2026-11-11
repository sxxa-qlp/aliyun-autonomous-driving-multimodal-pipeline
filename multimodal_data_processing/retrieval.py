"""Exact cosine Top-K for image→image and text→image, no vector database."""
from __future__ import annotations
import argparse
import numpy as np
from config import settings
from .embedding import image_embedding, text_embedding


def cosine_topk(vectors: np.ndarray, query_vector: np.ndarray, k: int = 5,
                exclude_index: int | None = None) -> list[tuple[int, float]]:
    values = np.asarray(vectors, dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32)
    if values.ndim != 2 or query.shape != (values.shape[1],):
        raise ValueError("Embedding dimensions differ")
    norms = np.linalg.norm(values, axis=1)
    q_norm = np.linalg.norm(query)
    if np.any(norms == 0) or q_norm == 0:
        raise ValueError("Zero-norm vector")
    scores = (values / norms[:, None]) @ (query / q_norm)
    if exclude_index is not None:
        scores[exclude_index] = -np.inf
    order = np.argsort(-scores, kind="stable")
    return [(int(i), float(scores[i])) for i in order[:min(k, len(order))] if np.isfinite(scores[i])]


def main() -> None:
    from .paimon_io import connect_catalog, read_table
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--text", help="Text-to-image query")
    modes.add_argument("--image-index", type=int, help="Image-to-image query index")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    df = read_table(connect_catalog().get_table(settings.enriched_table)).reset_index(drop=True)
    if df.empty or df["vector"].isna().any():
        raise RuntimeError("Missing image vectors; run enrichment backfill")
    matrix = np.stack([np.asarray(v, dtype=np.float32) for v in df["vector"]])
    if args.text is not None:
        qvec, excluded = text_embedding(args.text), None
    else:
        if not 0 <= args.image_index < len(df):
            parser.error("--image-index out of range")
        qvec = image_embedding(bytes(df.iloc[args.image_index]["picture"]))
        excluded = args.image_index
    for rank, (i, score) in enumerate(cosine_topk(matrix, qvec, args.top_k, excluded), 1):
        print(f"{rank}. cosine={score:.4f} file={df.loc[i, 'fileName']} caption={df.loc[i, 'caption']}")

if __name__ == "__main__":
    main()
