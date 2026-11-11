"""DashScope-native multimodal embeddings for images and text (1152-dim in demo).

Image and text use separate vectors in a shared embedding space. This is NOT
Qwen2-VL SFT input. The embedding API is not OpenAI-compatible in this workflow.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import requests
from config import settings
from .caption_metadata import image_to_data_url


def _embed(content: dict) -> np.ndarray:
    if not settings.embedding_url or not settings.embedding_key:
        raise RuntimeError("Set EMBEDDING_API_URL and EMBEDDING_API_KEY")
    response = requests.post(
        settings.embedding_url,
        headers={"Authorization": f"Bearer {settings.embedding_key}", "Content-Type": "application/json"},
        json={"model": settings.embedding_model, "input": {"contents": [content]}},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code"):
        raise RuntimeError(f"Embedding API error code: {payload['code']}; {payload.get('message', '')}")
    entries = payload.get("output", {}).get("embeddings", [])
    if len(entries) != 1:
        raise ValueError(f"Expected one embedding, received {len(entries)}")
    vector = np.asarray(entries[0]["embedding"], dtype=np.float32)
    if vector.shape != (settings.vector_dim,) or not np.isfinite(vector).all():
        raise ValueError(f"Invalid embedding: {vector.shape}, expected {settings.vector_dim}")
    return vector


def _cached(key: bytes, kind: str, produce) -> np.ndarray:
    settings.embedding_cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(settings.embedding_model.encode() + kind.encode() + key).hexdigest()
    path = settings.embedding_cache / (digest + ".npy")
    if path.is_file():
        vector = np.load(path, allow_pickle=False)
    else:
        vector = produce()
        temp = path.with_suffix(".tmp.npy")
        np.save(temp, vector)
        temp.replace(path)
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (settings.vector_dim,) or not np.isfinite(vector).all():
        raise ValueError("Invalid cached vector")
    return vector


def image_embedding(picture: bytes) -> np.ndarray:
    raw = bytes(picture)
    return _cached(raw, "image", lambda: _embed({"image": image_to_data_url(raw)}))


def text_embedding(query: str) -> np.ndarray:
    if not query.strip():
        raise ValueError("Query must not be empty")
    return _cached(query.encode("utf-8"), "text", lambda: _embed({"text": query}))


def main() -> None:
    from .paimon_io import connect_catalog, read_table
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Generate one text embedding")
    args = parser.parse_args()
    if args.text:
        print("Vector shape:", text_embedding(args.text).shape)
    else:
        df = read_table(connect_catalog().get_table(settings.enriched_table))
        print("Vector shape:", image_embedding(df.iloc[0]["picture"]).shape)

if __name__ == "__main__":
    main()
