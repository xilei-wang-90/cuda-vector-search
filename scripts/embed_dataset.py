"""Fetch AG News rows, embed them, and save raw float32 vectors for CUDA.

Outputs (under --output-dir, default ./data):
  raw/ag_news.jsonl                 one JSON object per row: id, text, label, label_name
  embeddings/vectors.fp32.bin       raw row-major float32, shape (N, D)
  embeddings/metadata.json          {n, d, model, dataset, split, normalized, dtype}

The .bin layout is what the CUDA pipeline ingests: a contiguous float* of length N*D.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

AG_NEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-docs", type=int, default=10_000)
    p.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--dataset", type=str, default="ag_news")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    p.add_argument("--no-normalize", action="store_true",
                   help="Skip L2 normalization. Default is normalized so cosine == dot product.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}  model={args.model}  num_docs={args.num_docs}")

    raw_dir = args.output_dir / "raw"
    emb_dir = args.output_dir / "embeddings"
    raw_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    ds = load_dataset(args.dataset, split=args.split)
    if args.num_docs > len(ds):
        raise ValueError(f"requested {args.num_docs} but split has {len(ds)}")
    ds = ds.select(range(args.num_docs))
    texts = ds["text"]
    labels = ds["label"]
    print(f"[load]  {len(texts)} rows from {args.dataset}/{args.split} in {time.perf_counter()-t0:.2f}s")

    raw_path = raw_dir / "ag_news.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for i, (text, label) in enumerate(zip(texts, labels)):
            f.write(json.dumps({
                "id": i,
                "text": text,
                "label": int(label),
                "label_name": AG_NEWS_LABELS[label],
            }, ensure_ascii=False) + "\n")
    print(f"[write] {raw_path}  ({raw_path.stat().st_size/1e6:.2f} MB)")

    t0 = time.perf_counter()
    model = SentenceTransformer(args.model, device=device)
    dim = model.get_sentence_embedding_dimension()
    print(f"[model] dim={dim}  loaded in {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=not args.no_normalize,
    ).astype(np.float32, copy=False)
    enc_secs = time.perf_counter() - t0
    print(f"[encode] {vectors.shape} float32 in {enc_secs:.2f}s "
          f"({len(texts)/enc_secs:.0f} docs/s)")

    if not vectors.flags["C_CONTIGUOUS"]:
        vectors = np.ascontiguousarray(vectors)
    bin_path = emb_dir / "vectors.fp32.bin"
    vectors.tofile(bin_path)
    print(f"[write] {bin_path}  ({bin_path.stat().st_size/1e6:.2f} MB)")

    meta = {
        "n": int(vectors.shape[0]),
        "d": int(vectors.shape[1]),
        "dtype": "float32",
        "layout": "row-major",
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "normalized": not args.no_normalize,
        "labels": AG_NEWS_LABELS,
    }
    meta_path = emb_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[write] {meta_path}")

    norms = np.linalg.norm(vectors, axis=1)
    print(f"[stats] norm mean={norms.mean():.4f} std={norms.std():.4f} "
          f"min={norms.min():.4f} max={norms.max():.4f}")
    print("[done]")


if __name__ == "__main__":
    main()
