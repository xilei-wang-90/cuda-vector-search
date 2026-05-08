# CUDA Vector Search

A high-performance, custom vector search algorithm built in CUDA C++. It processes
document embeddings directly on the GPU, bypassing sequential CPU-side scoring. By
computing distances across the entire corpus in parallel and reducing on-device, it
returns the closest matches without round-tripping the dataset back to host memory.
Tuned for consumer hardware — easily handles 10,000+ high-dimensional vectors on an
RTX 3060.

## Pipeline

1. **Data ingestion** — Python script pulls documents from a Hugging Face dataset
   and emits float32 embeddings.
2. **Device transfer** — efficient host-to-device (H2D) copy of the embedding
   matrix and query vector.
3. **Parallel compute** — custom CUDA kernels compute cosine similarity (or other
   distance metrics) across all rows in parallel.
4. **Top-K reduction** — on-device sort/heap reduction returns only the top 5
   nearest neighbours instead of shipping the full score array back to the host.

The CUDA stages are coming next; this repo currently ships the ingestion stage.

## Repo layout

```
scripts/embed_dataset.py   # Stage 1: fetch + embed + serialize
requirements.txt           # Python dependencies for the ingestion stage
data/                      # Generated locally; gitignored
  raw/ag_news.jsonl
  embeddings/vectors.fp32.bin
  embeddings/metadata.json
```

## Stage 1: ingest + embed

The script pulls 10,000 rows from
[ag_news](https://huggingface.co/datasets/ag_news), embeds each headline+body with
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
(384-dim), L2-normalises so cosine similarity reduces to a dot product, and
serialises the matrix as a flat float32 binary that the CUDA pipeline mmaps
directly.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU acceleration is automatic if a CUDA-enabled PyTorch build is present
(`torch.cuda.is_available()`); otherwise the script falls back to CPU.

### Run

```bash
python3 scripts/embed_dataset.py
```

Useful flags:

| Flag             | Default                                          | Notes                                |
| ---------------- | ------------------------------------------------ | ------------------------------------ |
| `--num-docs`     | `10000`                                          | rows pulled from the train split     |
| `--model`        | `sentence-transformers/all-MiniLM-L6-v2`         | any sentence-transformers model      |
| `--batch-size`   | `128`                                            | encoder batch size                   |
| `--output-dir`   | `data`                                           | base directory for outputs           |
| `--no-normalize` | off                                              | skip L2 normalisation                |

On an RTX 3060 the 10k run finishes in ~6s of encode time (~1700 docs/s).

### Outputs

```
data/raw/ag_news.jsonl              # one row per line: {id, text, label, label_name}
data/embeddings/vectors.fp32.bin    # row-major float32, shape (N, D), no header
data/embeddings/metadata.json       # {n, d, dtype, layout, model, normalized, ...}
```

The binary is a contiguous `float*` of length `N*D` — the CUDA loader can map it
directly into device memory using the `n` / `d` values from `metadata.json`. With
defaults, that is `10000 × 384 × 4 = 15,360,000` bytes.

### Sanity check

```python
import json, numpy as np
meta = json.load(open("data/embeddings/metadata.json"))
vecs = np.fromfile("data/embeddings/vectors.fp32.bin", dtype=np.float32).reshape(meta["n"], meta["d"])
sims = vecs @ vecs[0]                    # cosine, since vectors are L2-normalised
print(np.argsort(-sims)[:5])             # indices of top-5 nearest neighbours of doc 0
```

For the default ag_news run, the top neighbours of doc 0 are all Wall Street
stories from the Business label — confirming the embeddings carry semantic signal.

## Data and licensing

The `data/` directory is gitignored — re-run the script to regenerate it. The
ag_news dataset is distributed under its own license (see the dataset card on
Hugging Face). Embeddings are derived from that text via the listed
sentence-transformers model.
