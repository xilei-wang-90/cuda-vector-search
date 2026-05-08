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

Stages 1 and 2 are wired up; the parallel-compute and Top-K kernels come next.

## Repo layout

```
scripts/embed_dataset.py     # Stage 1: fetch + embed + serialize
src/check_roundtrip.cu       # Stage 2: H2D / D2H sanity check
Makefile                     # builds CUDA binaries into build/
pyproject.toml               # Python dependencies (Poetry-managed)
poetry.lock                  # Pinned dependency versions
data/                        # Generated locally; gitignored
  raw/ag_news.jsonl
  embeddings/vectors.fp32.bin
  embeddings/metadata.json
build/                       # CUDA build outputs; gitignored
```

## Stage 1: ingest + embed

The script pulls 10,000 rows from
[ag_news](https://huggingface.co/datasets/ag_news), embeds each headline+body with
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
(384-dim), L2-normalises so cosine similarity reduces to a dot product, and
serialises the matrix as a flat float32 binary that the CUDA pipeline mmaps
directly.

### Setup

Dependencies are managed with [Poetry](https://python-poetry.org/). The repo
ships a `poetry.toml` that pins the virtualenv to `.venv/` in the project root.

```bash
poetry install --no-root
```

GPU acceleration is automatic if a CUDA-enabled PyTorch build is present
(`torch.cuda.is_available()`); otherwise the script falls back to CPU.

### Run

```bash
poetry run python3 scripts/embed_dataset.py
```

(or `poetry shell` once and drop the `poetry run` prefix.)

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

## Stage 2: device round-trip

A minimal CUDA program that confirms the embedding bytes survive a host →
device → host round-trip. It's the simplest possible test that the file loader,
`cudaMalloc` allocation, and `cudaMemcpy` directions are all wired up correctly
before any real kernel work begins.

### Build

Requires the CUDA toolkit (`nvcc`). The Makefile defaults to `sm_86` (RTX 30xx);
override `CUDA_ARCH` for other GPUs (e.g. `CUDA_ARCH=89` for RTX 40xx).

```bash
make
```

### Run

```bash
./build/check_roundtrip                                        # uses defaults
./build/check_roundtrip data/embeddings/vectors.fp32.bin 384   # explicit args
```

Sample output on an RTX 3060:

```
[load]  data/embeddings/vectors.fp32.bin
        bytes=15360000  floats=3840000  rows=10000  dim=384
[gpu]   device 0: NVIDIA GeForce RTX 3060 Laptop GPU  (sm_86, 6.44 GB)
[alloc] device buffer @ 0x506800000 (15.36 MB)
[xfer]  H2D: 7.331 ms (2.10 GB/s)   D2H: 2.427 ms (6.33 GB/s)
[check] OK -- 15360000 bytes round-tripped intact
[sample] first 5 floats: 0.007439 0.028562 0.041096 0.105001 0.023282
```

The check uses `memcmp` on the full buffer; any single bit flip would fail it.
Exit code is `0` on success, `1` otherwise.

## Data and licensing

The `data/` directory is gitignored — re-run the script to regenerate it. The
ag_news dataset is distributed under its own license (see the dataset card on
Hugging Face). Embeddings are derived from that text via the listed
sentence-transformers model.
