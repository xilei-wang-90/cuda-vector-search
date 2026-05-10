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

Stages 1–3 are wired up; the on-device Top-K reduction is next (today the
top-K is taken host-side after the distances come back).

## Repo layout

```
scripts/embed_dataset.py     # Stage 1: fetch + embed + serialize
src/check_roundtrip.cu       # Stage 2: H2D / D2H sanity check
src/distance_kernel.cuh      # Stage 3: DistanceKernel class header
src/distance_kernel.cu       # Stage 3: kernel + class implementation
src/compute_distances.cu     # Stage 3: driver — load, score, top-K
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

## Stage 3: distance kernel

A custom CUDA kernel scores every embedding against a query vector in
parallel, picking up where stage 2 leaves off (right after the H2D copy).

The kernel computes squared L2 distance:

```
d2[r] = sum_i (E[r, i] - q[i])^2
```

For the L2-normalised embeddings produced by stage 1 this is monotonically
equivalent to cosine similarity — `||a - q||^2 = 2 - 2·dot(a, q)` — so the
ranking is the same as the cosine pipeline mentioned above.

The implementation lives in its own translation unit:

- `src/distance_kernel.cuh` — `DistanceKernel` class interface.
- `src/distance_kernel.cu` — the `__global__ l2_squared_kernel` and the
  class methods that own device buffers, upload the matrix, launch the
  kernel, and copy results back. The matrix is uploaded once and the
  buffer can be reused across many queries.
- `src/compute_distances.cu` — driver that loads `vectors.fp32.bin`,
  picks one row as the query, runs the kernel, and prints timing plus
  the top-K nearest rows by distance.

### Kernel design

- One CUDA block per row (10,000 blocks for the default dataset).
- 128 threads per block; each thread accumulates a partial sum over a
  strided slice of the row, then the block tree-reduces in shared memory
  to a single distance value.
- The query vector is staged into shared memory once per block before the
  reduction starts, so each thread reads it from on-chip memory rather
  than global memory.
- Top-K is currently a host-side `std::partial_sort` over the returned
  distance vector; the on-device reduction is the next pipeline stage.

### Build

The kernel binary is built by the same `make` invocation as stage 2:

```bash
make
```

### Run

```bash
./build/compute_distances                                                  # uses defaults
./build/compute_distances data/embeddings/vectors.fp32.bin 384 0 5         # path dim query_idx top_k
```

Sample output on an RTX 3060:

```
[load]  data/embeddings/vectors.fp32.bin
        bytes=15360000  rows=10000  dim=384
[gpu]   device 0: NVIDIA GeForce RTX 3060 Laptop GPU  (sm_86, 6.44 GB)
[xfer]  H2D embeddings: 3.105 ms (4.95 GB/s)
[kernel] l2_squared_kernel  blocks=10000  threads/block=128  shmem=2048 B
         1.258 ms  (9.15 GFLOP/s, 3 flops/elem)
[query] index=0  self-distance=0.000000 (sanity: ~0)
[top-5 nearest by squared L2]
   0: row=0       d2=0.000000
   1: row=9       d2=0.115367
   2: row=3817    d2=0.850716
   3: row=8800    d2=0.913705
   4: row=3813    d2=0.984546
```

The self-distance line (`row=query_idx, d2≈0`) is a built-in sanity check:
the kernel must rank the query as the nearest match to itself with a
distance of zero. The rest of the ranking matches the numpy reference
`np.argsort(np.sum((vecs - vecs[0])**2, axis=1))` to six decimals.

## Data and licensing

The `data/` directory is gitignored — re-run the script to regenerate it. The
ag_news dataset is distributed under its own license (see the dataset card on
Hugging Face). Embeddings are derived from that text via the listed
sentence-transformers model.
