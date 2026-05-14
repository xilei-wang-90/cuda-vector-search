#pragma once

#include <cuda_runtime.h>

// GPU-side squared-L2 distance: for each row in an (N, D) embedding matrix,
// compute sum_i (row[i] - query[i])^2.
//
// For L2-normalised vectors (which the ingestion stage produces by default)
// this is monotonically equivalent to cosine similarity:
//   ||a - q||^2 = 2 - 2 * dot(a, q)
// so ranking by ascending squared-L2 == ranking by descending cosine.
//
// The class owns its device buffers so the embedding matrix can be uploaded
// once and reused across many queries.
class DistanceKernel {
public:
    // Compile-time upper bound on K for the on-device top-K kernel. The
    // kernel allocates per-thread sorted scratch of this size, so the bound
    // also caps how much shared memory and register/local storage the
    // reduction can use.
    static constexpr int kMaxTopK = 32;

    DistanceKernel(int n_rows,
                   int dim,
                   int threads_per_block = 128,
                   int topk_threads_per_block = 128);
    ~DistanceKernel();

    DistanceKernel(const DistanceKernel&) = delete;
    DistanceKernel& operator=(const DistanceKernel&) = delete;

    // Copy the row-major (n_rows, dim) float32 matrix to the device.
    void upload_embeddings(const float* h_embeddings);

    // Upload `h_query` (dim floats), launch the distance kernel, copy the
    // (n_rows,) distance vector back into `h_distances`. Returns kernel-only
    // elapsed time in milliseconds (excludes H2D/D2H).
    float compute(const float* h_query, float* h_distances);

    // Stage 4: full GPU pipeline. Upload query, run the distance kernel,
    // then run the on-device top-K reduction. Only k (distance, index) pairs
    // come back to the host — the (n_rows,) distance vector never leaves
    // device memory. `k` must satisfy 1 <= k <= kMaxTopK and k <= n_rows.
    //
    // On return:
    //   h_topk_dists[0..k) — distances in ascending order
    //   h_topk_idxs [0..k) — row indices, aligned with h_topk_dists
    //   *ms_distance       — distance-kernel time in ms (if non-null)
    //   *ms_topk           — top-K-kernel time in ms (if non-null)
    // The return value is the combined kernel time (ms_distance + ms_topk).
    float compute_topk(const float* h_query,
                       int          k,
                       float*       h_topk_dists,
                       int*         h_topk_idxs,
                       float*       ms_distance = nullptr,
                       float*       ms_topk     = nullptr);

    int n_rows() const { return n_rows_; }
    int dim() const { return dim_; }
    int threads_per_block() const { return threads_per_block_; }
    int topk_threads_per_block() const { return topk_threads_per_block_; }

private:
    int n_rows_;
    int dim_;
    int threads_per_block_;
    int topk_threads_per_block_;
    float* d_embeddings_;
    float* d_query_;
    float* d_distances_;
    float* d_topk_dists_;
    int*   d_topk_idxs_;
    cudaEvent_t start_;
    cudaEvent_t stop_;
    cudaEvent_t topk_start_;
    cudaEvent_t topk_stop_;
};
