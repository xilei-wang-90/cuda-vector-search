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
    DistanceKernel(int n_rows, int dim, int threads_per_block = 128);
    ~DistanceKernel();

    DistanceKernel(const DistanceKernel&) = delete;
    DistanceKernel& operator=(const DistanceKernel&) = delete;

    // Copy the row-major (n_rows, dim) float32 matrix to the device.
    void upload_embeddings(const float* h_embeddings);

    // Upload `h_query` (dim floats), launch the kernel, copy the (n_rows,)
    // distance vector back into `h_distances`. Returns kernel-only elapsed
    // time in milliseconds (excludes H2D/D2H).
    float compute(const float* h_query, float* h_distances);

    int n_rows() const { return n_rows_; }
    int dim() const { return dim_; }
    int threads_per_block() const { return threads_per_block_; }

private:
    int n_rows_;
    int dim_;
    int threads_per_block_;
    float* d_embeddings_;
    float* d_query_;
    float* d_distances_;
    cudaEvent_t start_;
    cudaEvent_t stop_;
};
