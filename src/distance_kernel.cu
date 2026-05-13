#include "distance_kernel.cuh"

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                    \
    do {                                                                    \
        cudaError_t _e = (expr);                                            \
        if (_e != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, cudaGetErrorString(_e));       \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

namespace {

// One block per row. Each thread accumulates a partial sum of squared diffs
// over a strided slice of the row, then the block tree-reduces in shared
// memory to a single distance value.
//
// Shared-memory layout (single extern array):
//   [0 .. dim)                        staged copy of the query
//   [dim .. dim + blockDim.x)         per-thread reduction scratch
__global__ void l2_squared_kernel(
    const float* __restrict__ embeddings,
    const float* __restrict__ query,
    float* __restrict__ distances,
    int n,
    int dim)
{
    extern __shared__ float smem[];
    float* squery   = smem;
    float* sreduce  = smem + dim;

    const int row = blockIdx.x;
    if (row >= n) return;

    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;

    for (int i = tid; i < dim; i += bdim) {
        squery[i] = query[i];
    }
    __syncthreads();

    const float* row_ptr = embeddings + static_cast<size_t>(row) * dim;
    float partial = 0.0f;
    for (int i = tid; i < dim; i += bdim) {
        float diff = row_ptr[i] - squery[i];
        partial += diff * diff;
    }
    sreduce[tid] = partial;
    __syncthreads();

    for (int s = bdim >> 1; s > 0; s >>= 1) {
        if (tid < s) sreduce[tid] += sreduce[tid + s];
        __syncthreads();
    }

    if (tid == 0) distances[row] = sreduce[0];
}

}  // namespace

DistanceKernel::DistanceKernel(int n_rows, int dim, int threads_per_block)
    : n_rows_(n_rows),
      dim_(dim),
      threads_per_block_(threads_per_block),
      d_embeddings_(nullptr),
      d_query_(nullptr),
      d_distances_(nullptr)
{
    // Tree reduction below assumes a power-of-two block size.
    if (threads_per_block_ <= 0 ||
        (threads_per_block_ & (threads_per_block_ - 1)) != 0) {
        std::fprintf(stderr,
                     "DistanceKernel: threads_per_block must be a positive "
                     "power of two (got %d)\n",
                     threads_per_block_);
        std::exit(1);
    }

    const size_t emb_bytes =
        static_cast<size_t>(n_rows_) * dim_ * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_embeddings_, emb_bytes));
    CUDA_CHECK(cudaMalloc(&d_query_, dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_distances_, n_rows_ * sizeof(float)));
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
}

DistanceKernel::~DistanceKernel() {
    cudaFree(d_embeddings_);
    cudaFree(d_query_);
    cudaFree(d_distances_);
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

void DistanceKernel::upload_embeddings(const float* h_embeddings) {
    const size_t bytes =
        static_cast<size_t>(n_rows_) * dim_ * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_embeddings_, h_embeddings, bytes,
                          cudaMemcpyHostToDevice));
}

float DistanceKernel::compute(const float* h_query, float* h_distances) {
    CUDA_CHECK(cudaMemcpy(d_query_, h_query,
                          static_cast<size_t>(dim_) * sizeof(float),
                          cudaMemcpyHostToDevice));

    const size_t shmem_bytes =
        static_cast<size_t>(dim_ + threads_per_block_) * sizeof(float);

    CUDA_CHECK(cudaEventRecord(start_));
    l2_squared_kernel<<<n_rows_, threads_per_block_, shmem_bytes>>>(
        d_embeddings_, d_query_, d_distances_, n_rows_, dim_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));

    CUDA_CHECK(cudaMemcpy(h_distances, d_distances_,
                          static_cast<size_t>(n_rows_) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return ms;
}
