#include "distance_kernel.cuh"

#include <math_constants.h>

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

// Single-block on-device top-K reduction over the distance vector.
//
// Each thread scans a strided slice of `distances` and maintains a length-K
// sorted (ascending) list of (distance, index) pairs in local arrays via
// insertion. After the scan the per-thread lists are written into shared
// memory and pairwise-merged in a log2(blockDim.x) tree until thread 0 owns
// the global top-K. The block is launched with a power-of-two thread count.
//
// MAX_K is a compile-time upper bound that sizes the per-thread arrays; the
// runtime `k` may be any value in [1, MAX_K].
//
// Shared-memory layout (single extern array, byte view):
//   [0 .. blockDim.x*MAX_K*sizeof(float))           merge scratch: distances
//   [blockDim.x*MAX_K*sizeof(float) .. +ints)       merge scratch: indices
template <int MAX_K>
__global__ void topk_kernel(
    const float* __restrict__ distances,
    int n,
    int k,
    float* __restrict__ out_dists,
    int*   __restrict__ out_idxs)
{
    extern __shared__ unsigned char raw_smem[];
    float* s_dists = reinterpret_cast<float*>(raw_smem);
    int*   s_idxs  = reinterpret_cast<int*>(s_dists + blockDim.x * MAX_K);

    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;

    float local_d[MAX_K];
    int   local_i[MAX_K];
    #pragma unroll
    for (int j = 0; j < MAX_K; ++j) {
        local_d[j] = CUDART_INF_F;
        local_i[j] = -1;
    }

    // Strided scan: maintain a length-k sorted prefix of local_d/local_i.
    for (int i = tid; i < n; i += bdim) {
        float d = distances[i];
        if (d < local_d[k - 1]) {
            int pos = k - 1;
            while (pos > 0 && local_d[pos - 1] > d) {
                local_d[pos] = local_d[pos - 1];
                local_i[pos] = local_i[pos - 1];
                --pos;
            }
            local_d[pos] = d;
            local_i[pos] = i;
        }
    }

    // Stage each thread's local top-k into shared memory for the merge tree.
    for (int j = 0; j < k; ++j) {
        s_dists[tid * MAX_K + j] = local_d[j];
        s_idxs [tid * MAX_K + j] = local_i[j];
    }
    __syncthreads();

    // Tree-merge: at each step half the threads merge their length-k list
    // with the matching partner's into a single sorted length-k list. After
    // log2(bdim) steps the merged result lives at slot 0.
    for (int s = bdim >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float a_d[MAX_K];
            int   a_i[MAX_K];
            for (int j = 0; j < k; ++j) {
                a_d[j] = s_dists[tid * MAX_K + j];
                a_i[j] = s_idxs [tid * MAX_K + j];
            }
            const int base_b = (tid + s) * MAX_K;

            float merged_d[MAX_K];
            int   merged_i[MAX_K];
            int ia = 0, ib = 0;
            for (int j = 0; j < k; ++j) {
                float bd = (ib < k) ? s_dists[base_b + ib] : CUDART_INF_F;
                if (ia < k && a_d[ia] <= bd) {
                    merged_d[j] = a_d[ia];
                    merged_i[j] = a_i[ia];
                    ++ia;
                } else {
                    merged_d[j] = bd;
                    merged_i[j] = s_idxs[base_b + ib];
                    ++ib;
                }
            }
            for (int j = 0; j < k; ++j) {
                s_dists[tid * MAX_K + j] = merged_d[j];
                s_idxs [tid * MAX_K + j] = merged_i[j];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int j = 0; j < k; ++j) {
            out_dists[j] = s_dists[j];
            out_idxs [j] = s_idxs [j];
        }
    }
}

}  // namespace

DistanceKernel::DistanceKernel(int n_rows,
                               int dim,
                               int threads_per_block,
                               int topk_threads_per_block)
    : n_rows_(n_rows),
      dim_(dim),
      threads_per_block_(threads_per_block),
      topk_threads_per_block_(topk_threads_per_block),
      d_embeddings_(nullptr),
      d_query_(nullptr),
      d_distances_(nullptr),
      d_topk_dists_(nullptr),
      d_topk_idxs_(nullptr)
{
    // Tree reductions below assume a power-of-two block size.
    auto is_pow2 = [](int x) { return x > 0 && (x & (x - 1)) == 0; };
    if (!is_pow2(threads_per_block_)) {
        std::fprintf(stderr,
                     "DistanceKernel: threads_per_block must be a positive "
                     "power of two (got %d)\n",
                     threads_per_block_);
        std::exit(1);
    }
    if (!is_pow2(topk_threads_per_block_)) {
        std::fprintf(stderr,
                     "DistanceKernel: topk_threads_per_block must be a "
                     "positive power of two (got %d)\n",
                     topk_threads_per_block_);
        std::exit(1);
    }

    const size_t emb_bytes =
        static_cast<size_t>(n_rows_) * dim_ * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_embeddings_, emb_bytes));
    CUDA_CHECK(cudaMalloc(&d_query_, dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_distances_, n_rows_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topk_dists_, kMaxTopK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topk_idxs_,  kMaxTopK * sizeof(int)));
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
    CUDA_CHECK(cudaEventCreate(&topk_start_));
    CUDA_CHECK(cudaEventCreate(&topk_stop_));
}

DistanceKernel::~DistanceKernel() {
    cudaFree(d_embeddings_);
    cudaFree(d_query_);
    cudaFree(d_distances_);
    cudaFree(d_topk_dists_);
    cudaFree(d_topk_idxs_);
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
    cudaEventDestroy(topk_start_);
    cudaEventDestroy(topk_stop_);
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

float DistanceKernel::compute_topk(const float* h_query,
                                   int          k,
                                   float*       h_topk_dists,
                                   int*         h_topk_idxs,
                                   float*       ms_distance,
                                   float*       ms_topk)
{
    if (k <= 0 || k > kMaxTopK || k > n_rows_) {
        std::fprintf(stderr,
                     "DistanceKernel::compute_topk: k out of range "
                     "(got %d, allowed 1..%d and <= n_rows=%d)\n",
                     k, kMaxTopK, n_rows_);
        std::exit(1);
    }

    CUDA_CHECK(cudaMemcpy(d_query_, h_query,
                          static_cast<size_t>(dim_) * sizeof(float),
                          cudaMemcpyHostToDevice));

    const size_t dist_shmem_bytes =
        static_cast<size_t>(dim_ + threads_per_block_) * sizeof(float);
    const size_t topk_shmem_bytes =
        static_cast<size_t>(topk_threads_per_block_) * kMaxTopK *
        (sizeof(float) + sizeof(int));

    CUDA_CHECK(cudaEventRecord(start_));
    l2_squared_kernel<<<n_rows_, threads_per_block_, dist_shmem_bytes>>>(
        d_embeddings_, d_query_, d_distances_, n_rows_, dim_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_));

    CUDA_CHECK(cudaEventRecord(topk_start_));
    topk_kernel<kMaxTopK>
        <<<1, topk_threads_per_block_, topk_shmem_bytes>>>(
            d_distances_, n_rows_, k, d_topk_dists_, d_topk_idxs_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(topk_stop_));
    CUDA_CHECK(cudaEventSynchronize(topk_stop_));

    float ms_d = 0.0f, ms_t = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_d, start_, stop_));
    CUDA_CHECK(cudaEventElapsedTime(&ms_t, topk_start_, topk_stop_));
    if (ms_distance) *ms_distance = ms_d;
    if (ms_topk)     *ms_topk     = ms_t;

    CUDA_CHECK(cudaMemcpy(h_topk_dists, d_topk_dists_,
                          static_cast<size_t>(k) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_topk_idxs, d_topk_idxs_,
                          static_cast<size_t>(k) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    return ms_d + ms_t;
}
