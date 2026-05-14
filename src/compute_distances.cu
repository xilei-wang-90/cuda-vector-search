// Stage 3 + 4: compute the squared-L2 distance from a query vector to every
// embedding in the matrix using a custom CUDA kernel, then reduce to the
// top-K nearest rows entirely on-device. The (N,) distance vector never
// leaves device memory — only K (distance, index) pairs come back.
//
// Layout matches stage 2: the binary at `path` is a contiguous row-major
// float32 matrix of shape (N, dim). One row is selected as the query;
// the rest are scored on-device by `DistanceKernel`.
//
// Usage:
//   ./build/compute_distances [path/to/vectors.fp32.bin] [dim] [query_idx] [top_k]
// Defaults: data/embeddings/vectors.fp32.bin, dim=384, query_idx=0, top_k=5.

#include "distance_kernel.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#define CUDA_CHECK(expr)                                                    \
    do {                                                                    \
        cudaError_t _e = (expr);                                            \
        if (_e != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                         __FILE__, __LINE__, cudaGetErrorString(_e));       \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

int main(int argc, char** argv) {
    const char* path     = (argc > 1) ? argv[1] : "data/embeddings/vectors.fp32.bin";
    const int   dim      = (argc > 2) ? std::atoi(argv[2]) : 384;
    const int   query_ix = (argc > 3) ? std::atoi(argv[3]) : 0;
    const int   top_k    = (argc > 4) ? std::atoi(argv[4]) : 5;

    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "open failed: %s\n", path);
        return 1;
    }
    const std::streamsize bytes = in.tellg();
    in.seekg(0, std::ios::beg);

    if (bytes <= 0 ||
        bytes % static_cast<std::streamsize>(sizeof(float)) != 0 ||
        dim <= 0) {
        std::fprintf(stderr, "bad file size or dim: bytes=%lld dim=%d\n",
                     static_cast<long long>(bytes), dim);
        return 1;
    }
    const size_t n_floats = static_cast<size_t>(bytes) / sizeof(float);
    if (n_floats % static_cast<size_t>(dim) != 0) {
        std::fprintf(stderr,
                     "file does not divide evenly by dim: floats=%zu dim=%d\n",
                     n_floats, dim);
        return 1;
    }
    const int rows = static_cast<int>(n_floats / static_cast<size_t>(dim));
    std::printf("[load]  %s\n        bytes=%lld  rows=%d  dim=%d\n",
                path, static_cast<long long>(bytes), rows, dim);

    if (query_ix < 0 || query_ix >= rows) {
        std::fprintf(stderr, "query_idx out of range: %d (rows=%d)\n",
                     query_ix, rows);
        return 1;
    }
    if (top_k <= 0 || top_k > rows || top_k > DistanceKernel::kMaxTopK) {
        std::fprintf(stderr,
                     "top_k out of range: %d (rows=%d, max=%d)\n",
                     top_k, rows, DistanceKernel::kMaxTopK);
        return 1;
    }

    std::vector<float> h_embeddings(n_floats);
    if (!in.read(reinterpret_cast<char*>(h_embeddings.data()), bytes)) {
        std::fprintf(stderr, "short read\n");
        return 1;
    }

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("[gpu]   device %d: %s  (sm_%d%d, %.2f GB)\n",
                dev, prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / 1.0e9);

    DistanceKernel kernel(rows, dim);

    cudaEvent_t e_h2d_start, e_h2d_end;
    CUDA_CHECK(cudaEventCreate(&e_h2d_start));
    CUDA_CHECK(cudaEventCreate(&e_h2d_end));
    CUDA_CHECK(cudaEventRecord(e_h2d_start));
    kernel.upload_embeddings(h_embeddings.data());
    CUDA_CHECK(cudaEventRecord(e_h2d_end));
    CUDA_CHECK(cudaEventSynchronize(e_h2d_end));
    float ms_h2d = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, e_h2d_start, e_h2d_end));
    const double mb_emb = bytes / 1.0e6;
    std::printf("[xfer]  H2D embeddings: %.3f ms (%.2f GB/s)\n",
                ms_h2d, mb_emb / ms_h2d);

    const float* h_query = h_embeddings.data() +
                           static_cast<size_t>(query_ix) * dim;

    const int k = top_k;
    std::vector<float> topk_dists(k);
    std::vector<int>   topk_idxs(k);

    float ms_distance = 0.0f, ms_topk = 0.0f;
    const float ms_total = kernel.compute_topk(
        h_query, k, topk_dists.data(), topk_idxs.data(),
        &ms_distance, &ms_topk);

    const size_t dist_shmem =
        static_cast<size_t>(dim + kernel.threads_per_block()) * sizeof(float);
    const size_t topk_shmem =
        static_cast<size_t>(kernel.topk_threads_per_block()) *
        DistanceKernel::kMaxTopK * (sizeof(float) + sizeof(int));
    std::printf("[kernel] l2_squared_kernel  blocks=%d  threads/block=%d  shmem=%zu B\n",
                rows, kernel.threads_per_block(), dist_shmem);
    std::printf("         %.3f ms  (%.2f GFLOP/s, 3 flops/elem)\n",
                ms_distance,
                3.0 * rows * dim / (ms_distance * 1.0e6));
    std::printf("[kernel] topk_kernel        blocks=1      threads/block=%d  shmem=%zu B  k=%d\n",
                kernel.topk_threads_per_block(), topk_shmem, k);
    std::printf("         %.3f ms\n", ms_topk);
    std::printf("[total]  on-device kernels: %.3f ms (distance + top-K)\n",
                ms_total);

    CUDA_CHECK(cudaEventDestroy(e_h2d_start));
    CUDA_CHECK(cudaEventDestroy(e_h2d_end));

    // The on-device top-K is sorted ascending, so the query's own row should
    // be slot 0 with distance ≈ 0 for an L2-normalised matrix.
    std::printf("[query] index=%d  nearest=row=%d  d2=%.6f (sanity: row==query_idx, d2~0)\n",
                query_ix, topk_idxs[0], topk_dists[0]);
    std::printf("[top-%d nearest by squared L2]\n", k);
    for (int i = 0; i < k; ++i) {
        std::printf("  %2d: row=%-6d  d2=%.6f\n",
                    i, topk_idxs[i], topk_dists[i]);
    }
    return 0;
}
