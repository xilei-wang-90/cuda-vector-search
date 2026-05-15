// Diagnostic: cross-check the on-device top-K reduction against a host
// reference for a sweep of (query_idx, k) pairs.
//
// For each pair, computes the squared-L2 distances on the GPU via
// DistanceKernel::compute_topk and a parallel host reference via
// std::partial_sort over a (distance, row_index) vector. The host sort
// orders by (distance, index) so ties are broken on the lower row index —
// matching the on-device kernel, whose strided insertion sort visits row
// indices in ascending order and therefore also keeps the lower index on
// equal distances.
//
// Indices must match exactly. Distances are compared with a small tolerance
// because the GPU sums the dim-D squared diffs in a different order than the
// host (block-strided + tree reduction vs. left-to-right), so fp32
// reassociation can shift the last bit or two.
//
// Exit code: 0 if every (query, k) pair matches, 1 otherwise.
//
// Usage:
//   ./build/check_topk [path/to/vectors.fp32.bin] [dim]
// Defaults: data/embeddings/vectors.fp32.bin, dim=384.

#include "distance_kernel.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <utility>
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

namespace {

constexpr float kDistTol = 1e-5f;

void host_topk(const std::vector<float>& embeddings,
               int rows,
               int dim,
               int query_ix,
               int k,
               std::vector<int>&   out_idxs,
               std::vector<float>& out_dists)
{
    const float* q = embeddings.data() + static_cast<size_t>(query_ix) * dim;
    std::vector<std::pair<float, int>> ranked(rows);
    for (int r = 0; r < rows; ++r) {
        const float* row = embeddings.data() + static_cast<size_t>(r) * dim;
        float acc = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float d = row[i] - q[i];
            acc += d * d;
        }
        // pair<distance, index> compares lexicographically — ties on
        // distance break on the smaller row index, matching the GPU.
        ranked[r] = {acc, r};
    }
    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end());

    out_idxs.resize(k);
    out_dists.resize(k);
    for (int i = 0; i < k; ++i) {
        out_dists[i] = ranked[i].first;
        out_idxs[i]  = ranked[i].second;
    }
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "data/embeddings/vectors.fp32.bin";
    const int   dim  = (argc > 2) ? std::atoi(argv[2]) : 384;

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

    std::vector<float> h_embeddings(n_floats);
    if (!in.read(reinterpret_cast<char*>(h_embeddings.data()), bytes)) {
        std::fprintf(stderr, "short read\n");
        return 1;
    }
    std::printf("[load]  %s\n        rows=%d  dim=%d\n", path, rows, dim);

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("[gpu]   device %d: %s  (sm_%d%d)\n",
                dev, prop.name, prop.major, prop.minor);

    DistanceKernel kernel(rows, dim);
    kernel.upload_embeddings(h_embeddings.data());

    const int queries[] = {0, 1, 42, 1234, rows - 1};
    const int ks[]      = {1, 5, 10, DistanceKernel::kMaxTopK};

    int total = 0, fail = 0;
    for (int q : queries) {
        if (q < 0 || q >= rows) continue;
        for (int k : ks) {
            if (k <= 0 || k > rows || k > DistanceKernel::kMaxTopK) continue;
            ++total;

            const float* h_query =
                h_embeddings.data() + static_cast<size_t>(q) * dim;
            std::vector<float> gpu_d(k);
            std::vector<int>   gpu_i(k);
            kernel.compute_topk(h_query, k, gpu_d.data(), gpu_i.data());

            std::vector<int>   ref_i;
            std::vector<float> ref_d;
            host_topk(h_embeddings, rows, dim, q, k, ref_i, ref_d);

            int idx_mismatch = 0, dist_mismatch = 0;
            for (int j = 0; j < k; ++j) {
                if (gpu_i[j] != ref_i[j]) ++idx_mismatch;
                if (std::abs(gpu_d[j] - ref_d[j]) > kDistTol) ++dist_mismatch;
            }
            const bool ok = (idx_mismatch == 0 && dist_mismatch == 0);
            std::printf("  q=%-5d k=%-3d %s",
                        q, k, ok ? "OK" : "FAIL");
            if (!ok) {
                std::printf("  (idx_mismatch=%d dist_mismatch=%d)",
                            idx_mismatch, dist_mismatch);
                std::printf("\n    gpu:");
                for (int j = 0; j < k; ++j)
                    std::printf(" (%d,%.6f)", gpu_i[j], gpu_d[j]);
                std::printf("\n    ref:");
                for (int j = 0; j < k; ++j)
                    std::printf(" (%d,%.6f)", ref_i[j], ref_d[j]);
                ++fail;
            }
            std::printf("\n");
        }
    }

    std::printf("[check] %s -- %d / %d (query, k) pairs match  (dist tol=%.0e)\n",
                fail == 0 ? "OK" : "FAILED",
                total - fail, total, kDistTol);
    return fail == 0 ? 0 : 1;
}
