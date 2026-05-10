// Round-trip sanity check for the embedding pipeline.
//
// Loads vectors.fp32.bin from disk, copies the buffer to the GPU
// (cudaMemcpyHostToDevice), copies it straight back to a second host
// buffer (cudaMemcpyDeviceToHost), and verifies the bytes match.
//
// Usage:
//   ./build/check_roundtrip [path/to/vectors.fp32.bin] [dim]
// Defaults: data/embeddings/vectors.fp32.bin, dim=384.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
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
    const char* path = (argc > 1) ? argv[1] : "data/embeddings/vectors.fp32.bin";
    const int dim = (argc > 2) ? std::atoi(argv[2]) : 384;

    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "open failed: %s\n", path);
        return 1;
    }
    const std::streamsize bytes = in.tellg();
    in.seekg(0, std::ios::beg);

    if (bytes <= 0 || bytes % static_cast<std::streamsize>(sizeof(float)) != 0) {
        std::fprintf(stderr, "bad file size: %lld\n", static_cast<long long>(bytes));
        return 1;
    }
    const size_t n_floats = static_cast<size_t>(bytes) / sizeof(float);
    const size_t rows = (dim > 0) ? n_floats / static_cast<size_t>(dim) : 0;
    std::printf("[load]  %s\n        bytes=%lld  floats=%zu  rows=%zu  dim=%d\n",
                path, static_cast<long long>(bytes), n_floats, rows, dim);

    std::vector<float> h_in(n_floats);
    if (!in.read(reinterpret_cast<char*>(h_in.data()), bytes)) {
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

    float* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, bytes));
    std::printf("[alloc] device buffer @ %p (%.2f MB)\n",
                static_cast<void*>(d_buf), bytes / 1.0e6);

    cudaEvent_t e_start, e_h2d, e_d2h;
    CUDA_CHECK(cudaEventCreate(&e_start));
    CUDA_CHECK(cudaEventCreate(&e_h2d));
    CUDA_CHECK(cudaEventCreate(&e_d2h));

    std::vector<float> h_out(n_floats, std::numeric_limits<float>::quiet_NaN());

    CUDA_CHECK(cudaEventRecord(e_start));
    CUDA_CHECK(cudaMemcpy(d_buf, h_in.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e_h2d));
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_buf, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e_d2h));
    CUDA_CHECK(cudaEventSynchronize(e_d2h));

    float ms_h2d = 0.0f, ms_d2h = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, e_start, e_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, e_h2d, e_d2h));
    const double mb = bytes / 1.0e6;
    std::printf("[xfer]  H2D: %.3f ms (%.2f GB/s)   D2H: %.3f ms (%.2f GB/s)\n",
                ms_h2d, mb / ms_h2d, ms_d2h, mb / ms_d2h);

    const int cmp = std::memcmp(h_in.data(), h_out.data(), bytes);
    int rc = 0;
    if (cmp == 0) {
        std::printf("[check] OK -- %zu bytes round-tripped intact\n",
                    static_cast<size_t>(bytes));
    } else {
        size_t diffs = 0, first_diff = n_floats;
        for (size_t i = 0; i < n_floats; ++i) {
            if (h_in[i] != h_out[i]) {
                if (diffs == 0) first_diff = i;
                ++diffs;
            }
        }
        std::printf("[check] FAILED -- %zu / %zu floats differ (first at index %zu)\n",
                    diffs, n_floats, first_diff);
        rc = 1;
    }

    std::printf("[sample] first 5 floats: ");
    for (size_t i = 0; i < 5 && i < n_floats; ++i) {
        std::printf("%.6f ", h_in[i]);
    }
    std::printf("\n");

    CUDA_CHECK(cudaEventDestroy(e_start));
    CUDA_CHECK(cudaEventDestroy(e_h2d));
    CUDA_CHECK(cudaEventDestroy(e_d2h));
    CUDA_CHECK(cudaFree(d_buf));
    return rc;
}
