// =============================================================================
//  reduction.cu
//
//  Phase 5 correctness tool for the project
//      "Adaptive CUDA Thread-Block Configuration for Workload-Aware GPU
//       Optimization".
//
//  WHAT THIS PROGRAM DOES
//  ----------------------
//  Computes the sum over a float32 array of length N using a two-stage
//  block reduction:
//
//      stage 1 (device):  each thread-block reduces its slice of `input`
//                         into one partial sum, stored in `partial[block]`
//      stage 2 (host):    the partial sums are summed sequentially on the
//                         CPU and compared to a reference
//
//  The reduction inside a block uses a shared-memory tree with a
//  `__syncthreads()` between levels.  The tree is the canonical one:
//
//      for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
//          if (tid < offset) sdata[tid] += sdata[tid + offset];
//
//  so the number of barriers per block depends on log2(block_size).
//
//  WHAT THIS PROGRAM DELIBERATELY DOES NOT DO
//  ------------------------------------------
//      * It does not measure time.         (The benchmark tool does.)
//      * It does not sweep block sizes.    (Phase 6's runner does.)
//      * It does not write any file.       (Phase 6 owns datasets.)
//      * It does not extract features.     (Phase 7 owns features.)
//
//  The only outputs are a human-readable PASS/FAIL line, optionally a JSON
//  object with the same information, and an exit code.
//
//  USAGE
//  -----
//      reduction
//      reduction --n 4194304 --block-size 128
//      reduction --n 1000000 --block-size 384 --json
//
//  EXIT CODES
//  ----------
//      0  PASS
//      1  FAIL
//      2  CONFIG
//      3  RUNTIME
// =============================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Exit codes
// ---------------------------------------------------------------------------

enum : int {
    EXIT_PASS    = 0,
    EXIT_FAIL    = 1,
    EXIT_CONFIG  = 2,
    EXIT_RUNTIME = 3,
};

// ---------------------------------------------------------------------------
// Kernel
//
// One block reduces one contiguous slice of `input` into `partial[blockIdx.x]`.
// The kernel handles the case where the slice is not a multiple of blockDim.x
// by having each thread sum several strided elements first; the shared-memory
// tree then reduces those per-thread partials.
//
// The kernel never reads outside [0, n).  The block boundary is computed
// from blockIdx.x and blockDim.x; the guard `i < n` protects the tail.
// ---------------------------------------------------------------------------

__global__ void reduction_kernel(const float* __restrict__ input,
                                 float* __restrict__ partial,
                                 int n) {
    extern __shared__ float sdata[];

    const int tid      = threadIdx.x;
    const int gtid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride   = gridDim.x * blockDim.x;

    float local = 0.0f;
    for (int i = gtid; i < n; i += stride) {
        local += input[i];
    }
    sdata[tid] = local;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline float deterministic_value(std::uint32_t seed) {
    std::uint32_t x = seed ? seed : 0x9E3779B9u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return static_cast<float>(x) / 4294967296.0f;
}

static void fill_input(std::vector<float>& v, std::uint32_t base_seed) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = deterministic_value(base_seed + static_cast<std::uint32_t>(i));
    }
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static void print_usage(const char* argv0) {
    std::printf(
        "Usage: %s [--n N] [--block-size B] [--device D] [--json]\n"
        "\n"
        "  --n N           input length (default: 1048576)\n"
        "  --block-size B  threads per block (default: 256)\n"
        "  --device D      CUDA device index (default: 0)\n"
        "  --json          emit a single JSON object\n"
        "  --help, -h      print this message\n"
        "\n"
        "This program only checks correctness.  It does not measure time.\n",
        argv0);
}

struct Options {
    long long n          = 1LL << 20;
    int       block_size = 256;
    int       device     = 0;
    bool      json       = false;
};

static bool parse_options(int argc, char** argv, Options& out, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { err = std::string("missing value for ") + name; return nullptr; }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); std::exit(EXIT_PASS); }
        else if (arg == "--json") { out.json = true; }
        else if (arg == "--n") {
            const char* v = need_value("--n"); if (!v) return false;
            char* end = nullptr; long long parsed = std::strtoll(v, &end, 10);
            if (end == v || *end != '\0' || parsed <= 0) { err = std::string("invalid --n value: ") + v; return false; }
            out.n = parsed;
        } else if (arg == "--block-size") {
            const char* v = need_value("--block-size"); if (!v) return false;
            char* end = nullptr; long parsed = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || parsed <= 0 || parsed > 1024) { err = std::string("invalid --block-size value: ") + v; return false; }
            out.block_size = static_cast<int>(parsed);
        } else if (arg == "--device") {
            const char* v = need_value("--device"); if (!v) return false;
            char* end = nullptr; long parsed = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || parsed < 0) { err = std::string("invalid --device value: ") + v; return false; }
            out.device = static_cast<int>(parsed);
        } else {
            err = std::string("unknown argument: ") + arg; return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    Options opts;
    std::string parse_err;
    if (!parse_options(argc, argv, opts, parse_err)) {
        std::fprintf(stderr, "argument error: %s\n", parse_err.c_str());
        std::fprintf(stderr, "run with --help for usage\n");
        return EXIT_CONFIG;
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_RUNTIME;
    }
    if (device_count == 0) {
        std::fprintf(stderr, "no CUDA device visible; run on a GPU machine.\n");
        return EXIT_RUNTIME;
    }
    if (opts.device < 0 || opts.device >= device_count) {
        std::fprintf(stderr, "--device %d out of range (%d present)\n", opts.device, device_count);
        return EXIT_CONFIG;
    }
    if (cudaSetDevice(opts.device) != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice(%d) failed\n", opts.device);
        return EXIT_RUNTIME;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, opts.device) != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceProperties failed\n");
        return EXIT_RUNTIME;
    }
    if (opts.block_size > prop.maxThreadsPerBlock) {
        std::fprintf(stderr, "--block-size %d exceeds maxThreadsPerBlock (%d)\n",
                     opts.block_size, prop.maxThreadsPerBlock);
        return EXIT_CONFIG;
    }
    if (opts.block_size % prop.warpSize != 0) {
        std::fprintf(stderr, "--block-size %d not a multiple of warp size (%d)\n",
                     opts.block_size, prop.warpSize);
        return EXIT_CONFIG;
    }

    if (opts.n > static_cast<long long>(INT32_MAX)) {
        std::fprintf(stderr, "--n %lld exceeds INT32_MAX\n", opts.n);
        return EXIT_CONFIG;
    }
    const int n = static_cast<int>(opts.n);
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    const int block_size = opts.block_size;
    const int grid_size  = (n + block_size - 1) / block_size;
    const std::size_t smem = static_cast<std::size_t>(block_size) * sizeof(float);

    // ---- Host input and CPU reference -----------------------------------

    std::vector<float> h_in(static_cast<std::size_t>(n));
    fill_input(h_in, 0xC0FFEEu);

    // Double-precision accumulation for the reference so that we can tell
    // apart "kernel is wrong" from "single-precision round-off".
    double cpu_ref = 0.0;
    for (int i = 0; i < n; ++i) {
        cpu_ref += static_cast<double>(h_in[static_cast<std::size_t>(i)]);
    }

    // ---- Device buffers --------------------------------------------------

    float* d_in = nullptr;
    float* d_partial = nullptr;
    std::vector<float> h_partial(static_cast<std::size_t>(grid_size), 0.0f);

    if (cudaMalloc(&d_in, bytes) != cudaSuccess ||
        cudaMalloc(&d_partial, static_cast<std::size_t>(grid_size) * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        cudaFree(d_in); cudaFree(d_partial);
        return EXIT_RUNTIME;
    }
    if (cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy host->device failed\n");
        cudaFree(d_in); cudaFree(d_partial);
        return EXIT_RUNTIME;
    }

    // ---- Launch ----------------------------------------------------------

    reduction_kernel<<<grid_size, block_size, smem>>>(d_in, d_partial, n);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(launch_err));
        cudaFree(d_in); cudaFree(d_partial);
        return EXIT_RUNTIME;
    }
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(sync_err));
        cudaFree(d_in); cudaFree(d_partial);
        return EXIT_RUNTIME;
    }

    if (cudaMemcpy(h_partial.data(), d_partial,
                   static_cast<std::size_t>(grid_size) * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy device->host failed\n");
        cudaFree(d_in); cudaFree(d_partial);
        return EXIT_RUNTIME;
    }
    cudaFree(d_in);
    cudaFree(d_partial);

    // ---- Verify against the double-precision reference ------------------

    double gpu_sum = 0.0;
    for (int i = 0; i < grid_size; ++i) {
        gpu_sum += static_cast<double>(h_partial[static_cast<std::size_t>(i)]);
    }

    // Relative tolerance.  For n = 2^20 single-precision additions the
    // round-off is a few parts in 10^4; we allow 1 part in 10^3 to be safe,
    // but this is far tighter than any real bug would produce.
    const double denom = std::max(1.0, std::fabs(cpu_ref));
    const double rel_err = std::fabs(gpu_sum - cpu_ref) / denom;
    const double rel_tol = 1.0e-3;
    const bool passed = (rel_err <= rel_tol);

    // ---- Report ---------------------------------------------------------

    if (opts.json) {
        std::printf("{\n");
        std::printf("  \"workload\": \"reduction\",\n");
        std::printf("  \"status\": \"%s\",\n", passed ? "PASS" : "FAIL");
        std::printf("  \"n\": %d,\n", n);
        std::printf("  \"block_size\": %d,\n", block_size);
        std::printf("  \"grid_size\": %d,\n", grid_size);
        std::printf("  \"shared_memory_bytes\": %zu,\n", smem);
        std::printf("  \"device_index\": %d,\n", opts.device);
        std::printf("  \"device_name\": \"%s\",\n", json_escape(prop.name).c_str());
        std::printf("  \"compute_capability\": \"%d.%d\",\n", prop.major, prop.minor);
        std::printf("  \"warp_size\": %d,\n", prop.warpSize);
        std::printf("  \"max_threads_per_block\": %d,\n", prop.maxThreadsPerBlock);
        std::printf("  \"gpu_sum\": %.9g,\n", gpu_sum);
        std::printf("  \"cpu_sum\": %.9g,\n", cpu_ref);
        std::printf("  \"rel_error\": %.9g,\n", rel_err);
        std::printf("  \"rel_tolerance\": %.9g\n", rel_tol);
        std::printf("}\n");
    } else {
        std::printf("reduction correctness check\n");
        std::printf("  status              : %s\n", passed ? "PASS" : "FAIL");
        std::printf("  n                   : %d\n", n);
        std::printf("  block size          : %d\n", block_size);
        std::printf("  grid size           : %d\n", grid_size);
        std::printf("  shared memory       : %zu bytes\n", smem);
        std::printf("  device              : [%d] %s (cc %d.%d)\n",
                    opts.device, prop.name, prop.major, prop.minor);
        std::printf("  gpu sum             : %.9g\n", gpu_sum);
        std::printf("  cpu reference       : %.9g\n", cpu_ref);
        std::printf("  relative error      : %.3e (tolerance %.3e)\n", rel_err, rel_tol);
    }

    return passed ? EXIT_PASS : EXIT_FAIL;
}