// =============================================================================
//  vector_add.cu
//
//  Phase 3 workload for the project
//      "Adaptive CUDA Thread-Block Configuration for Workload-Aware GPU
//       Optimization".
//
//  WHAT THIS PROGRAM DOES
//  ----------------------
//  Computes  y[i] = a[i] + b[i]  for float32 vectors of length N, using a
//  single CUDA kernel launch with a caller-chosen thread-block size, and
//  verifies the result element-by-element against a CPU reference computed
//  on the host.
//
//  WHAT THIS PROGRAM DELIBERATELY DOES NOT DO
//  ------------------------------------------
//      * It does not measure time.         (Phase 4 owns timing.)
//      * It does not run warm-ups.         (Phase 4 owns warm-ups.)
//      * It does not write any data file.  (Phase 6 owns datasets.)
//      * It does not extract features.     (Phase 7 owns features.)
//      * It does not use any ML model.     (Phase 9 owns ML.)
//
//  The only outputs are (a) a human-readable PASS/FAIL line, optionally
//  (b) a JSON object with the same information, and (c) a process exit code.
//
//  DESIGN NOTES
//  ------------
//  * The thread-block size is a runtime parameter, not a compile-time
//    constant.  This is required so that Phase 4 can sweep it.
//  * The block size is validated against cudaDeviceProp::maxThreadsPerBlock
//    at run time.  No GPU model, no architecture and no hardware limit is
//    hard-coded anywhere in this file.
//  * The grid size is derived as ceil(N / block_size) on the host.
//  * The kernel uses one thread per element with a bounds check; this is
//    the canonical, side-effect-free formulation of vector addition and is
//    what the later benchmarking phases will measure.
//
//  USAGE
//  -----
//      vector_add                                   # N = 1<<20,  block = 256
//      vector_add --n 4194304 --block-size 128
//      vector_add --n 1000000 --block-size 512 --json
//      vector_add --help
//
//  EXIT CODES
//  ----------
//      0  PASS      (kernel output matched the CPU reference)
//      1  FAIL      (kernel output did not match)
//      2  CONFIG    (bad CLI arguments, or block size exceeds the device limit)
//      3  RUNTIME   (a CUDA runtime call failed)
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
// One thread per element.  A single bounds check.  No shared memory, no
// atomics, no branching beyond the guard, so that the effect measured in
// later phases is dominated by the thread-block geometry, not by the
// kernel's own control flow.
// ---------------------------------------------------------------------------

__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ y,
                                  int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

static void print_usage(const char* argv0) {
    std::printf(
        "Usage: %s [--n N] [--block-size B] [--device D] [--json]\n"
        "\n"
        "  --n N           vector length (default: 1048576)\n"
        "  --block-size B  threads per block (default: 256)\n"
        "  --device D      CUDA device index (default: 0)\n"
        "  --json          emit a single JSON object instead of plain text\n"
        "  --help, -h      print this message\n"
        "\n"
        "This program only checks correctness.  It does not measure time.\n",
        argv0);
}

// A tiny, deterministic pseudorandom generator so that the input data is
// reproducible across runs and across machines without pulling in <random>
// (whose distributions are not guaranteed stable between standard libraries).
static inline float deterministic_value(std::uint32_t seed) {
    // xorshift32 -> [0, 1)
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

// JSON string escaping (identical approach to device_query.cu, but local so
// that the two files remain independently compilable).
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

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

struct Options {
    long long n          = 1LL << 20;  // 1,048,576
    int       block_size = 256;
    int       device     = 0;
    bool      json       = false;
};

static bool parse_options(int argc, char** argv, Options& out, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                err = std::string("missing value for ") + name;
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(EXIT_PASS);
        } else if (arg == "--json") {
            out.json = true;
        } else if (arg == "--n") {
            const char* v = need_value("--n");
            if (!v) return false;
            char* end = nullptr;
            long long parsed = std::strtoll(v, &end, 10);
            if (end == v || *end != '\0' || parsed <= 0) {
                err = std::string("invalid --n value: ") + v;
                return false;
            }
            out.n = parsed;
        } else if (arg == "--block-size") {
            const char* v = need_value("--block-size");
            if (!v) return false;
            char* end = nullptr;
            long parsed = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || parsed <= 0 || parsed > 1024) {
                err = std::string("invalid --block-size value: ") + v;
                return false;
            }
            out.block_size = static_cast<int>(parsed);
        } else if (arg == "--device") {
            const char* v = need_value("--device");
            if (!v) return false;
            char* end = nullptr;
            long parsed = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || parsed < 0) {
                err = std::string("invalid --device value: ") + v;
                return false;
            }
            out.device = static_cast<int>(parsed);
        } else {
            err = std::string("unknown argument: ") + arg;
            return false;
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

    // ---- 1. Device selection and capability check ------------------------

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n",
                     cudaGetErrorString(err));
        return EXIT_RUNTIME;
    }
    if (device_count == 0) {
        std::fprintf(stderr,
            "no CUDA device is visible from this process.\n"
            "this program must be run on a machine with an NVIDIA GPU;\n"
            "see README.md for Google Colab instructions.\n");
        return EXIT_RUNTIME;
    }
    if (opts.device < 0 || opts.device >= device_count) {
        std::fprintf(stderr,
            "--device %d is out of range; %d device(s) present.\n",
            opts.device, device_count);
        return EXIT_CONFIG;
    }

    if (cudaSetDevice(opts.device) != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice(%d) failed\n", opts.device);
        return EXIT_RUNTIME;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, opts.device) != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceProperties(%d) failed\n", opts.device);
        return EXIT_RUNTIME;
    }

    // Runtime validation of the caller's block size against the actual
    // device.  Never assume a limit; always read it.
    if (opts.block_size > prop.maxThreadsPerBlock) {
        std::fprintf(stderr,
            "--block-size %d exceeds this device's maxThreadsPerBlock (%d).\n",
            opts.block_size, prop.maxThreadsPerBlock);
        return EXIT_CONFIG;
    }
    if (opts.block_size % prop.warpSize != 0) {
        std::fprintf(stderr,
            "--block-size %d is not a multiple of this device's warp size (%d).\n",
            opts.block_size, prop.warpSize);
        return EXIT_CONFIG;
    }

    // ---- 2. Size the problem --------------------------------------------

    const long long n_ll = opts.n;
    if (n_ll > static_cast<long long>(INT32_MAX)) {
        std::fprintf(stderr,
            "--n %lld exceeds INT32_MAX; this workload uses 32-bit indexing.\n",
            n_ll);
        return EXIT_CONFIG;
    }
    const int n = static_cast<int>(n_ll);

    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    const int block_size = opts.block_size;
    const int grid_size  = (n + block_size - 1) / block_size;

    // ---- 3. Host allocation and input generation ------------------------

    std::vector<float> h_a(static_cast<std::size_t>(n));
    std::vector<float> h_b(static_cast<std::size_t>(n));
    std::vector<float> h_y(static_cast<std::size_t>(n), 0.0f);
    std::vector<float> h_ref(static_cast<std::size_t>(n));

    fill_input(h_a, 0xA5A5A5A5u);
    fill_input(h_b, 0x5A5A5A5Au);

    for (int i = 0; i < n; ++i) {
        h_ref[static_cast<std::size_t>(i)] =
            h_a[static_cast<std::size_t>(i)] + h_b[static_cast<std::size_t>(i)];
    }

    // ---- 4. Device allocation -------------------------------------------

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_y = nullptr;

    if (cudaMalloc(&d_a, bytes) != cudaSuccess ||
        cudaMalloc(&d_b, bytes) != cudaSuccess ||
        cudaMalloc(&d_y, bytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed for %zu bytes per buffer\n", bytes);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
        return EXIT_RUNTIME;
    }

    if (cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy host->device failed\n");
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
        return EXIT_RUNTIME;
    }

    // ---- 5. Launch the kernel exactly once ------------------------------
    //
    // Phase 3 policy: exactly one launch.  No warm-up, no repetition.  The
    // benchmarking harness in Phase 4 will reuse vector_add_kernel and add
    // the timing and repetition logic around it.
    // ---------------------------------------------------------------------

    vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_y, n);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr, "kernel launch failed: %s\n",
                     cudaGetErrorString(launch_err));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
        return EXIT_RUNTIME;
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
                     cudaGetErrorString(sync_err));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
        return EXIT_RUNTIME;
    }

    // ---- 6. Copy back and verify against the CPU reference --------------

    if (cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy device->host failed\n");
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
        return EXIT_RUNTIME;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_y);

    std::size_t first_mismatch = static_cast<std::size_t>(-1);
    float       mismatch_gpu   = 0.0f;
    float       mismatch_cpu   = 0.0f;

    // Absolute tolerance is fine here because the inputs are in [0, 1) and
    // the reference sum is also in [0, 2).  Vector addition on float32 is
    // exact for these magnitudes, but we still allow one ULP-worth of slack
    // to avoid depending on the exact rounding of the addition.
    const float tolerance = 1.0e-6f;

    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
        const float diff = std::fabs(h_y[i] - h_ref[i]);
        if (diff > tolerance) {
            first_mismatch = i;
            mismatch_gpu   = h_y[i];
            mismatch_cpu   = h_ref[i];
            break;
        }
    }

    const bool passed = (first_mismatch == static_cast<std::size_t>(-1));

    // ---- 7. Report ------------------------------------------------------

    if (opts.json) {
        std::printf("{\n");
        std::printf("  \"workload\": \"vector_add\",\n");
        std::printf("  \"status\": \"%s\",\n", passed ? "PASS" : "FAIL");
        std::printf("  \"n\": %d,\n", n);
        std::printf("  \"block_size\": %d,\n", block_size);
        std::printf("  \"grid_size\": %d,\n", grid_size);
        std::printf("  \"device_index\": %d,\n", opts.device);
        std::printf("  \"device_name\": \"%s\",\n", json_escape(prop.name).c_str());
        std::printf("  \"compute_capability\": \"%d.%d\",\n", prop.major, prop.minor);
        std::printf("  \"warp_size\": %d,\n", prop.warpSize);
        std::printf("  \"max_threads_per_block\": %d,\n", prop.maxThreadsPerBlock);
        std::printf("  \"tolerance\": %g,\n", static_cast<double>(tolerance));
        if (passed) {
            std::printf("  \"first_mismatch\": null,\n");
            std::printf("  \"gpu_value\": null,\n");
            std::printf("  \"cpu_value\": null\n");
        } else {
            std::printf("  \"first_mismatch\": %zu,\n", first_mismatch);
            std::printf("  \"gpu_value\": %.9g,\n", static_cast<double>(mismatch_gpu));
            std::printf("  \"cpu_value\": %.9g\n",  static_cast<double>(mismatch_cpu));
        }
        std::printf("}\n");
    } else {
        std::printf("vector_add correctness check\n");
        std::printf("  status              : %s\n", passed ? "PASS" : "FAIL");
        std::printf("  n                   : %d\n", n);
        std::printf("  block size          : %d\n", block_size);
        std::printf("  grid size           : %d\n", grid_size);
        std::printf("  device              : [%d] %s (cc %d.%d)\n",
                    opts.device, prop.name, prop.major, prop.minor);
        std::printf("  warp size           : %d\n", prop.warpSize);
        std::printf("  max threads / block : %d\n", prop.maxThreadsPerBlock);
        std::printf("  tolerance           : %g\n", static_cast<double>(tolerance));
        if (!passed) {
            std::printf("  first mismatch      : index %zu  (gpu=%.9g  cpu=%.9g)\n",
                        first_mismatch,
                        static_cast<double>(mismatch_gpu),
                        static_cast<double>(mismatch_cpu));
        }
    }

    return passed ? EXIT_PASS : EXIT_FAIL;
}