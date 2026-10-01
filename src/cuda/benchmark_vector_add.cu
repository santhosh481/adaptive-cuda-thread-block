// =============================================================================
//  benchmark_vector_add.cu
//
//  Phase 4 workload driver for the project
//      "Adaptive CUDA Thread-Block Configuration for Workload-Aware GPU
//       Optimization".
//
//  WHAT THIS PROGRAM DOES
//  ----------------------
//  Given a vector length N, a thread-block size, and three counting
//  parameters (warmup, iterations, repetitions), it:
//
//      1. allocates host and device buffers once,
//      2. uploads the input vectors once,
//      3. for each of `repetitions` independent repetitions:
//            - runs `warmup` launches (not timed),
//            - runs `iterations` launches, timing each one with
//              cudaEventRecord on a dedicated pair of events,
//            - aggregates those `iterations` timings into
//              median / mean / min / max / stddev,
//      4. computes the same aggregates across the `repetitions` medians,
//      5. prints a single JSON object to stdout.
//
//  WHAT THIS PROGRAM DELIBERATELY DOES NOT DO
//  ------------------------------------------
//      * It does not write any file.          (Phase 6 owns datasets.)
//      * It does not extract features.        (Phase 7 owns features.)
//      * It does not compare to a baseline.   (Phase 8 owns baselines.)
//      * It does not use any ML model.        (Phase 9 owns ML.)
//      * It does not sweep block sizes itself.(Phase 4's Python driver
//                                              and Phase 6's experiment
//                                              runner will call this
//                                              program once per
//                                              block size.)
//
//  The unit of work is one vector-add launch.  Nothing else is timed: the
//  host-to-device and device-to-host copies happen exactly once, outside
//  the timed region, so the reported number reflects kernel execution
//  only, which is the quantity the ML model in Phase 9 will learn to
//  predict.
//
//  KERNEL DUPLICATION NOTE
//  -----------------------
//  The kernel `vector_add_kernel` below is byte-for-byte identical to the
//  one in `vector_add.cu`.  The duplication is deliberate: it avoids
//  CUDA separable compilation and keeps both executables independently
//  buildable on any toolchain.  If the kernel ever changes, change it in
//  both files and re-run the Phase 3 correctness check.
//
//  USAGE
//  -----
//      benchmark_vector_add                                       # defaults
//      benchmark_vector_add --n 4194304 --block-size 128
//      benchmark_vector_add --warmup 5 --iterations 200 --repetitions 3
//      benchmark_vector_add --json
//
//  Defaults match config/experiment.yaml:
//      warmup_iterations      = 10
//      measurement_iterations = 100
//      repetitions            = 5
//      n                      = 1<<20
//      block_size             = 256
//
//  EXIT CODES
//  ----------
//      0  success (JSON emitted)
//      2  configuration / usage error
//      3  CUDA runtime error
// =============================================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Exit codes (same convention as vector_add.cu)
// ---------------------------------------------------------------------------

enum : int {
    EXIT_OK      = 0,
    EXIT_CONFIG  = 2,
    EXIT_RUNTIME = 3,
};

// ---------------------------------------------------------------------------
// Kernel
//
// Byte-for-byte identical to vector_add.cu.  Do not modify one without
// modifying the other.
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
// Statistics
// ---------------------------------------------------------------------------

struct Summary {
    double median_ms = 0.0;
    double mean_ms   = 0.0;
    double min_ms    = 0.0;
    double max_ms    = 0.0;
    double stddev_ms = 0.0;
    std::size_t count = 0;
};

static Summary summarise(std::vector<double> samples) {
    Summary s;
    s.count = samples.size();
    if (samples.empty()) {
        return s;
    }

    std::sort(samples.begin(), samples.end());
    s.min_ms = samples.front();
    s.max_ms = samples.back();

    // Median: for even counts, average the two middle values.
    const std::size_t n = samples.size();
    if (n % 2 == 1) {
        s.median_ms = samples[n / 2];
    } else {
        s.median_ms = 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
    }

    double sum = 0.0;
    for (double v : samples) sum += v;
    s.mean_ms = sum / static_cast<double>(n);

    // Sample standard deviation (n-1 denominator); zero for n == 1.
    if (n > 1) {
        double acc = 0.0;
        for (double v : samples) {
            const double d = v - s.mean_ms;
            acc += d * d;
        }
        s.stddev_ms = std::sqrt(acc / static_cast<double>(n - 1));
    }

    return s;
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

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
// Input data
//
// Same deterministic generator as vector_add.cu so that benchmark runs and
// correctness runs use identical inputs for a given N.
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

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

struct Options {
    long long n           = 1LL << 20;
    int       block_size  = 256;
    int       device      = 0;
    int       warmup      = 10;
    int       iterations  = 100;
    int       repetitions = 5;
    bool      json        = true;   // JSON is the default output of this tool
};

static void print_usage(const char* argv0) {
    std::printf(
        "Usage: %s [options]\n"
        "\n"
        "  --n N             vector length (default: 1048576)\n"
        "  --block-size B    threads per block (default: 256)\n"
        "  --device D        CUDA device index (default: 0)\n"
        "  --warmup W        untimed launches before timing (default: 10)\n"
        "  --iterations I    timed launches per repetition (default: 100)\n"
        "  --repetitions R   independent repetitions (default: 5)\n"
        "  --json            emit JSON (default; kept for symmetry)\n"
        "  --help, -h        print this message\n"
        "\n"
        "The kernel under test is byte-for-byte identical to the one in\n"
        "vector_add.  Correctness is verified once at startup, before any\n"
        "timing is taken.\n",
        argv0);
}

static bool parse_options(int argc, char** argv, Options& out, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { err = std::string("missing value for ") + name; return nullptr; }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(EXIT_OK);
        } else if (arg == "--json") {
            out.json = true;
        } else if (arg == "--n") {
            const char* v = need_value("--n"); if (!v) return false;
            char* end = nullptr;
            long long parsed = std::strtoll(v, &end, 10);
            if (end == v || *end != '\0' || parsed <= 0) { err = std::string("invalid --n value: ") + v; return false; }
            out.n = parsed;
        } else if (arg == "--block-size") {
            const char* v = need_value("--block-size"); if (!v) return false;
            char* end = nullptr;
            long parsed = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || parsed <= 0 || parsed > 1024) { err = std::string("invalid --block-size value: ") + v; return false; }
            out.block_size = static_cast<int>(parsed);
        } else if (arg == "--device") {
            const char* v = need_value("--device"); if (!v) return false;
            char* end = nullptr;
            long parsed = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || parsed < 0) { err = std::string("invalid --device value: ") + v; return false; }
            out.device = static_cast<int>(parsed);
        } else if (arg == "--warmup") {
            const char* v = need_value("--warmup"); if (!v) return false;
            char* end = nullptr;
            long parsed = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || parsed < 0) { err = std::string("invalid --warmup value: ") + v; return false; }
            out.warmup = static_cast<int>(parsed);
        } else if (arg == "--iterations") {
            const char* v = need_value("--iterations"); if (!v) return false;
            char* end = nullptr;
            long parsed = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || parsed <= 0) { err = std::string("invalid --iterations value: ") + v; return false; }
            out.iterations = static_cast<int>(parsed);
        } else if (arg == "--repetitions") {
            const char* v = need_value("--repetitions"); if (!v) return false;
            char* end = nullptr;
            long parsed = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || parsed <= 0) { err = std::string("invalid --repetitions value: ") + v; return false; }
            out.repetitions = static_cast<int>(parsed);
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

    // ---- 2. Size the problem --------------------------------------------

    if (opts.n > static_cast<long long>(INT32_MAX)) {
        std::fprintf(stderr, "--n %lld exceeds INT32_MAX\n", opts.n);
        return EXIT_CONFIG;
    }
    const int n = static_cast<int>(opts.n);
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    const int block_size = opts.block_size;
    const int grid_size  = (n + block_size - 1) / block_size;

    // ---- 3. Host input and CPU reference --------------------------------

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

    // ---- 4. Device buffers and upload -----------------------------------

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

    // ---- 5. Correctness check (once, before any timing) -----------------
    //
    // We do not rely on Phase 3 having been run: this executable proves
    // the kernel it is about to time actually produces the right answer.

    vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_y, n);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "correctness launch failed\n");
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
        return EXIT_RUNTIME;
    }
    if (cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "correctness memcpy failed\n");
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
        return EXIT_RUNTIME;
    }
    const float tolerance = 1.0e-6f;
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_y[static_cast<std::size_t>(i)] -
                      h_ref[static_cast<std::size_t>(i)]) > tolerance) {
            correct = false;
            break;
        }
    }
    if (!correct) {
        std::fprintf(stderr, "correctness check FAILED; aborting before timing\n");
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
        return EXIT_RUNTIME;
    }

    // ---- 6. Events for timing -------------------------------------------

    cudaEvent_t ev_start{}, ev_stop{};
    if (cudaEventCreate(&ev_start) != cudaSuccess ||
        cudaEventCreate(&ev_stop) != cudaSuccess) {
        std::fprintf(stderr, "cudaEventCreate failed\n");
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
        return EXIT_RUNTIME;
    }

    // ---- 7. Warm-up + measurement loop ----------------------------------

    std::vector<double> per_rep_median(static_cast<std::size_t>(opts.repetitions), 0.0);
    std::vector<std::vector<double>> all_samples;
    all_samples.reserve(static_cast<std::size_t>(opts.repetitions));

    for (int r = 0; r < opts.repetitions; ++r) {
        // Warm-up (untimed).
        for (int w = 0; w < opts.warmup; ++w) {
            vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_y, n);
        }
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "warm-up launch failed\n");
            cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
            cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
            return EXIT_RUNTIME;
        }

        // Measurement (timed, one event pair per launch).
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(opts.iterations));
        for (int it = 0; it < opts.iterations; ++it) {
            if (cudaEventRecord(ev_start) != cudaSuccess) {
                std::fprintf(stderr, "cudaEventRecord(start) failed\n");
                cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
                cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
                return EXIT_RUNTIME;
            }
            vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_y, n);
            if (cudaEventRecord(ev_stop) != cudaSuccess) {
                std::fprintf(stderr, "cudaEventRecord(stop) failed\n");
                cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
                cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
                return EXIT_RUNTIME;
            }
            if (cudaEventSynchronize(ev_stop) != cudaSuccess) {
                std::fprintf(stderr, "cudaEventSynchronize failed\n");
                cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
                cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
                return EXIT_RUNTIME;
            }
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, ev_start, ev_stop) != cudaSuccess) {
                std::fprintf(stderr, "cudaEventElapsedTime failed\n");
                cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
                cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
                return EXIT_RUNTIME;
            }
            samples.push_back(static_cast<double>(ms));
        }

        const Summary s = summarise(samples);
        per_rep_median[static_cast<std::size_t>(r)] = s.median_ms;
        all_samples.push_back(std::move(samples));
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_y);

    // ---- 8. Aggregates across repetitions -------------------------------

    const Summary across = summarise(per_rep_median);

    // ---- 9. Report ------------------------------------------------------
    //
    // We always emit JSON from this tool; Phase 6's experiment runner will
    // parse it.  The `--json` flag exists so that this tool's CLI is
    // symmetric with vector_add's, but it is on by default here.

    std::printf("{\n");
    std::printf("  \"workload\": \"vector_add\",\n");
    std::printf("  \"n\": %d,\n", n);
    std::printf("  \"block_size\": %d,\n", block_size);
    std::printf("  \"grid_size\": %d,\n", grid_size);
    std::printf("  \"device_index\": %d,\n", opts.device);
    std::printf("  \"device_name\": \"%s\",\n", json_escape(prop.name).c_str());
    std::printf("  \"compute_capability\": \"%d.%d\",\n", prop.major, prop.minor);
    std::printf("  \"warp_size\": %d,\n", prop.warpSize);
    std::printf("  \"max_threads_per_block\": %d,\n", prop.maxThreadsPerBlock);
    std::printf("  \"warmup_iterations\": %d,\n", opts.warmup);
    std::printf("  \"measurement_iterations\": %d,\n", opts.iterations);
    std::printf("  \"repetitions\": %d,\n", opts.repetitions);

    // Per-repetition medians.
    std::printf("  \"per_repetition_median_ms\": [");
    for (std::size_t i = 0; i < per_rep_median.size(); ++i) {
        std::printf("%.6f%s", per_rep_median[i],
                    (i + 1 < per_rep_median.size()) ? ", " : "");
    }
    std::printf("],\n");

    // Aggregate over the per-repetition medians.
    std::printf("  \"aggregate\": {\n");
    std::printf("    \"median_ms\": %.6f,\n", across.median_ms);
    std::printf("    \"mean_ms\":   %.6f,\n", across.mean_ms);
    std::printf("    \"min_ms\":    %.6f,\n", across.min_ms);
    std::printf("    \"max_ms\":    %.6f,\n", across.max_ms);
    std::printf("    \"stddev_ms\": %.6f,\n", across.stddev_ms);
    std::printf("    \"count\":     %zu\n",  across.count);
    std::printf("  }\n");
    std::printf("}\n");

    return EXIT_OK;
}