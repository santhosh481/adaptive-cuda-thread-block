// =============================================================================
//  benchmark_stencil.cu
//
//  Phase 5 timing harness for the stencil workload.  Same structure as
//  benchmark_vector_add.cu and benchmark_reduction.cu.
//
//  The kernel is byte-for-byte identical to stencil.cu.
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

enum : int { EXIT_OK = 0, EXIT_CONFIG = 2, EXIT_RUNTIME = 3 };

__global__ void stencil_kernel(const float* __restrict__ a,
                               float* __restrict__ y,
                               int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float center = a[i];
    const float left   = (i > 0)     ? a[i - 1] : 0.0f;
    const float right  = (i < n - 1) ? a[i + 1] : 0.0f;

    if (i == 0) {
        y[i] = center + right;
    } else if (i == n - 1) {
        y[i] = left + center;
    } else {
        y[i] = left + center + right;
    }
}

struct Summary {
    double median_ms = 0.0;
    double mean_ms   = 0.0;
    double min_ms    = 0.0;
    double max_ms    = 0.0;
    double stddev_ms = 0.0;
    std::size_t count = 0;
};

static Summary summarise(std::vector<double> samples) {
    Summary s; s.count = samples.size();
    if (samples.empty()) return s;
    std::sort(samples.begin(), samples.end());
    s.min_ms = samples.front(); s.max_ms = samples.back();
    const std::size_t n = samples.size();
    if (n % 2 == 1) s.median_ms = samples[n / 2];
    else            s.median_ms = 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
    double sum = 0.0; for (double v : samples) sum += v;
    s.mean_ms = sum / static_cast<double>(n);
    if (n > 1) {
        double acc = 0.0;
        for (double v : samples) { const double d = v - s.mean_ms; acc += d * d; }
        s.stddev_ms = std::sqrt(acc / static_cast<double>(n - 1));
    }
    return s;
}

static std::string json_escape(const std::string& s) {
    std::string out; out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", c); out += buf;
                } else out += c;
        }
    }
    return out;
}

static inline float deterministic_value(std::uint32_t seed) {
    std::uint32_t x = seed ? seed : 0x9E3779B9u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return static_cast<float>(x) / 4294967296.0f;
}

static void fill_input(std::vector<float>& v, std::uint32_t base_seed) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = deterministic_value(base_seed + static_cast<std::uint32_t>(i));
    }
}

struct Options {
    long long n           = 1LL << 20;
    int       block_size  = 256;
    int       device      = 0;
    int       warmup      = 10;
    int       iterations  = 100;
    int       repetitions = 5;
    bool      json        = true;
};

static void print_usage(const char* argv0) {
    std::printf(
        "Usage: %s [options]\n"
        "\n"
        "  --n N             array length (default: 1048576)\n"
        "  --block-size B    threads per block (default: 256)\n"
        "  --device D        CUDA device index (default: 0)\n"
        "  --warmup W        untimed launches before timing (default: 10)\n"
        "  --iterations I    timed launches per repetition (default: 100)\n"
        "  --repetitions R   independent repetitions (default: 5)\n"
        "  --json            emit JSON (default)\n"
        "  --help, -h        print this message\n",
        argv0);
}

static bool parse_options(int argc, char** argv, Options& out, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { err = std::string("missing value for ") + name; return nullptr; }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); std::exit(EXIT_OK); }
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
        } else if (arg == "--warmup") {
            const char* v = need_value("--warmup"); if (!v) return false;
            char* end = nullptr; long parsed = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || parsed < 0) { err = std::string("invalid --warmup value: ") + v; return false; }
            out.warmup = static_cast<int>(parsed);
        } else if (arg == "--iterations") {
            const char* v = need_value("--iterations"); if (!v) return false;
            char* end = nullptr; long parsed = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || parsed <= 0) { err = std::string("invalid --iterations value: ") + v; return false; }
            out.iterations = static_cast<int>(parsed);
        } else if (arg == "--repetitions") {
            const char* v = need_value("--repetitions"); if (!v) return false;
            char* end = nullptr; long parsed = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || parsed <= 0) { err = std::string("invalid --repetitions value: ") + v; return false; }
            out.repetitions = static_cast<int>(parsed);
        } else {
            err = std::string("unknown argument: ") + arg; return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    Options opts;
    std::string parse_err;
    if (!parse_options(argc, argv, opts, parse_err)) {
        std::fprintf(stderr, "argument error: %s\n", parse_err.c_str());
        std::fprintf(stderr, "run with --help for usage\n");
        return EXIT_CONFIG;
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
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
    if (opts.n < 2) {
        std::fprintf(stderr, "--n must be at least 2 for the 3-point stencil\n");
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

    // ---- Input + CPU reference -----------------------------------------

    std::vector<float> h_a(static_cast<std::size_t>(n));
    std::vector<float> h_y(static_cast<std::size_t>(n), 0.0f);
    std::vector<float> h_ref(static_cast<std::size_t>(n), 0.0f);
    fill_input(h_a, 0xBADC0DEu);
    for (int i = 0; i < n; ++i) {
        const float c = h_a[static_cast<std::size_t>(i)];
        const float l = (i > 0)     ? h_a[static_cast<std::size_t>(i - 1)] : 0.0f;
        const float r = (i < n - 1) ? h_a[static_cast<std::size_t>(i + 1)] : 0.0f;
        if (i == 0)        h_ref[static_cast<std::size_t>(i)] = c + r;
        else if (i == n-1) h_ref[static_cast<std::size_t>(i)] = l + c;
        else               h_ref[static_cast<std::size_t>(i)] = l + c + r;
    }

    // ---- Device buffers -------------------------------------------------

    float* d_a = nullptr;
    float* d_y = nullptr;
    if (cudaMalloc(&d_a, bytes) != cudaSuccess || cudaMalloc(&d_y, bytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        cudaFree(d_a); cudaFree(d_y);
        return EXIT_RUNTIME;
    }
    if (cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy host->device failed\n");
        cudaFree(d_a); cudaFree(d_y);
        return EXIT_RUNTIME;
    }

    // ---- Correctness gate (once) ---------------------------------------

    stencil_kernel<<<grid_size, block_size>>>(d_a, d_y, n);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "correctness launch failed\n");
        cudaFree(d_a); cudaFree(d_y);
        return EXIT_RUNTIME;
    }
    if (cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "correctness memcpy failed\n");
        cudaFree(d_a); cudaFree(d_y);
        return EXIT_RUNTIME;
    }
    {
        const float tol = 1.0e-6f;
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(h_y[static_cast<std::size_t>(i)] -
                          h_ref[static_cast<std::size_t>(i)]) > tol) { ok = false; break; }
        }
        if (!ok) {
            std::fprintf(stderr, "correctness check FAILED; aborting before timing\n");
            cudaFree(d_a); cudaFree(d_y);
            return EXIT_RUNTIME;
        }
    }

    // ---- Timing loop ----------------------------------------------------

    cudaEvent_t ev_start{}, ev_stop{};
    if (cudaEventCreate(&ev_start) != cudaSuccess || cudaEventCreate(&ev_stop) != cudaSuccess) {
        std::fprintf(stderr, "cudaEventCreate failed\n");
        cudaFree(d_a); cudaFree(d_y);
        return EXIT_RUNTIME;
    }

    std::vector<double> per_rep_median(static_cast<std::size_t>(opts.repetitions), 0.0);
    for (int r = 0; r < opts.repetitions; ++r) {
        for (int w = 0; w < opts.warmup; ++w) {
            stencil_kernel<<<grid_size, block_size>>>(d_a, d_y, n);
        }
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "warm-up launch failed\n");
            cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
            cudaFree(d_a); cudaFree(d_y);
            return EXIT_RUNTIME;
        }
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(opts.iterations));
        for (int it = 0; it < opts.iterations; ++it) {
            cudaEventRecord(ev_start);
            stencil_kernel<<<grid_size, block_size>>>(d_a, d_y, n);
            cudaEventRecord(ev_stop);
            if (cudaEventSynchronize(ev_stop) != cudaSuccess) {
                std::fprintf(stderr, "cudaEventSynchronize failed\n");
                cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
                cudaFree(d_a); cudaFree(d_y);
                return EXIT_RUNTIME;
            }
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);
            samples.push_back(static_cast<double>(ms));
        }
        const Summary s = summarise(samples);
        per_rep_median[static_cast<std::size_t>(r)] = s.median_ms;
    }

    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    cudaFree(d_a); cudaFree(d_y);

    const Summary across = summarise(per_rep_median);

    std::printf("{\n");
    std::printf("  \"workload\": \"stencil\",\n");
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
    std::printf("  \"per_repetition_median_ms\": [");
    for (std::size_t i = 0; i < per_rep_median.size(); ++i) {
        std::printf("%.6f%s", per_rep_median[i], (i + 1 < per_rep_median.size()) ? ", " : "");
    }
    std::printf("],\n");
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