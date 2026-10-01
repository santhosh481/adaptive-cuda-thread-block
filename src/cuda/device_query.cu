// =============================================================================
//  device_query.cu
//
//  CUDA environment / correctness probe for the project
//      "Adaptive CUDA Thread-Block Configuration for Workload-Aware GPU
//       Optimization".
//
//  This program is NOT a benchmark.  It answers exactly two questions:
//
//      1. Which CUDA devices exist, and what hardware limits constrain the
//         thread-block configuration (SM count, max threads per block,
//         registers per block, shared memory per block, ...)?
//
//      2. Does a trivial kernel actually launch and complete correctly on
//         each device?
//
//  It is deliberately architecture-agnostic:  it reads every hardware
//  property from the CUDA runtime at execution time.  Nothing about a
//  specific GPU model is compiled in.
//
//  Usage:
//      device_query           human-readable report
//      device_query --json    machine-readable report (single JSON object)
//
//  Exit code:
//      0  at least one device was found and the kernel smoke test passed
//      1  no CUDA device was found, or the smoke test failed
//      2  CUDA runtime error while enumerating devices
// =============================================================================

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK_RETURN(call)                                              \
    do {                                                                     \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                         __FILE__, __LINE__, cudaGetErrorString(_err));      \
            return 2;                                                        \
        }                                                                    \
    } while (0)

#define CUDA_CHECK_BOOL(call)                                                \
    do {                                                                     \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                         __FILE__, __LINE__, cudaGetErrorString(_err));      \
            return false;                                                    \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Smoke-test kernel
//
// Minimal, portable, and impossible to confuse with a benchmark.  Every
// thread writes one value into the output buffer.  If this kernel runs, the
// CUDA toolchain, the driver, and the device are all consistent.
// ---------------------------------------------------------------------------

__global__ void hello_kernel(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = i;
    }
}

static bool run_kernel_smoke_test(int device_index, std::string& message) {
    const int n = 1024;
    int* d_out = nullptr;
    int* h_out = static_cast<int*>(std::malloc(sizeof(int) * n));
    if (h_out == nullptr) {
        message = "host allocation failed";
        return false;
    }
    std::memset(h_out, 0, sizeof(int) * n);

    if (cudaSetDevice(device_index) != cudaSuccess) {
        message = "cudaSetDevice failed";
        std::free(h_out);
        return false;
    }
    if (cudaMalloc(&d_out, sizeof(int) * n) != cudaSuccess) {
        message = "cudaMalloc failed";
        std::free(h_out);
        return false;
    }

    const int threads = 128;
    const int blocks  = (n + threads - 1) / threads;
    hello_kernel<<<blocks, threads>>>(d_out, n);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        message = std::string("kernel launch failed: ") + cudaGetErrorString(launch_err);
        cudaFree(d_out);
        std::free(h_out);
        return false;
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        message = std::string("cudaDeviceSynchronize failed: ") + cudaGetErrorString(sync_err);
        cudaFree(d_out);
        std::free(h_out);
        return false;
    }

    if (cudaMemcpy(h_out, d_out, sizeof(int) * n, cudaMemcpyDeviceToHost) != cudaSuccess) {
        message = "cudaMemcpy failed";
        cudaFree(d_out);
        std::free(h_out);
        return false;
    }

    for (int i = 0; i < n; ++i) {
        if (h_out[i] != i) {
            message = "kernel produced incorrect output";
            cudaFree(d_out);
            std::free(h_out);
            return false;
        }
    }

    cudaFree(d_out);
    std::free(h_out);
    message = "passed";
    return true;
}

// ---------------------------------------------------------------------------
// JSON helpers (hand-rolled, no dependencies)
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

static std::string json_string_or_null(const std::string& s) {
    if (s.empty()) return "null";
    return "\"" + json_escape(s) + "\"";
}

// ---------------------------------------------------------------------------
// Device reporting
// ---------------------------------------------------------------------------

struct DeviceInfo {
    int    index = 0;
    std::string name;
    std::string uuid;
    int    compute_capability_major = 0;
    int    compute_capability_minor = 0;
    int    multiprocessor_count = 0;
    int    max_threads_per_block = 0;
    int    max_threads_per_multiprocessor = 0;
    int    max_threads_per_block_dim[3] = {0, 0, 0};
    int    max_block_dim[3] = {0, 0, 0};
    int    max_grid_dim[3] = {0, 0, 0};
    int    warp_size = 0;
    int    registers_per_block = 0;
    int    registers_per_multiprocessor = 0;
    std::size_t shared_memory_per_block = 0;
    std::size_t shared_memory_per_multiprocessor = 0;
    std::size_t global_memory_bytes = 0;
    std::size_t constant_memory_bytes = 0;
    std::size_t l2_cache_bytes = 0;
    int    memory_clock_khz = 0;
    int    memory_bus_width_bits = 0;
    int    clock_rate_khz = 0;
    bool   unified_addressing = false;
    bool   concurrent_kernels = false;
    bool   kernel_smoke_test = false;
    std::string kernel_smoke_test_message;
};

static bool query_device(int index, DeviceInfo& out) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, index) != cudaSuccess) {
        return false;
    }

    out.index = index;
    out.name = prop.name;
    out.compute_capability_major = prop.major;
    out.compute_capability_minor = prop.minor;
    out.multiprocessor_count = prop.multiProcessorCount;
    out.max_threads_per_block = prop.maxThreadsPerBlock;
    out.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    for (int i = 0; i < 3; ++i) {
        out.max_threads_per_block_dim[i] = prop.maxThreadsDim[i];
        out.max_block_dim[i] = prop.maxGridSize[i];
    }
    out.warp_size = prop.warpSize;
    out.registers_per_block = prop.regsPerBlock;
    out.registers_per_multiprocessor = prop.regsPerMultiprocessor;
    out.shared_memory_per_block = prop.sharedMemPerBlock;
    out.shared_memory_per_multiprocessor = prop.sharedMemPerMultiprocessor;
    out.global_memory_bytes = prop.totalGlobalMem;
    out.constant_memory_bytes = prop.totalConstMem;
    out.l2_cache_bytes = prop.l2CacheSize;
    out.memory_clock_khz = prop.memoryClockRate;
    out.memory_bus_width_bits = prop.memoryBusWidth;
    out.clock_rate_khz = prop.clockRate;
    out.unified_addressing = prop.unifiedAddressing != 0;
    out.concurrent_kernels = prop.concurrentKernels != 0;

    // UUID (device 0 only reports a meaningful value on some drivers; if
    // the runtime refuses, we simply leave it empty and the JSON printer
    // emits "null").
    cudaUUID_t uuid{};
    if (cudaDeviceGetUuid(&uuid, index) == cudaSuccess) {
        char buf[64];
        std::snprintf(buf, sizeof(buf),
                      "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
                      "%02x%02x%02x%02x%02x%02x",
                      static_cast<unsigned char>(uuid.bytes[0]),
                      static_cast<unsigned char>(uuid.bytes[1]),
                      static_cast<unsigned char>(uuid.bytes[2]),
                      static_cast<unsigned char>(uuid.bytes[3]),
                      static_cast<unsigned char>(uuid.bytes[4]),
                      static_cast<unsigned char>(uuid.bytes[5]),
                      static_cast<unsigned char>(uuid.bytes[6]),
                      static_cast<unsigned char>(uuid.bytes[7]),
                      static_cast<unsigned char>(uuid.bytes[8]),
                      static_cast<unsigned char>(uuid.bytes[9]),
                      static_cast<unsigned char>(uuid.bytes[10]),
                      static_cast<unsigned char>(uuid.bytes[11]),
                      static_cast<unsigned char>(uuid.bytes[12]),
                      static_cast<unsigned char>(uuid.bytes[13]),
                      static_cast<unsigned char>(uuid.bytes[14]),
                      static_cast<unsigned char>(uuid.bytes[15]));
        out.uuid = buf;
    }

    out.kernel_smoke_test = run_kernel_smoke_test(index, out.kernel_smoke_test_message);
    return true;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

static void print_human_report(const std::vector<DeviceInfo>& devices,
                               int cuda_runtime_version,
                               int cuda_driver_version) {
    std::printf("=============================================================\n");
    std::printf(" CUDA DEVICE QUERY  (project: adaptive-cuda-thread-block)\n");
    std::printf("=============================================================\n\n");

    std::printf("CUDA runtime version : %d\n", cuda_runtime_version);
    std::printf("CUDA driver  version : %d\n", cuda_driver_version);
    std::printf("Device count         : %zu\n\n", devices.size());

    for (const auto& d : devices) {
        std::printf("-------------------------------------------------------------\n");
        std::printf(" Device %d: %s\n", d.index, d.name.c_str());
        std::printf("-------------------------------------------------------------\n");
        std::printf("  UUID                          : %s\n",
                    d.uuid.empty() ? "(unavailable)" : d.uuid.c_str());
        std::printf("  Compute capability            : %d.%d\n",
                    d.compute_capability_major, d.compute_capability_minor);
        std::printf("  Multiprocessors (SMs)         : %d\n", d.multiprocessor_count);
        std::printf("  Warp size                     : %d\n", d.warp_size);
        std::printf("  Max threads per block         : %d\n", d.max_threads_per_block);
        std::printf("  Max threads per SM            : %d\n", d.max_threads_per_multiprocessor);
        std::printf("  Max threads per block dim    : (%d, %d, %d)\n",
                    d.max_threads_per_block_dim[0],
                    d.max_threads_per_block_dim[1],
                    d.max_threads_per_block_dim[2]);
        std::printf("  Max grid size                : (%d, %d, %d)\n",
                    d.max_block_dim[0], d.max_block_dim[1], d.max_block_dim[2]);
        std::printf("  Registers per block           : %d\n", d.registers_per_block);
        std::printf("  Registers per SM              : %d\n", d.registers_per_multiprocessor);
        std::printf("  Shared memory per block       : %zu bytes\n", d.shared_memory_per_block);
        std::printf("  Shared memory per SM          : %zu bytes\n",
                    d.shared_memory_per_multiprocessor);
        std::printf("  Global memory                 : %.2f GiB\n",
                    static_cast<double>(d.global_memory_bytes) / (1024.0 * 1024.0 * 1024.0));
        std::printf("  Constant memory               : %zu bytes\n", d.constant_memory_bytes);
        std::printf("  L2 cache                      : %zu bytes\n", d.l2_cache_bytes);
        std::printf("  Memory clock                  : %d kHz\n", d.memory_clock_khz);
        std::printf("  Memory bus width              : %d bits\n", d.memory_bus_width_bits);
        std::printf("  SM clock                      : %d kHz\n", d.clock_rate_khz);
        std::printf("  Unified addressing            : %s\n",
                    d.unified_addressing ? "yes" : "no");
        std::printf("  Concurrent kernels            : %s\n",
                    d.concurrent_kernels ? "yes" : "no");
        std::printf("  Kernel smoke test             : %s (%s)\n",
                    d.kernel_smoke_test ? "PASSED" : "FAILED",
                    d.kernel_smoke_test_message.c_str());
        std::printf("\n");
    }
    std::printf("=============================================================\n");
}

static void print_json_report(const std::vector<DeviceInfo>& devices,
                              int cuda_runtime_version,
                              int cuda_driver_version) {
    std::printf("{\n");
    std::printf("  \"cuda_runtime_version\": %d,\n", cuda_runtime_version);
    std::printf("  \"cuda_driver_version\": %d,\n", cuda_driver_version);
    std::printf("  \"device_count\": %zu,\n", devices.size());
    std::printf("  \"devices\": [\n");

    for (std::size_t i = 0; i < devices.size(); ++i) {
        const auto& d = devices[i];
        std::printf("    {\n");
        std::printf("      \"index\": %d,\n", d.index);
        std::printf("      \"name\": \"%s\",\n", json_escape(d.name).c_str());
        std::printf("      \"uuid\": %s,\n", json_string_or_null(d.uuid).c_str());
        std::printf("      \"compute_capability\": \"%d.%d\",\n",
                    d.compute_capability_major, d.compute_capability_minor);
        std::printf("      \"multiprocessor_count\": %d,\n", d.multiprocessor_count);
        std::printf("      \"warp_size\": %d,\n", d.warp_size);
        std::printf("      \"max_threads_per_block\": %d,\n", d.max_threads_per_block);
        std::printf("      \"max_threads_per_multiprocessor\": %d,\n",
                    d.max_threads_per_multiprocessor);
        std::printf("      \"max_threads_per_block_dim\": [%d, %d, %d],\n",
                    d.max_threads_per_block_dim[0],
                    d.max_threads_per_block_dim[1],
                    d.max_threads_per_block_dim[2]);
        std::printf("      \"max_grid_size\": [%d, %d, %d],\n",
                    d.max_block_dim[0], d.max_block_dim[1], d.max_block_dim[2]);
        std::printf("      \"registers_per_block\": %d,\n", d.registers_per_block);
        std::printf("      \"registers_per_multiprocessor\": %d,\n",
                    d.registers_per_multiprocessor);
        std::printf("      \"shared_memory_per_block\": %zu,\n", d.shared_memory_per_block);
        std::printf("      \"shared_memory_per_multiprocessor\": %zu,\n",
                    d.shared_memory_per_multiprocessor);
        std::printf("      \"global_memory_bytes\": %zu,\n", d.global_memory_bytes);
        std::printf("      \"constant_memory_bytes\": %zu,\n", d.constant_memory_bytes);
        std::printf("      \"l2_cache_bytes\": %zu,\n", d.l2_cache_bytes);
        std::printf("      \"memory_clock_khz\": %d,\n", d.memory_clock_khz);
        std::printf("      \"memory_bus_width_bits\": %d,\n", d.memory_bus_width_bits);
        std::printf("      \"clock_rate_khz\": %d,\n", d.clock_rate_khz);
        std::printf("      \"unified_addressing\": %s,\n",
                    d.unified_addressing ? "true" : "false");
        std::printf("      \"concurrent_kernels\": %s,\n",
                    d.concurrent_kernels ? "true" : "false");
        std::printf("      \"kernel_smoke_test\": %s,\n",
                    d.kernel_smoke_test ? "\"passed\"" : "\"failed\"");
        std::printf("      \"kernel_smoke_test_message\": \"%s\"\n",
                    json_escape(d.kernel_smoke_test_message).c_str());
        std::printf("    }%s\n", (i + 1 < devices.size()) ? "," : "");
    }

    std::printf("  ]\n");
    std::printf("}\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    bool want_json = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0) {
            want_json = true;
        } else if (std::strcmp(argv[i], "--help") == 0 ||
                   std::strcmp(argv[i], "-h") == 0) {
            std::printf("Usage: %s [--json]\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            std::fprintf(stderr, "Usage: %s [--json]\n", argv[0]);
            return 2;
        }
    }

    int cuda_runtime_version = 0;
    int cuda_driver_version = 0;
    cudaRuntimeGetVersion(&cuda_runtime_version);
    cudaDriverGetVersion(&cuda_driver_version);

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        if (!want_json) {
            std::fprintf(stderr,
                "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
            std::fprintf(stderr,
                "No CUDA device is visible from this process.  This is expected\n"
                "on a machine without an NVIDIA GPU (e.g. a Windows laptop);\n"
                "run the CUDA parts on Google Colab.  See README.md.\n");
        } else {
            std::printf("{\"device_count\": 0, \"devices\": [], "
                        "\"error\": \"%s\"}\n",
                        json_escape(cudaGetErrorString(err)).c_str());
        }
        return 1;
    }

    if (device_count == 0) {
        if (!want_json) {
            std::fprintf(stderr,
                "No CUDA devices reported.  Run the CUDA parts on Google Colab.\n");
        } else {
            std::printf("{\"device_count\": 0, \"devices\": []}\n");
        }
        return 1;
    }

    std::vector<DeviceInfo> devices;
    devices.reserve(static_cast<std::size_t>(device_count));
    for (int i = 0; i < device_count; ++i) {
        DeviceInfo info{};
        if (!query_device(i, info)) {
            std::fprintf(stderr, "Failed to query device %d\n", i);
            return 2;
        }
        devices.push_back(info);
    }

    if (want_json) {
        print_json_report(devices, cuda_runtime_version, cuda_driver_version);
    } else {
        print_human_report(devices, cuda_runtime_version, cuda_driver_version);
    }

    for (const auto& d : devices) {
        if (d.kernel_smoke_test) {
            return 0;
        }
    }
    return 1;
}