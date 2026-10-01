# Adaptive CUDA Thread-Block Configuration for Workload-Aware GPU Optimization

A research prototype that studies how the CUDA **thread-block size** affects kernel
performance, and that learns to *predict* a good block size from workload
characteristics instead of using a fixed value or an exhaustive search.

> **Status: Phases 1, 2, 3 and 4 complete.**
> The repository contains the project skeleton, the configuration system, a CUDA
> device query / correctness tool, a first CUDA workload with a correctness
> checker, and a timing / repetition harness for that workload.  No machine
> learning has been added yet, and **no performance number is reported in this
> repository** unless it was produced by the code in `src/` running on real
> hardware; when that happens, the environment metadata is stored next to the
> measurement (see `data/README.md`).

---

## 1. Research motivation

CUDA kernels are launched with an explicit *execution configuration*:

```cpp
kernel<<<grid_size, block_size>>>(...);
```

The `block_size` (threads per block) is one of the few tuning knobs available
for **every** kernel without rewriting it, and it interacts with:

* the number of resident warps per streaming multiprocessor (occupancy),
* register-file and shared-memory pressure per block,
* the amount of exploitable memory-level parallelism,
* tail effects when `N` is not a multiple of the block size,
* per-thread divergence in kernels with boundary conditions.

Practitioners usually pick 128, 256 or 512 out of habit.  An **exhaustive
search** over all legal block sizes finds a better answer but costs a full
sweep of measurements per kernel.  This project investigates whether a cheap,
workload-aware model can predict a near-optimal block size directly.

The eventual comparison is:

1. Fixed 64
2. Fixed 128
3. Fixed 256
4. Fixed 512
5. Exhaustive search (upper bound / oracle)
6. **Proposed adaptive method** (workload features → ML model → block size)

---

## 2. Current implementation status

| Phase | Description | Status |
|---|---|---|
| 1 | Project architecture and repository setup | ✅ done |
| 2 | CUDA environment detection and validation | ✅ done |
| 3 | First CUDA workload — vector addition (correctness) | ✅ done |
| 4 | Thread-block timing and repetition harness | ✅ done |
| 5 | Additional CUDA workloads (reduction, stencil) | ⬜ not started |
| 6 | Dataset generation | ⬜ not started |
| 7 | Workload feature extraction | ⬜ not started |
| 8 | Fixed and exhaustive baselines | ⬜ not started |
| 9 | Adaptive ML model | ⬜ not started |
| 10 | Adaptive CUDA execution | ⬜ not started |
| 11 | Complete experiment automation | ⬜ not started |
| 12 | Statistical analysis | ⬜ not started |
| 13 | Visualization | ⬜ not started |
| 14 | Research evaluation | ⬜ not started |
| 15 | IEEE paper preparation | ⬜ not started |

---

## 3. Architecture

```
config/experiment.yaml          single source of truth for experiment parameters
        │
        ▼
src/config.py                   load + validate the configuration (no hard-coded values)
        │
        ├──────────────► src/cuda/*.cu              (CUDA C++, built by CMake)
        │                        │
        │                        │  --json
        │                        ▼
        └──────────────► src/profiling/gpu_info.py  (Python, stdlib only)
                                 │
                                 ▼
                    unified GPU environment report
```

Later phases plug into the same skeleton:

```
CUDA workloads (src/cuda)
   → thread-block benchmarking (src/benchmarking, Phase 6)
   → performance dataset (data/raw, Phase 6)
   → feature extraction (src/features, Phase 7)
   → optimal-configuration labels (src/optimizer, Phase 8)
   → ML model (src/model, Phase 9)
   → adaptive prediction (Phase 10)
   → experimental evaluation (src/analysis, Phases 11–14)
```

---

## 4. Current executable inventory

The repository builds the following CUDA executables on a GPU host.  On a
machine without the CUDA toolkit, CMake configures successfully and simply
skips these targets (see section 6).

| Executable | Phase | Purpose |
|---|---|---|
| `device_query` | 2 | Read every hardware limit from the CUDA runtime and run a trivial smoke-test kernel. |
| `vector_add` | 3 | Run one `y = a + b` launch and verify the result against a CPU reference. |
| `benchmark_vector_add` | 4 | Time the same vector-add kernel over warm-up + measurement launches, repeated `repetitions` times, and emit JSON. |

All three accept `--json` for machine-readable output and are designed to be
driven by Python in later phases without recompilation.

---

## 5. Requirements

### Development machine (Windows 11 / Linux, no GPU required)

* Python 3.12
* CMake ≥ 3.21 (3.24+ recommended so that `CMAKE_CUDA_ARCHITECTURES=native` works)
* Git
* A C++17 compiler (MSVC on Windows, GCC/Clang on Linux)

### GPU machine (Google Colab or any Linux box with an NVIDIA GPU)

* An NVIDIA GPU
* NVIDIA driver
* CUDA Toolkit with `nvcc` (Colab provides this)
* CMake ≥ 3.21

### CUDA requirement

**CUDA execution is not possible without an NVIDIA GPU.**  This project
therefore splits cleanly:

* **Windows** is used for editing, configuration, tests and analysis.
* **Linux / Google Colab** is used for building and running the CUDA targets.

`CMakeLists.txt` detects CUDA with `check_language(CUDA)`.  If `nvcc` is
missing, configuration still succeeds and simply skips the CUDA targets with a
clear diagnostic message.  No GPU model, compute capability or architecture is
hard-coded anywhere.

On Linux/Colab, where the CUDA context is not necessarily initialised at
configure time, `src/cuda/CMakeLists.txt` additionally asks `nvidia-smi` for the
visible GPU's compute capability and appends it to `CMAKE_CUDA_ARCHITECTURES`
if it is not already present.  No architecture is ever hard-coded.

---

## 6. Installation

```bash
git clone <your-repository-url> adaptive-cuda-thread-block
cd adaptive-cuda-thread-block

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

`numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib` are only needed from
Phase 6 onward; they are listed now so the environment is stable across phases.
Phases 1–4 only need `PyYAML` and `pytest`.

---

## 7. Build

### A. Linux / Google Colab

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

Executables: `build/bin/device_query`, `build/bin/vector_add`,
`build/bin/benchmark_vector_add`.

### B. Windows with CUDA Toolkit installed

```bat
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release --parallel
```

Executables: `build\bin\Release\device_query.exe` etc.

### C. Windows without CUDA

```bat
cmake -S . -B build
cmake --build build --config Release
```

Expected output:

```
-- CUDA toolkit NOT found - CUDA targets will be skipped.
--   Install the CUDA Toolkit (with nvcc) and re-run CMake, or run the CUDA
--   parts on a GPU machine such as Google Colab.  See README.md.
```

This is the normal state on the development laptop.  CUDA work happens on
Colab.

### Architecture selection

By default CMake uses `CMAKE_CUDA_ARCHITECTURES=native`.  If the visible GPU is
not correctly detected (typical on fresh Colab runtimes), the value is
augmented from `nvidia-smi` at configure time, as described in section 5.  To
override explicitly:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75
```

---

## 8. Running the tools

```bash
# Environment probe (Phase 2) — works with or without a GPU
python -m src.profiling.gpu_info
python -m src.profiling.gpu_info --json

# CUDA device limits and kernel smoke test (Phase 2) — GPU host only
./build/bin/device_query
./build/bin/device_query --json

# Vector-add correctness check (Phase 3) — GPU host only
./build/bin/vector_add
./build/bin/vector_add --n 1000000 --block-size 384      # tail-guard case
./build/bin/vector_add --json

# Vector-add timing harness (Phase 4) — GPU host only
./build/bin/benchmark_vector_add                                  # config defaults
./build/bin/benchmark_vector_add --n 4194304 --block-size 128
./build/bin/benchmark_vector_add --warmup 5 --iterations 20 --repetitions 3
```

`python -m src.profiling.gpu_info` exits with code `0` when an NVIDIA GPU was
detected and `1` when it was not.  `vector_add` exits `0` on PASS, `1` on FAIL,
`2` on configuration error and `3` on CUDA runtime error.  `benchmark_vector_add`
exits `0` on success, `2` on configuration error and `3` on CUDA runtime error.

The two benchmark tools **do not write any file**.  Their only output is a
single JSON object on stdout, plus an exit code.  Writing datasets is Phase 6's
responsibility.

---

## 9. Google Colab execution

Open `notebooks/01_environment_test.ipynb` in Colab with a GPU runtime and run
all cells.  The notebook:

1. prints the Python version,
2. runs `nvidia-smi`,
3. runs `nvcc --version`,
4. checks the CMake version,
5. locates (or clones) the repository,
6. installs the Python requirements,
7. configures and builds the CUDA targets,
8. runs `device_query` and `python -m src.profiling.gpu_info`,
9. prints a readiness verdict.

Because a Colab runtime wipes `/content` on restart, every fresh session begins
with a clone + build.  From Phase 6 onward, datasets and results will live in
Google Drive so they survive restarts.

---

## 10. Project structure

```
adaptive-cuda-thread-block/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── environment.yml
├── CMakeLists.txt
├── conftest.py
│
├── config/
│   └── experiment.yaml
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── cuda/
│   │   ├── CMakeLists.txt
│   │   ├── device_query.cu
│   │   ├── vector_add.cu
│   │   └── benchmark_vector_add.cu
│   ├── benchmarking/__init__.py
│   ├── profiling/
│   │   ├── __init__.py
│   │   └── gpu_info.py
│   ├── features/__init__.py
│   ├── optimizer/__init__.py
│   ├── model/__init__.py
│   └── analysis/__init__.py
│
├── include/
├── benchmarks/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── models/
├── experiments/
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── logs/
│
├── notebooks/
│   └── 01_environment_test.ipynb
│
├── tests/
│   ├── test_config.py
│   └── test_gpu_info.py
│
└── docs/
```

---

## 11. Research roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Project architecture and repository setup | ✅ |
| 2 | CUDA environment detection and validation | ✅ |
| 3 | First CUDA workload — vector addition (correctness) | ✅ |
| 4 | Thread-block timing and repetition harness | ✅ |
| 5 | Additional CUDA workloads (reduction, stencil) | ⬜ |
| 6 | Dataset generation | ⬜ |
| 7 | Workload feature extraction | ⬜ |
| 8 | Fixed and exhaustive baselines | ⬜ |
| 9 | Adaptive ML model | ⬜ |
| 10 | Adaptive CUDA execution | ⬜ |
| 11 | Complete experiment automation | ⬜ |
| 12 | Statistical analysis | ⬜ |
| 13 | Visualization | ⬜ |
| 14 | Research evaluation | ⬜ |
| 15 | IEEE paper preparation | ⬜ |

Every phase is implemented, tested and verified on real hardware before the
next one begins.  No benchmark result is committed to this repository unless
it was produced by the code in `src/` and is stored together with the GPU
model, driver version, CUDA version and the exact configuration used.

---

## 12. License

MIT — see `LICENSE`.