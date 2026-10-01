# Adaptive CUDA Thread-Block Configuration for Workload-Aware GPU Optimization

A research prototype that studies how the CUDA **thread-block size** affects kernel
performance, and that learns to *predict* a good block size from workload
characteristics instead of using a fixed value or an exhaustive search.

> **Status: Phase 1 + Phase 2 complete.**
> The repository contains the project skeleton, the configuration system, a CUDA
> device query / correctness tool, and a Python GPU environment reporter.
> No benchmarks have been run yet and **no performance numbers are reported
> anywhere in this repository** until they are measured on real hardware.

---

## 1. Research motivation

CUDA kernels are launched with an explicit *execution configuration*:

```cpp
kernel<<<grid_size, block_size>>>(...);