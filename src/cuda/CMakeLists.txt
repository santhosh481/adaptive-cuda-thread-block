# =============================================================================
#  CUDA workloads and diagnostic tools
#
#  This directory is only added by the top-level CMakeLists.txt when a CUDA
#  toolkit was detected, so we can safely assume CUDA is available here.
#
#  Targets
#  -------
#    device_query   (Phase 2)  environment / correctness probe
#    vector_add     (Phase 3)  first workload - correctness check only
#
#  Phase 4+ will add the benchmarking harness targets.
# =============================================================================

# -----------------------------------------------------------------------------
# Common settings applied to every CUDA executable in this directory.
# -----------------------------------------------------------------------------

function(acu_configure_cuda_target target)
    target_include_directories(${target}
        PRIVATE "${PROJECT_SOURCE_DIR}/include")

    set_target_properties(${target} PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RUNTIME_LIBRARY Shared)

    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>)
    else()
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
    endif()
endfunction()

# -----------------------------------------------------------------------------
# device_query  (Phase 2)
# -----------------------------------------------------------------------------

add_executable(device_query device_query.cu)
acu_configure_cuda_target(device_query)

# -----------------------------------------------------------------------------
# vector_add  (Phase 3)
#
# One workload, one launch, correctness check only.  No timing.  The kernel
# inside this file is the one Phase 4 will sweep over block sizes.
# -----------------------------------------------------------------------------

add_executable(vector_add vector_add.cu)
acu_configure_cuda_target(vector_add)