# =============================================================================
#  CUDA workloads and diagnostic tools
#
#  This directory is only added by the top-level CMakeLists.txt when a CUDA
#  toolkit was detected, so we can safely assume CUDA is available here.
#
#  Targets
#  -------
#    device_query            (Phase 2)  environment / correctness probe
#    vector_add              (Phase 3)  first workload - correctness only
#    benchmark_vector_add    (Phase 4)  timed measurement of vector_add
#
#  Architecture selection
#  ----------------------
#  The top-level CMakeLists.txt sets CMAKE_CUDA_ARCHITECTURES to "native"
#  when CMake >= 3.24 is used.  On Linux build hosts where the CUDA context
#  has not been initialised yet (typical on Google Colab), "native" can
#  silently fall back to the lowest architecture the toolkit still supports
#  rather than the architecture of the physical GPU.  We therefore check,
#  at configure time, whether the CMAKE_CUDA_ARCHITECTURES value contains
#  the visible GPU's compute capability, and if not we ask nvidia-smi for
#  the correct value and append it.  No architecture is ever hard-coded.
# =============================================================================

# -----------------------------------------------------------------------------
# Runtime-aware architecture detection.
# -----------------------------------------------------------------------------

function(acu_ensure_native_cuda_arch)
    if(NOT CMAKE_CUDA_COMPILER)
        return()
    endif()

    find_program(ACUDA_NVIDIA_SMI nvidia-smi)
    if(NOT ACUDA_NVIDIA_SMI)
        message(STATUS "nvidia-smi not found - leaving CMAKE_CUDA_ARCHITECTURES as '${CMAKE_CUDA_ARCHITECTURES}'")
        return()
    endif()

    execute_process(
        COMMAND "${ACUDA_NVIDIA_SMI}"
                --query-gpu=compute_cap
                --format=csv,noheader
        OUTPUT_VARIABLE _acu_cc_raw
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _acu_cc_rc
        ERROR_QUIET)

    if(NOT _acu_cc_rc EQUAL 0 OR _acu_cc_raw STREQUAL "")
        message(STATUS "nvidia-smi did not report a compute capability - leaving CMAKE_CUDA_ARCHITECTURES as '${CMAKE_CUDA_ARCHITECTURES}'")
        return()
    endif()

    string(REGEX REPLACE "[\r\n].*$" "" _acu_cc "${_acu_cc_raw}")
    string(REPLACE "." "" _acu_cc_nodot "${_acu_cc}")

    if(_acu_cc_nodot STREQUAL "")
        message(STATUS "could not parse compute capability '${_acu_cc}' - leaving CMAKE_CUDA_ARCHITECTURES as '${CMAKE_CUDA_ARCHITECTURES}'")
        return()
    endif()

    set(_acu_have_cc FALSE)
    foreach(_acu_entry IN LISTS CMAKE_CUDA_ARCHITECTURES)
        if(_acu_entry STREQUAL "${_acu_cc_nodot}")
            set(_acu_have_cc TRUE)
        endif()
    endforeach()

    if(_acu_have_cc)
        message(STATUS "CMAKE_CUDA_ARCHITECTURES already contains sm_${_acu_cc_nodot} (from nvidia-smi)")
    else()
        message(STATUS
            "nvidia-smi reports compute capability ${_acu_cc} (sm_${_acu_cc_nodot}); "
            "appending to CMAKE_CUDA_ARCHITECTURES (was '${CMAKE_CUDA_ARCHITECTURES}')")
        set(CMAKE_CUDA_ARCHITECTURES
            "${CMAKE_CUDA_ARCHITECTURES};${_acu_cc_nodot}"
            CACHE STRING "CUDA architectures to compile for" FORCE)
        message(STATUS "CMAKE_CUDA_ARCHITECTURES is now '${CMAKE_CUDA_ARCHITECTURES}'")
    endif()
endfunction()

acu_ensure_native_cuda_arch()

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
# -----------------------------------------------------------------------------

add_executable(vector_add vector_add.cu)
acu_configure_cuda_target(vector_add)

# -----------------------------------------------------------------------------
# benchmark_vector_add  (Phase 4)
#
# Times the same vector_add kernel over `warmup` + `iterations` launches,
# repeats `repetitions` times, and emits a single JSON object.  No file is
# written; the Phase 6 experiment runner will parse stdout.
# -----------------------------------------------------------------------------

add_executable(benchmark_vector_add benchmark_vector_add.cu)
acu_configure_cuda_target(benchmark_vector_add)