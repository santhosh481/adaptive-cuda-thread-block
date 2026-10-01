# =============================================================================
#  Adaptive CUDA Thread-Block Configuration for Workload-Aware GPU Optimization
#
#  Top-level CMake configuration.
#
#  Design rules:
#    * CUDA is DETECTED, never assumed.
#    * No GPU architecture is hard-coded.
#    * If CUDA is missing, configuration still succeeds and the CUDA targets
#      are skipped with an explicit diagnostic message.
#
#  Typical usage:
#      cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#      cmake --build build --config Release --parallel
# =============================================================================

cmake_minimum_required(VERSION 3.21)

project(adaptive_cuda_thread_block
        VERSION 0.1.0
        DESCRIPTION "Adaptive CUDA Thread-Block Configuration for Workload-Aware GPU Optimization"
        LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Global build settings
# -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
endif()

# Every executable ends up in <build>/bin so that Python tooling and shell
# scripts can locate it without knowing which CMake generator was used.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(_acu_config IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_acu_config} "${CMAKE_BINARY_DIR}/bin")
endforeach()

option(ACUDA_REQUIRE_CUDA
       "Abort CMake configuration when no CUDA toolkit is found" OFF)
option(ACUDA_BUILD_CUDA_TOOLS
       "Build the CUDA diagnostic / correctness executables" ON)

# -----------------------------------------------------------------------------
# CUDA detection
# -----------------------------------------------------------------------------

include(CheckLanguage)
check_language(CUDA)

set(ACUDA_CUDA_AVAILABLE OFF)

if(CMAKE_CUDA_COMPILER)

    enable_language(CUDA)
    set(ACUDA_CUDA_AVAILABLE ON)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    # Do NOT hard-code an architecture. Prefer the GPU that is visible at
    # configure time; the user can always override explicitly:
    #     -DCMAKE_CUDA_ARCHITECTURES=75
    #     -DCMAKE_CUDA_ARCHITECTURES=80;86
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING
                "CUDA architectures to compile for (e.g. '75;86', or 'native')")
        else()
            message(STATUS
                "CMake < 3.24: CMAKE_CUDA_ARCHITECTURES left unset (nvcc default). "
                "Pass -DCMAKE_CUDA_ARCHITECTURES=<arch> to target a specific GPU.")
        endif()
    endif()

    message(STATUS "CUDA toolkit found")
    message(STATUS "  CMAKE_CUDA_COMPILER         : ${CMAKE_CUDA_COMPILER}")
    message(STATUS "  CMAKE_CUDA_COMPILER_VERSION : ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "  CMAKE_CUDA_ARCHITECTURES    : ${CMAKE_CUDA_ARCHITECTURES}")

else()

    message(STATUS "CUDA toolkit NOT found - CUDA targets will be skipped.")
    message(STATUS "  Install the CUDA Toolkit (which provides nvcc) and re-run CMake,")
    message(STATUS "  or run the CUDA parts on a GPU machine such as Google Colab.")
    message(STATUS "  See README.md section 'Google Colab execution'.")

    if(ACUDA_REQUIRE_CUDA)
        message(FATAL_ERROR
            "ACUDA_REQUIRE_CUDA=ON but no CUDA toolkit was found on this machine.")
    endif()

endif()

# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------

if(ACUDA_CUDA_AVAILABLE AND ACUDA_BUILD_CUDA_TOOLS)
    add_subdirectory(src/cuda)
elseif(ACUDA_CUDA_AVAILABLE)
    message(STATUS "ACUDA_BUILD_CUDA_TOOLS=OFF - skipping CUDA executables.")
endif()

# -----------------------------------------------------------------------------
# Configuration summary
# -----------------------------------------------------------------------------

message(STATUS "")
message(STATUS "================ Configuration summary ================")
message(STATUS "  Project            : ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "  Build type         : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler       : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CUDA available     : ${ACUDA_CUDA_AVAILABLE}")
message(STATUS "  Runtime output dir : ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "=======================================================")
message(STATUS "")