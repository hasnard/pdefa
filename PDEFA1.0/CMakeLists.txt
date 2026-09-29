cmake_minimum_required(VERSION 3.20)

# ---- Proje bilgisi ----
project(engine-ai
    VERSION 1.0.3
    DESCRIPTION "CPU-optimized custom inference engine"
    LANGUAGES CXX C
)

# ---- Global ayarlar ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

# Default build tipi
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ---- Kullanıcı seçenekleri ----
option(ENGINE_BUILD_TESTS        "Build unit/integration tests"       ON)
option(ENGINE_BUILD_BENCHMARKS   "Build benchmarks"                    ON)
option(ENGINE_BUILD_EXAMPLES     "Build examples"                      ON)
option(ENGINE_BUILD_SHARED       "Build as shared library"             OFF)
option(ENGINE_BUILD_TOOLS        "Build tools (converter, profiler)"   ON)

option(ENGINE_ENABLE_NATIVE      "Use -march=native"                   OFF)
option(ENGINE_ENABLE_AVX2        "Enable AVX2 kernels"                 ON)
option(ENGINE_ENABLE_AVX512      "Enable AVX-512 kernels"              ON)
option(ENGINE_ENABLE_FMA         "Enable FMA"                          ON)
option(ENGINE_ENABLE_OPENMP      "Use OpenMP fallback (if no TBB)"     ON)
option(ENGINE_ENABLE_TBB         "Use Intel TBB for threading"         OFF)
option(ENGINE_ENABLE_LTO         "Enable Link Time Optimization"       ON)
option(ENGINE_ENABLE_NATIVE_ARCH "Alias for ENGINE_ENABLE_NATIVE"      OFF)

option(ENGINE_WARNINGS_AS_ERRORS "Treat warnings as errors"            OFF)
option(ENGINE_PROFILING          "Enable per-op profiling"             OFF)

# ---- Çıktı dizinleri ----
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ---- CMake modüllerini bul ----
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(CompilerFlags)
include(SimdDetection)
include(Dependencies)

# ---- Sürüm header'ı oluştur ----
configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/version.hpp.in"
    "${CMAKE_BINARY_DIR}/generated/engine/version.hpp"
    @ONLY
)

# ---- Alt dizinler ----
add_subdirectory(src)
add_subdirectory(third_party)

if(ENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ENGINE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(ENGINE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(ENGINE_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# ---- Özet çıktı ----
message(STATUS "")
message(STATUS "================ Engine-AI Configuration ================")
message(STATUS "  Version          : ${PROJECT_VERSION}")
message(STATUS "  Build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard     : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler         : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Architecture     : ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Shared library   : ${ENGINE_BUILD_SHARED}")
message(STATUS "  Native optim.    : ${ENGINE_ENABLE_NATIVE}")
message(STATUS "  AVX2             : ${ENGINE_ENABLE_AVX2}")
message(STATUS "  AVX-512          : ${ENGINE_ENABLE_AVX512}")
message(STATUS "  FMA              : ${ENGINE_ENABLE_FMA}")
message(STATUS "  LTO              : ${ENGINE_ENABLE_LTO}")
message(STATUS "  OpenMP           : ${ENGINE_ENABLE_OPENMP}")
message(STATUS "  TBB              : ${ENGINE_ENABLE_TBB}")
message(STATUS "  Tests            : ${ENGINE_BUILD_TESTS}")
message(STATUS "  Benchmarks       : ${ENGINE_BUILD_BENCHMARKS}")
message(STATUS "  Examples         : ${ENGINE_BUILD_EXAMPLES}")
message(STATUS "  Tools            : ${ENGINE_BUILD_TOOLS}")
message(STATUS "=========================================================")
message(STATUS "")