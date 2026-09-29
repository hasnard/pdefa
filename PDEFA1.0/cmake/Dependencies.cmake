# ============================================================
#  Dependencies.cmake
#  Threading ve diğer opsiyonel bağımlılıkları bulur.
# ============================================================

include(FetchContent)

# ---------- Threads (zorunlu) ----------
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(engine_deps INTERFACE)
target_link_libraries(engine_deps INTERFACE Threads::Threads)

# ---------- OpenMP (fallback threading) ----------
if(ENGINE_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(engine_deps INTERFACE OpenMP::OpenMP_CXX)
        target_compile_definitions(engine_deps INTERFACE ENGINE_USE_OPENMP=1)
        message(STATUS "[Deps] OpenMP enabled")
    else()
        message(STATUS "[Deps] OpenMP not found, using std::thread pool")
    endif()
endif()

# ---------- TBB (isteğe bağlı) ----------
if(ENGINE_ENABLE_TBB)
    find_package(TBB QUIET)
    if(TBB_FOUND)
        target_link_libraries(engine_deps INTERFACE TBB::tbb)
        target_compile_definitions(engine_deps INTERFACE ENGINE_USE_TBB=1)
        message(STATUS "[Deps] TBB enabled")
    else()
        message(WARNING "[Deps] TBB requested but not found")
    endif()
endif()

# ---------- GoogleTest (testler için) ----------
if(ENGINE_BUILD_TESTS)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "[Deps] Fetching GoogleTest...")
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
            GIT_SHALLOW    TRUE
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
endif()

# ---------- Google Benchmark ----------
if(ENGINE_BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
    if(NOT benchmark_FOUND)
        message(STATUS "[Deps] Fetching Google Benchmark...")
        FetchContent_Declare(
            benchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG        v1.8.3
            GIT_SHALLOW    TRUE
        )
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(benchmark)
    endif()
endif()

# ---------- OpenCV (opsiyonel, tools için) ----------
option(ENGINE_ENABLE_OPENCV "Enable OpenCV for tools (image loading)" OFF)
if(ENGINE_ENABLE_OPENCV)
    find_package(OpenCV QUIET)
    if(OpenCV_FOUND)
        target_link_libraries(engine_deps INTERFACE opencv_core opencv_imgproc opencv_imgcodecs)
        target_compile_definitions(engine_deps INTERFACE ENGINE_USE_OPENCV=1)
        message(STATUS "[Deps] OpenCV ${OpenCV_VERSION} enabled")
    endif()
endif()