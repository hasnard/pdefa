# ============================================================
#  SimdDetection.cmake
#  Derleme zamanında CPU özelliklerini tespit eder,
#  ilgili flag'leri ve makroları tanımlar.
# ============================================================

include(CheckCXXSourceCompiles)

# ---------- AVX2 ----------
if(ENGINE_ENABLE_AVX2 AND NOT MSVC)
    check_cxx_compiler_flag("-mavx2" HAS_AVX2_FLAG)
    if(HAS_AVX2_FLAG)
        set(ENGINE_HAS_AVX2 TRUE)
        message(STATUS "[SIMD] AVX2 supported by compiler")
    endif()
elseif(ENGINE_ENABLE_AVX2 AND MSVC)
    check_cxx_compiler_flag("/arch:AVX2" HAS_AVX2_MSVC)
    if(HAS_AVX2_MSVC)
        set(ENGINE_HAS_AVX2 TRUE)
    endif()
endif()

# ---------- FMA ----------
if(ENGINE_ENABLE_FMA AND NOT MSVC)
    check_cxx_compiler_flag("-mfma" HAS_FMA_FLAG)
    if(HAS_FMA_FLAG)
        set(ENGINE_HAS_FMA TRUE)
    endif()
endif()

# ---------- AVX-512 ----------
if(ENGINE_ENABLE_AVX512 AND NOT MSVC)
    check_cxx_compiler_flag("-mavx512f" HAS_AVX512F)
    check_cxx_compiler_flag("-mavx512vl" HAS_AVX512VL)
    check_cxx_compiler_flag("-mavx512bw" HAS_AVX512BW)
    check_cxx_compiler_flag("-mavx512vnni" HAS_AVX512VNNI)

    if(HAS_AVX512F AND HAS_AVX512VL AND HAS_AVX512BW)
        set(ENGINE_HAS_AVX512 TRUE)
        message(STATUS "[SIMD] AVX-512 supported by compiler")
    endif()
    if(HAS_AVX512VNNI)
        set(ENGINE_HAS_AVX512VNNI TRUE)
        message(STATUS "[SIMD] AVX-512 VNNI (INT8) supported")
    endif()
endif()

# ---------- ARM NEON ----------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    set(ENGINE_HAS_NEON TRUE)
    message(STATUS "[SIMD] ARM NEON detected")
endif()

# ---------- Global tanımlar ----------
add_library(engine_simd INTERFACE)
target_link_libraries(engine_simd INTERFACE engine_flags)

if(ENGINE_HAS_AVX2)
    target_compile_definitions(engine_simd INTERFACE ENGINE_USE_AVX2=1)
endif()

if(ENGINE_HAS_FMA)
    target_compile_definitions(engine_simd INTERFACE ENGINE_USE_FMA=1)
endif()

if(ENGINE_HAS_AVX512)
    target_compile_definitions(engine_simd INTERFACE ENGINE_USE_AVX512=1)
endif()

if(ENGINE_HAS_AVX512VNNI)
    target_compile_definitions(engine_simd INTERFACE ENGINE_USE_AVX512VNNI=1)
endif()

if(ENGINE_HAS_NEON)
    target_compile_definitions(engine_simd INTERFACE ENGINE_USE_NEON=1)
endif()