# ============================================================
#  SimdDetection.cmake
#  Detects CPU features at configure time, sets the corresponding
#  compiler flags and preprocessor macros.
# ============================================================

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# ---------- AVX2 ----------
if(ENGINE_ENABLE_AVX2)
    if(MSVC)
        check_cxx_compiler_flag("/arch:AVX2" HAS_AVX2_MSVC)
        if(HAS_AVX2_MSVC)
            set(ENGINE_HAS_AVX2 TRUE)
            target_compile_options(engine_flags INTERFACE /arch:AVX2)
            message(STATUS "[SIMD] AVX2 enabled (/arch:AVX2)")
        endif()
    else()
        check_cxx_compiler_flag("-mavx2" HAS_AVX2_FLAG)
        if(HAS_AVX2_FLAG)
            set(ENGINE_HAS_AVX2 TRUE)
            target_compile_options(engine_flags INTERFACE -mavx2)
            message(STATUS "[SIMD] AVX2 enabled (-mavx2)")
        endif()
    endif()
endif()

# ---------- FMA ----------
if(ENGINE_ENABLE_FMA)
    if(MSVC)
        # MSVC bundles FMA with /arch:AVX2
        if(ENGINE_HAS_AVX2)
            set(ENGINE_HAS_FMA TRUE)
            message(STATUS "[SIMD] FMA enabled (bundled with /arch:AVX2)")
        endif()
    else()
        check_cxx_compiler_flag("-mfma" HAS_FMA_FLAG)
        if(HAS_FMA_FLAG)
            set(ENGINE_HAS_FMA TRUE)
            target_compile_options(engine_flags INTERFACE -mfma)
            message(STATUS "[SIMD] FMA enabled (-mfma)")
        endif()
    endif()
endif()

# ---------- AVX-512 ----------
if(ENGINE_ENABLE_AVX512 AND NOT MSVC)
    check_cxx_compiler_flag("-mavx512f"    HAS_AVX512F)
    check_cxx_compiler_flag("-mavx512vl"   HAS_AVX512VL)
    check_cxx_compiler_flag("-mavx512bw"   HAS_AVX512BW)
    check_cxx_compiler_flag("-mavx512dq"   HAS_AVX512DQ)
    check_cxx_compiler_flag("-mavx512vnni" HAS_AVX512VNNI)

    if(HAS_AVX512F AND HAS_AVX512VL AND HAS_AVX512BW AND HAS_AVX512DQ)
        set(ENGINE_HAS_AVX512 TRUE)
        target_compile_options(engine_flags INTERFACE
            -mavx512f -mavx512vl -mavx512bw -mavx512dq)
        message(STATUS "[SIMD] AVX-512 enabled")
    endif()

    if(HAS_AVX512VNNI)
        set(ENGINE_HAS_AVX512VNNI TRUE)
        target_compile_options(engine_flags INTERFACE -mavx512vnni)
        message(STATUS "[SIMD] AVX-512 VNNI (INT8) enabled")
    endif()
endif()

# ---------- ARM NEON ----------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    set(ENGINE_HAS_NEON TRUE)
    message(STATUS "[SIMD] ARM NEON detected")
endif()

# ---------- Global definitions ----------
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