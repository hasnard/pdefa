# ============================================================
#  src/CMakeLists.txt — Motor build (final)
# ============================================================

set(ENGINE_CORE_SOURCES
    core/tensor.cpp
    core/model.cpp
    core/session.cpp

    # Session Alt Parcalari (session.cpp tarafindan include edilir)
    core/session_parts/session_helpers.inc
    core/session_parts/session_memory.inc
    core/session_parts/session_fusion.inc
    core/session_parts/session_ops_nn.inc
    core/session_parts/session_ops_math.inc
    core/session_parts/session_ops_shape.inc

    memory/aligned_allocator.cpp
    memory/arena.cpp
    memory/memory_plan.cpp

    runtime/cpu_info.cpp
    runtime/thread_pool.cpp
    runtime/affinity.cpp
    runtime/scheduler.cpp

    kernels/kernel_registry.cpp
    kernels/generic/matmul_ref.cpp

    ops/matmul.cpp
    ops/activation.cpp
    ops/conv2d.cpp
    ops/pooling.cpp
    ops/norm.cpp
    ops/elementwise.cpp
    ops/concat.cpp
    ops/upsample.cpp
    ops/softmax.cpp
    ops/topk.cpp                
    ops/gather_elements.cpp 

    optimizer/graph_optimizer.cpp
    optimizer/layout_transform.cpp
    optimizer/quantization.cpp

    loader/custom_loader.cpp

    api/engine_api.cpp

    # AVX2 GEMM alt parcalari (conv2d_gemm.cpp tarafindan include edilir)
    kernels/simd/avx2/conv2d_gemm_parts/01_math.inc
    kernels/simd/avx2/conv2d_gemm_parts/02_microkernels.inc
    kernels/simd/avx2/conv2d_gemm_parts/03_weight_cache.inc
    kernels/simd/avx2/conv2d_gemm_parts/04_im2col.inc
    kernels/simd/avx2/conv2d_gemm_parts/05_gemm.inc
    kernels/simd/avx2/conv2d_gemm_parts/06_output.inc
    kernels/simd/avx2/conv2d_gemm_parts/07_dispatch.inc
    kernels/simd/avx2/conv2d_gemm_parts/01b_fast_div_silu.inc
)

# MSVC / GCC / Clang — kaynak dosyalar UTF-8 kodlanmis
if(MSVC)
    add_compile_options(/utf-8)
else()
    add_compile_options(-finput-charset=UTF-8 -fexec-charset=UTF-8)
endif()

# ============================================================
#  Statik / Standart Engine Hedefi
# ============================================================
if(ENGINE_BUILD_SHARED)
    add_library(engine SHARED ${ENGINE_CORE_SOURCES})
    target_compile_definitions(engine PRIVATE ENGINE_BUILDING_SHARED)
    target_compile_definitions(engine PUBLIC  ENGINE_SHARED)
else()
    add_library(engine STATIC ${ENGINE_CORE_SOURCES})
    target_compile_definitions(engine PUBLIC ENGINE_STATIC)
endif()

add_library(engine::engine ALIAS engine)

# engine kutuphanesi AVX2 intrinsics ayarlari
if(ENGINE_HAS_AVX2)
    if(MSVC)
        target_compile_options(engine PRIVATE /arch:AVX2)
    else()
        target_compile_options(engine PRIVATE -mavx2 -mfma)
    endif()
endif()

target_compile_definitions(engine PRIVATE ENGINE_PROFILING=1)

target_include_directories(engine
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(engine PUBLIC engine_simd engine_deps)

set_target_properties(engine PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================
#  AVX2 / AVX-512 Cekirdek Nesne Kutuphaneleri (OBJECT)
# ============================================================
if(ENGINE_HAS_AVX2)
    add_library(engine_kernels_avx2 OBJECT
        kernels/simd/avx2/matmul_avx2.cpp
        kernels/simd/avx2/activation_avx2.cpp
        kernels/simd/avx2/conv2d_avx2.cpp          # Direct kernel
        kernels/simd/avx2/conv2d_winograd.cpp      # Winograd F(2,3) — AVX2
        kernels/simd/avx2/conv2d_gemm.cpp          # GEMM kernel
        kernels/simd/avx2/conv2d_pointwise_avx2.cpp
        kernels/simd/avx2/conv2d_direct_small_avx2.cpp
    )
    target_include_directories(engine_kernels_avx2 PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/generated
    )
    target_compile_options(engine_kernels_avx2 PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>
        $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
    )
    target_compile_definitions(engine_kernels_avx2 PRIVATE ENGINE_PROFILING=1)
    target_link_libraries(engine_kernels_avx2 PUBLIC engine_simd)
    target_sources(engine PRIVATE $<TARGET_OBJECTS:engine_kernels_avx2>)
endif()

if(ENGINE_HAS_AVX512)
    add_library(engine_kernels_avx512 OBJECT
        kernels/simd/avx512/matmul_avx512.cpp
    )
    target_include_directories(engine_kernels_avx512 PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/generated
    )
    target_compile_options(engine_kernels_avx512 PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma>
        $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX512>
    )
    target_link_libraries(engine_kernels_avx512 PUBLIC engine_simd)
    target_sources(engine PRIVATE $<TARGET_OBJECTS:engine_kernels_avx512>)
endif()

# ============================================================
#  engine_dll — Python ctypes / C API Paylasimli Kutuphane (.dll)
# ============================================================
add_library(engine_dll SHARED ${ENGINE_CORE_SOURCES})

target_compile_definitions(engine_dll PRIVATE 
    ENGINE_BUILDING_SHARED 
    ENGINE_SHARED
    ENGINE_EXPORTS 
    ENGINE_PROFILING=1
)

target_include_directories(engine_dll
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

if(ENGINE_HAS_AVX2)
    if(MSVC)
        target_compile_options(engine_dll PRIVATE /arch:AVX2)
    else()
        target_compile_options(engine_dll PRIVATE -mavx2 -mfma)
    endif()
    target_sources(engine_dll PRIVATE $<TARGET_OBJECTS:engine_kernels_avx2>)
endif()

if(ENGINE_HAS_AVX512)
    if(MSVC)
        target_compile_options(engine_dll PRIVATE /arch:AVX512)
    else()
        target_compile_options(engine_dll PRIVATE -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma)
    endif()
    target_sources(engine_dll PRIVATE $<TARGET_OBJECTS:engine_kernels_avx512>)
endif()

target_link_libraries(engine_dll PUBLIC engine_simd engine_deps)

set_target_properties(engine_dll PROPERTIES
    RUNTIME_OUTPUT_NAME "engine"
    ARCHIVE_OUTPUT_NAME "engine_dll"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================
#  Install
# ============================================================
include(GNUInstallDirs)
install(TARGETS engine engine_dll EXPORT engineTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/engine
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES ${CMAKE_SOURCE_DIR}/include/engine/engine_c_api.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/engine
)










# --- Profiling ---
if(ENGINE_PROFILING)
    if(TARGET engine)
        target_compile_definitions(engine PRIVATE ENGINE_PROFILING=1)
    endif()
    if(TARGET engine_dll)
        target_compile_definitions(engine_dll PRIVATE ENGINE_PROFILING=1)
    endif()
    message(STATUS "[profiling] ENGINE_PROFILING enabled")
endif()