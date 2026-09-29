# ============================================================
#  CompilerFlags.cmake
#  Tüm derleyiciler için optimize edilmiş flag seti
# ============================================================

include(CheckCXXCompilerFlag)

# INTERFACE target: tüm hedefler buna link eder
add_library(engine_flags INTERFACE)

# ---------- Common flags ----------
target_compile_features(engine_flags INTERFACE cxx_std_20)

if(MSVC)
    target_compile_options(engine_flags INTERFACE
        /W4 /permissive- /Zc:__cplusplus /Zc:preprocessor
        /MP                     # Multi-processor compile
        /EHsc
        $<$<CONFIG:Release>:/O2 /Ob3 /Oi /Ot /Oy /GL /fp:fast /DNDEBUG>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Zi>
        $<$<CONFIG:Debug>:/Od /RTC1 /Zi>
    )
    target_link_options(engine_flags INTERFACE
        $<$<CONFIG:Release>:/LTCG /OPT:REF /OPT:ICF>
    )
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    # GCC / Clang ortak
    target_compile_options(engine_flags INTERFACE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -fno-math-errno
        -fno-trapping-math
        -ffp-contract=fast
    )

    if(ENGINE_WARNINGS_AS_ERRORS)
        target_compile_options(engine_flags INTERFACE -Werror)
    endif()

    # Release optimizasyonları
    target_compile_options(engine_flags INTERFACE
        $<$<CONFIG:Release>:
            -O3
            -funroll-loops
            -fstrict-aliasing
            -fno-semantic-interposition
            -DNDEBUG
        >
        $<$<CONFIG:RelWithDebInfo>:-O2 -g>
        $<$<CONFIG:Debug>:-O0 -g3>
    )

    # LTO
    if(ENGINE_ENABLE_LTO)
        check_cxx_compiler_flag("-flto" HAS_LTO)
        if(HAS_LTO)
            target_compile_options(engine_flags INTERFACE
                $<$<CONFIG:Release>:-flto>
            )
            target_link_options(engine_flags INTERFACE
                $<$<CONFIG:Release>:-flto>
            )
        endif()
    endif()

    # march=native
    if(ENGINE_ENABLE_NATIVE OR ENGINE_ENABLE_NATIVE_ARCH)
        check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
        if(HAS_MARCH_NATIVE)
            target_compile_options(engine_flags INTERFACE
                $<$<CONFIG:Release>:-march=native>
            )
            message(STATUS "[CompilerFlags] -march=native enabled")
        endif()
    endif()

    # Clang ekstra
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(engine_flags INTERFACE
            $<$<CONFIG:Release>:-fvectorize -fslp-vectorize>
        )
    endif()
endif()

# Sanitizers (opsiyonel -DENGINE_ENABLE_ASAN=ON)
option(ENGINE_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENGINE_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

if(ENGINE_ENABLE_ASAN AND NOT MSVC)
    target_compile_options(engine_flags INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(engine_flags INTERFACE -fsanitize=address)
endif()

if(ENGINE_ENABLE_UBSAN AND NOT MSVC)
    target_compile_options(engine_flags INTERFACE -fsanitize=undefined -fno-omit-frame-pointer)
    target_link_options(engine_flags INTERFACE -fsanitize=undefined)
endif()