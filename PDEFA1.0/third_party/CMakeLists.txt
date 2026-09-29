# ============================================================
#  third_party/CMakeLists.txt
#  Header-only veya gömülü kütüphaneler
# ============================================================

# xxHash (header-only, git submodule)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/xxhash")
    add_library(xxhash INTERFACE)
    target_include_directories(xxhash INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/xxhash
    )
    add_library(xxhash::xxhash ALIAS xxhash)
    message(STATUS "[third_party] xxhash found")
endif()

# cpu_features (isteğe bağlı — CPUID için)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpu_features")
    add_subdirectory(cpu_features)
endif()