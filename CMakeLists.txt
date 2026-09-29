cmake_minimum_required(VERSION 3.20)
project(pdefa_native VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

# --- Yollar ---
set(IMAGELOADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../imageloader")
set(ENGINE_ROOT      "${CMAKE_CURRENT_SOURCE_DIR}/../PDEFA1.0")

if(WIN32)
    set(ENGINE_STATIC_LIB "${ENGINE_ROOT}/build/lib/Release/engine.lib")
else()
    set(ENGINE_STATIC_LIB "${ENGINE_ROOT}/build/lib/libengine.a")
endif()

if(NOT EXISTS "${ENGINE_STATIC_LIB}")
    message(FATAL_ERROR "engine.lib yok: ${ENGINE_STATIC_LIB}")
endif()

# --- imageloader'i subdirectory olarak al ---
set(IMAGELOADER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(IMAGELOADER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${IMAGELOADER_ROOT} ${CMAKE_BINARY_DIR}/imageloader_build)

if(MSVC)
    set_target_properties(imageloader PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# --- pdefa_native ---
# Windows: .def ile explicit export  |  diger platformlar: visibility
if(WIN32)
    add_library(pdefa_native SHARED
        src/pdefa_c_api.cpp
        pdefa.def
    )
else()
    add_library(pdefa_native SHARED
        src/pdefa_c_api.cpp
    )
endif()

target_compile_definitions(pdefa_native PRIVATE PDEFA_BUILD)

target_include_directories(pdefa_native
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${ENGINE_ROOT}/include
    PRIVATE ${ENGINE_ROOT}/build/generated
    PRIVATE ${IMAGELOADER_ROOT}/include
)

target_link_libraries(pdefa_native PRIVATE
    imageloader
    ${ENGINE_STATIC_LIB}
)

find_package(Threads REQUIRED)
target_link_libraries(pdefa_native PRIVATE Threads::Threads)

# --- Boyut optimizasyonu ---
if(MSVC)
    target_compile_options(pdefa_native PRIVATE /Gy /Gw /O2 /Oi /GS- /utf-8)
    target_link_options(pdefa_native PRIVATE /LTCG /OPT:REF /OPT:ICF /INCREMENTAL:NO)
else()
    target_compile_options(pdefa_native PRIVATE -Os -ffunction-sections -fdata-sections -fvisibility=hidden)
    target_link_options(pdefa_native PRIVATE -Wl,--gc-sections -Wl,--as-needed)
endif()

set_target_properties(pdefa_native PROPERTIES
    OUTPUT_NAME "pdefa_native"
    WINDOWS_EXPORT_ALL_SYMBOLS OFF              # <<< /GL ile uyumlu olmasi icin OFF
    RUNTIME_OUTPUT_DIRECTORY         ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY         ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY         ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib
)

if(WIN32)
    target_link_libraries(pdefa_native PRIVATE
        kernel32 user32 gdi32 shell32 ole32 oleaut32 uuid advapi32)
endif()

message(STATUS "pdefa_native hazir — kaynak: ${ENGINE_STATIC_LIB}")