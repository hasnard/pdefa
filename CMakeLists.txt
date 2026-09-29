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

# =====================================================================
#  Layout:
#   D:/PROJE/pdefa/CMakeLists.txt        <- this file
#   D:/PROJE/pdefa/pdefa/src/*.cpp       <- native sources
#   D:/PROJE/pdefa/pdefa/include/*.h     <- native headers
#   D:/PROJE/pdefa/PDEFA1.0/             <- engine
#   D:/PROJE/pdefa/imageloader/          <- imageloader
# =====================================================================
set(ROOT            "${CMAKE_CURRENT_SOURCE_DIR}")
set(NATIVE_ROOT     "${ROOT}/pdefa")
set(ENGINE_ROOT     "${ROOT}/PDEFA1.0")
set(IMAGELOADER_ROOT "${ROOT}/imageloader")

# --- Engine artifacts ---
if(WIN32)
    set(ENGINE_STATIC_LIB "${ENGINE_ROOT}/build/lib/Release/engine.lib"
        CACHE FILEPATH "Path to engine.lib")
    set(ENGINE_DLL        "${ENGINE_ROOT}/build/bin/Release/engine.dll"
        CACHE FILEPATH "Path to engine.dll")
else()
    set(ENGINE_STATIC_LIB "${ENGINE_ROOT}/build/lib/libengine.a"
        CACHE FILEPATH "Path to libengine.a")
    set(ENGINE_DLL        "${ENGINE_ROOT}/build/lib/libengine.so"
        CACHE FILEPATH "Path to libengine.so")
endif()

if(NOT EXISTS "${ENGINE_STATIC_LIB}")
    message(FATAL_ERROR "engine.lib not found: ${ENGINE_STATIC_LIB}")
endif()

# --- imageloader ---
if(NOT EXISTS "${IMAGELOADER_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "imageloader not found: ${IMAGELOADER_ROOT}")
endif()

set(IMAGELOADER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(IMAGELOADER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${IMAGELOADER_ROOT}" "${CMAKE_BINARY_DIR}/imageloader_build")

if(MSVC AND TARGET imageloader)
    set_target_properties(imageloader PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# --- pdefa_native ---
set(PDEFA_SRC "${NATIVE_ROOT}/src/pdefa_c_api.cpp")
set(PDEFA_DEF "${NATIVE_ROOT}/pdefa.def")
set(PDEFA_INC "${NATIVE_ROOT}/include")

if(NOT EXISTS "${PDEFA_SRC}")
    message(FATAL_ERROR "Source not found: ${PDEFA_SRC}")
endif()
if(NOT EXISTS "${PDEFA_INC}/pdefa_c_api.h")
    message(FATAL_ERROR "Header not found: ${PDEFA_INC}/pdefa_c_api.h")
endif()

if(WIN32 AND EXISTS "${PDEFA_DEF}")
    add_library(pdefa_native SHARED "${PDEFA_SRC}" "${PDEFA_DEF}")
else()
    add_library(pdefa_native SHARED "${PDEFA_SRC}")
endif()

target_compile_definitions(pdefa_native PRIVATE PDEFA_BUILD)

target_include_directories(pdefa_native
    PUBLIC  "${PDEFA_INC}"
    PRIVATE "${NATIVE_ROOT}/src"           # headers next to the .cpp
    PRIVATE "${ENGINE_ROOT}/include"
    PRIVATE "${ENGINE_ROOT}/build/generated"
    PRIVATE "${IMAGELOADER_ROOT}/include"
)

target_link_libraries(pdefa_native PRIVATE
    imageloader
    "${ENGINE_STATIC_LIB}"
)

find_package(Threads REQUIRED)
target_link_libraries(pdefa_native PRIVATE Threads::Threads)

# --- Size / optimization ---
if(MSVC)
    target_compile_options(pdefa_native PRIVATE /Gy /Gw /O2 /Oi /GS- /utf-8)
    target_link_options(pdefa_native PRIVATE /LTCG /OPT:REF /OPT:ICF /INCREMENTAL:NO)
else()
    target_compile_options(pdefa_native PRIVATE
        -Os -ffunction-sections -fdata-sections -fvisibility=hidden)
    target_link_options(pdefa_native PRIVATE
        -Wl,--gc-sections -Wl,--as-needed)
endif()

set_target_properties(pdefa_native PROPERTIES
    OUTPUT_NAME "pdefa_native"
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
    RUNTIME_OUTPUT_DIRECTORY         "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY         "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY         "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib"
)

if(WIN32)
    target_link_libraries(pdefa_native PRIVATE
        kernel32 user32 gdi32 shell32 ole32 oleaut32 uuid advapi32)

    if(EXISTS "${ENGINE_DLL}")
        add_custom_command(TARGET pdefa_native POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ENGINE_DLL}"
                "$<TARGET_FILE_DIR:pdefa_native>"
            COMMENT "Copying engine.dll next to pdefa_native.dll")
    endif()
endif()

message(STATUS "")
message(STATUS "=========== pdefa_native ===========")
message(STATUS "  NATIVE_ROOT       = ${NATIVE_ROOT}")
message(STATUS "  PDEFA_SRC         = ${PDEFA_SRC}")
message(STATUS "  PDEFA_INC         = ${PDEFA_INC}")
message(STATUS "  ENGINE_ROOT       = ${ENGINE_ROOT}")
message(STATUS "  ENGINE_STATIC_LIB = ${ENGINE_STATIC_LIB}")
message(STATUS "  IMAGELOADER_ROOT  = ${IMAGELOADER_ROOT}")
message(STATUS "====================================")
message(STATUS "")