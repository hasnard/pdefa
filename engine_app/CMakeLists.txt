cmake_minimum_required(VERSION 3.20)
project(yolo_app LANGUAGES CXX)

# ---- C++ standardi (motor C++20 kullaniyor) ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Default build tipi ----
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ==================================================================
#  1) IMAGELOADER  (kardes klasor, add_subdirectory)
# ==================================================================
set(IMAGELOADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../imageloader")

if(NOT EXISTS "${IMAGELOADER_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "imageloader bulunamadi: ${IMAGELOADER_ROOT}")
endif()

add_subdirectory(
    ${IMAGELOADER_ROOT}
    ${CMAKE_BINARY_DIR}/imageloader_build
)

# ==================================================================
#  2) MOTOR  (hazir engine.lib linkle, add_subdirectory YAPMA)
# ==================================================================
set(ENGINE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../PDEFA1.0")
set(ENGINE_LIB  "/media/hasnard/Yeni Birim/PROJE/engineforaiincpu/PDEFA1.0/build_linux/lib/libengine.a")

if(NOT EXISTS "${ENGINE_LIB}")
    message(FATAL_ERROR
        "engine.lib bulunamadi: ${ENGINE_LIB}\n"
        "Once motoru derle:\n"
        "  cd ${ENGINE_ROOT}\n"
        "  cmake --build build --config Release -j 8")
endif()

# ==================================================================
#  3) UYGULAMA
# ==================================================================
add_executable(yolo_app
    main.cpp
    src/pipeline.cpp
)

target_include_directories(yolo_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ENGINE_ROOT}/include
    ${ENGINE_ROOT}/build/generated
)

target_link_libraries(yolo_app PRIVATE
    imageloader
    ${ENGINE_LIB}
)

# ---- Windows sistem kutuphaneleri ----
if(WIN32)
    target_link_libraries(yolo_app PRIVATE
        kernel32
        user32
        gdi32
        winspool
        shell32
        ole32
        oleaut32
        uuid
        comdlg32
        advapi32
    )
endif()

# ---- MSVC uyarilari ----
if(MSVC)
    target_compile_options(yolo_app PRIVATE /W4)
endif()

# ---- Cikti bilgisi ----
message(STATUS "")
message(STATUS "================ engine_app Configuration ================")
message(STATUS "  Target          : yolo_app")
message(STATUS "  imageloader     : ${IMAGELOADER_ROOT}")
message(STATUS "  engine root     : ${ENGINE_ROOT}")
message(STATUS "  engine.lib      : ${ENGINE_LIB}")
message(STATUS "=========================================================")
message(STATUS "")