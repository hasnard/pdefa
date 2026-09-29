cmake_minimum_required(VERSION 3.16)
project(imageloader LANGUAGES CXX)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------- Kutuphane ----------------
add_library(imageloader
    src/loader.cpp
    src/preprocess.cpp
    src/png/png_decoder.cpp
    src/png/png_chunks.cpp
    src/png/png_inflate.cpp
    src/png/png_filters.cpp
    src/png/png_unpack.cpp
    src/jpeg/jpeg_decoder.cpp
    src/jpeg/jpeg_markers.cpp
    src/jpeg/jpeg_huffman.cpp
    src/jpeg/jpeg_idct.cpp
    src/jpeg/jpeg_color.cpp
    src/engine_api.cpp
    
)

target_include_directories(imageloader
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/../PDEFA1.0/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/png
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/jpeg
)

# AVX2 ve FMA vektörizasyon bayrakları
if (MSVC)
    target_compile_options(imageloader PRIVATE /W4 /arch:AVX2)
else()
    target_compile_options(imageloader PRIVATE -Wall -Wextra -mavx2 -mfma)
endif()

option(IMAGELOADER_BUILD_TESTS    "tests"    ON)
option(IMAGELOADER_BUILD_EXAMPLES "examples" ON)

# ---------------- Testler ----------------
if(IMAGELOADER_BUILD_TESTS)
enable_testing()

add_executable(test_png tests/test_png.cpp)
target_link_libraries(test_png PRIVATE imageloader)
add_test(NAME test_png COMMAND test_png)
set_tests_properties(test_png PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)

add_executable(test_preprocess tests/test_preprocess.cpp)
target_link_libraries(test_preprocess PRIVATE imageloader)
add_test(NAME test_preprocess COMMAND test_preprocess)
set_tests_properties(test_preprocess PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)

add_executable(test_jpeg tests/test_jpeg.cpp)
target_link_libraries(test_jpeg PRIVATE imageloader)
add_test(NAME test_jpeg COMMAND test_jpeg)
set_tests_properties(test_jpeg PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

# ---------------- Ornekler ----------------
if(IMAGELOADER_BUILD_EXAMPLES)
add_executable(dump_png examples/dump_png.cpp)
target_link_libraries(dump_png PRIVATE imageloader)

add_executable(dump_jpeg examples/dump_jpeg.cpp)
target_link_libraries(dump_jpeg PRIVATE imageloader)

add_executable(png_info examples/png_info.cpp)
target_link_libraries(png_info PRIVATE imageloader)

add_executable(png_to_ppm examples/png_to_ppm.cpp)
target_link_libraries(png_to_ppm PRIVATE imageloader)

add_executable(preprocess_demo examples/preprocess_demo.cpp)
target_link_libraries(preprocess_demo PRIVATE imageloader)

add_executable(engine_demo examples/engine_demo.cpp)
target_link_libraries(engine_demo PRIVATE imageloader)
endif()