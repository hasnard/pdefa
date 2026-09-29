# ============================================================
#  tests/CMakeLists.txt
# ============================================================

find_package(GTest REQUIRED)

# ---------- Unit testler ----------
add_executable(engine_unit_tests
    unit/test_tensor.cpp
    unit/test_allocator.cpp
    unit/test_conv2d.cpp
    unit/test_matmul.cpp
)

target_link_libraries(engine_unit_tests
    PRIVATE
        engine::engine
        GTest::gtest
        GTest::gtest_main
)

target_include_directories(engine_unit_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

# ---------- Integration testleri ----------
add_executable(engine_integration_tests
    integration/test_model_pipeline.cpp
)

target_link_libraries(engine_integration_tests
    PRIVATE
        engine::engine
        GTest::gtest
        GTest::gtest_main
)

# ---------- CTest kayıtları ----------
include(GoogleTest)
gtest_discover_tests(engine_unit_tests)
gtest_discover_tests(engine_integration_tests)