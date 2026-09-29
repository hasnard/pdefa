# ============================================================
#  examples/CMakeLists.txt
# ============================================================

add_executable(smoke_test smoke_test.cpp)
target_link_libraries(smoke_test PRIVATE engine::engine)

add_executable(benchmark_matmul benchmark_matmul.cpp)
target_link_libraries(benchmark_matmul PRIVATE engine::engine)

add_executable(test_conv2d test_conv2d.cpp)
target_link_libraries(test_conv2d PRIVATE engine::engine)

add_executable(benchmark_conv2d benchmark_conv2d.cpp)
target_link_libraries(benchmark_conv2d PRIVATE engine::engine)


add_executable(test_engine_load test_engine_load.cpp)
target_link_libraries(test_engine_load PRIVATE engine::engine)

add_executable(benchmark_detection benchmark_detection.cpp)
target_link_libraries(benchmark_detection PRIVATE engine::engine)

add_executable(test_bn_load test_bn_load.cpp)
target_link_libraries(test_bn_load PRIVATE engine::engine)

add_executable(test_memory test_memory.cpp)
target_link_libraries(test_memory PRIVATE engine::engine)

add_executable(benchmark_big_model benchmark_big_model.cpp)
target_link_libraries(benchmark_big_model PRIVATE engine::engine)

add_executable(test_silu test_silu.cpp)
target_link_libraries(test_silu PRIVATE engine::engine)

add_executable(benchmark_yolo benchmark_yolo.cpp)
target_link_libraries(benchmark_yolo PRIVATE engine::engine)

