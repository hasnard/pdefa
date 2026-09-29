# ============================================================
#  benchmarks/CMakeLists.txt
# ============================================================

find_package(benchmark REQUIRED)

add_executable(bench_matmul       bench_matmul.cpp)
add_executable(bench_conv2d       bench_conv2d.cpp)
add_executable(bench_end_to_end   bench_end_to_end.cpp)

foreach(tgt bench_matmul bench_conv2d bench_end_to_end)
    target_link_libraries(${tgt} PRIVATE engine::engine benchmark::benchmark)
    target_include_directories(${tgt} PRIVATE ${CMAKE_SOURCE_DIR}/src)
endforeach()

# Kısayol: 'make run_benchmarks'
add_custom_target(run_benchmarks
    COMMAND $<TARGET_FILE:bench_matmul>     --benchmark_format=json --benchmark_out=results/matmul.json
    COMMAND $<TARGET_FILE:bench_conv2d>     --benchmark_format=json --benchmark_out=results/conv2d.json
    COMMAND $<TARGET_FILE:bench_end_to_end> --benchmark_format=json --benchmark_out=results/e2e.json
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS bench_matmul bench_conv2d bench_end_to_end
    COMMENT "Running all benchmarks -> results/*.json"
)