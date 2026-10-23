add_library(strata_hpu_experts STATIC expert_plan.cpp tile_stream.cpp packed_decode.cpp)
target_include_directories(strata_hpu_experts SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party/ggml)
# The reference codec is deterministic FP32 arithmetic before BF16 RNE, including Q2_K's
# subtractive min term. A host compiler must not contract that term into a different oracle.
if(MSVC)
  set_source_files_properties(packed_decode.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
else()
  set_source_files_properties(packed_decode.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
find_package(Threads REQUIRED)
target_link_libraries(strata_hpu_experts PUBLIC strata_artifact)
target_link_libraries(strata_hpu_experts PUBLIC Threads::Threads)
add_executable(strata-hpu-plan plan_main.cpp)
target_link_libraries(strata-hpu-plan PRIVATE strata_hpu_experts)

if(STRATA_BUILD_TESTS)
  enable_testing()
  add_executable(hpu_expert_plan_test ../../tests/hpu/expert_plan_test.cpp)
  target_link_libraries(hpu_expert_plan_test PRIVATE strata_hpu_experts)
  add_test(NAME hpu_expert_plan_test COMMAND hpu_expert_plan_test
           ${PROJECT_SOURCE_DIR}/data/glm5-synth.gguf)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME hpu_plan_cli_test COMMAND ${Python3_EXECUTABLE}
           ${PROJECT_SOURCE_DIR}/tests/hpu/plan_cli_test.py $<TARGET_FILE:strata-hpu-plan>)
  add_test(NAME hpu_real_header_plan_test COMMAND ${Python3_EXECUTABLE}
           ${PROJECT_SOURCE_DIR}/tests/hpu/real_header_plan_test.py $<TARGET_FILE:strata-hpu-plan>)
  add_executable(hpu_tile_stream_test ../../tests/hpu/tile_stream_test.cpp)
  target_link_libraries(hpu_tile_stream_test PRIVATE strata_hpu_experts)
  add_test(NAME hpu_tile_stream_test COMMAND hpu_tile_stream_test
           ${PROJECT_SOURCE_DIR}/data/glm5-synth.gguf)
  add_executable(hpu_packed_decode_test ../../tests/hpu/packed_decode_test.cpp)
  target_link_libraries(hpu_packed_decode_test PRIVATE strata_hpu_experts)
  add_test(NAME hpu_packed_decode_test COMMAND hpu_packed_decode_test
           ${PROJECT_SOURCE_DIR}/tests/hpu/fixtures/packed_decode_vectors.bin)
  set_tests_properties(hpu_expert_plan_test hpu_plan_cli_test hpu_real_header_plan_test hpu_tile_stream_test hpu_packed_decode_test
                       PROPERTIES TIMEOUT 30)
endif()
