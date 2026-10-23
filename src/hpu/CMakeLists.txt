add_library(strata_hpu_experts STATIC expert_plan.cpp tile_stream.cpp)
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
  set_tests_properties(hpu_expert_plan_test hpu_plan_cli_test hpu_real_header_plan_test hpu_tile_stream_test
                       PROPERTIES TIMEOUT 30)
endif()
