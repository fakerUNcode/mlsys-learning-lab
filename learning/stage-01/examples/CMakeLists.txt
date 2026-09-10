cmake_minimum_required(VERSION 3.16)
project(stage1_runtime LANGUAGES CXX)

option(ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)

add_library(stage1_core src/runtime_demo.cpp)
target_include_directories(stage1_core PUBLIC include)
target_compile_features(stage1_core PUBLIC cxx_std_17)

add_executable(stage1_demo src/main.cpp)
target_link_libraries(stage1_demo PRIVATE stage1_core)

add_executable(stage1_test tests/runtime_test.cpp)
target_link_libraries(stage1_test PRIVATE stage1_core)

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target stage1_core stage1_demo stage1_test)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endforeach()
endif()

enable_testing()
add_test(NAME stage1_test COMMAND stage1_test)
