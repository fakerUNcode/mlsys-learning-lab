cmake_minimum_required(VERSION 3.16)
project(stage1_examples LANGUAGES CXX)

option(ENABLE_SANITIZERS "Enable ASan and UBSan" OFF)

add_library(stage1_options INTERFACE)
target_compile_features(stage1_options INTERFACE cxx_std_17)

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(stage1_options INTERFACE
    -fsanitize=address,undefined
    -fno-omit-frame-pointer
  )
  target_link_options(stage1_options INTERFACE
    -fsanitize=address,undefined
  )
endif()

enable_testing()
add_subdirectory(01-lifetime)
add_subdirectory(02-smart-pointers)
add_subdirectory(03-modern-types)
