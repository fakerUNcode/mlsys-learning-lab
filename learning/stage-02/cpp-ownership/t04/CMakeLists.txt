# T04 独立构建入口：编译 buffer.cpp（包含 buffer.hpp），再交给 CTest 运行。
# 从仓库根目录使用 cmake -S learning/stage-02/cpp-ownership/t04 -B <构建目录>。
cmake_minimum_required(VERSION 3.16)
project(t04_buffer_ownership LANGUAGES CXX)

option(ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" ON)

add_executable(t04_buffer buffer.cpp)
target_compile_features(t04_buffer PRIVATE cxx_std_17)
set_target_properties(t04_buffer PROPERTIES CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(t04_buffer PRIVATE -Wall -Wextra -Wpedantic)
endif()

# 编译阶段插入检查代码，链接阶段引入检查运行库；两处都需要开启对应选项。
if(ENABLE_SANITIZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "This exercise's sanitizer flags require GCC or Clang")
  endif()
  target_compile_options(t04_buffer PRIVATE -fsanitize=address,undefined
                         -fno-omit-frame-pointer)
  target_link_options(t04_buffer PRIVATE -fsanitize=address,undefined)
endif()

enable_testing()
add_test(NAME t04_buffer_ownership COMMAND t04_buffer)
# CTest 启用泄漏检查并在发现错误时立即失败，避免程序打印 PASS 却忽略检查报告。
if(ENABLE_SANITIZERS)
  set_tests_properties(t04_buffer_ownership PROPERTIES ENVIRONMENT
    "ASAN_OPTIONS=detect_leaks=1:halt_on_error=1;UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1")
endif()
