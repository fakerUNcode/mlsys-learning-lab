cmake_minimum_required(VERSION 3.16)  # 要求使用的 CMake 版本不能低于 3.16

project(stage1_runtime LANGUAGES CXX)  # 定义项目名为 stage1_runtime，并声明这个项目使用 C++

option(ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)  # 定义一个名为 ENABLE_SANITIZERS 的开关，用来控制是否启用 ASan 和 UBSan，默认关闭

add_library(stage1_core src/runtime_demo.cpp)  # 把 src/runtime_demo.cpp 编译成一个名为 stage1_core 的库

target_include_directories(stage1_core PUBLIC include)  # 告诉 stage1_core 以及使用它的目标，都可以到 include 目录里查找头文件

target_compile_features(stage1_core PUBLIC cxx_std_17)  # 要求 stage1_core 以及依赖它的目标至少使用 C++17 标准

add_executable(stage1_demo src/main.cpp)  # 把 src/main.cpp 编译成一个名为 stage1_demo 的可执行程序

target_link_libraries(stage1_demo PRIVATE stage1_core)  # 让 stage1_demo 链接 stage1_core，从而可以使用核心库中的函数

add_executable(stage1_test tests/runtime_test.cpp)  # 把 tests/runtime_test.cpp 编译成一个名为 stage1_test 的测试程序

target_link_libraries(stage1_test PRIVATE stage1_core)  # 让 stage1_test 链接 stage1_core，从而测试核心库中的函数

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")  # 如果开启了 Sanitizer，并且当前 C++ 编译器是 GNU 或 Clang，就进入下面的配置

  foreach(target stage1_core stage1_demo stage1_test)  # 依次对 stage1_core、stage1_demo 和 stage1_test 三个目标执行下面的设置

    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)  # 编译时给当前目标开启 ASan、UBSan，并保留栈帧信息方便定位错误

    target_link_options(${target} PRIVATE -fsanitize=address,undefined)  # 链接时也加入 ASan 和 UBSan 所需的运行库参数

  endforeach()  # 结束 foreach 循环

endif()  # 结束 Sanitizer 条件判断

enable_testing()  # 启用 CMake 的测试功能，也就是允许后面使用 CTest 管理测试

add_test(NAME stage1_test COMMAND stage1_test)  # 注册一个名为 stage1_test 的测试，执行测试时实际运行 stage1_test 这个程序