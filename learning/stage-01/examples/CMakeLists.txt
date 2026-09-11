# Stage 1 的构建说明。
#
# 本目录最终生成两个可运行程序：
#
#   stage1_demo  读取命令行输入并打印结果，供学习者手动运行；
#   stage1_test  自动检查核心函数，供 CTest 运行。
#
# 它们不各自复制解析逻辑，而是共同链接 stage1_core：
#
#   runtime_demo.cpp -> stage1_core -> stage1_demo / stage1_test
#
# 这种拆分让“可复用的功能”“命令行界面”“测试”各自只承担一种任务。

# CMake 负责生成编译和链接规则；真正编译 C++ 的仍是系统编译器。
# 本示例要求 CMake 不低于 3.16。
cmake_minimum_required(VERSION 3.16)

# CXX 表示本项目使用 C++。这里尚未启用 CUDA，因此不会处理 .cu 文件。
project(stage1_runtime LANGUAGES CXX)

# 配置时可通过 -DENABLE_SANITIZERS=ON 打开运行时检查。默认关闭，避免每次
# 普通构建都引入额外开销；学习排查内存问题时再单独打开。
option(ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)

# 核心库只放可以被多个程序复用的代码。静态库本身没有 main 函数，不能直接运行。
add_library(stage1_core src/runtime_demo.cpp)

# runtime_demo.hpp 位于 include/。PUBLIC 的含义是：核心库编译时需要这个目录，
# 链接核心库的 demo 和 test 也会自动获得同一个头文件搜索路径。
target_include_directories(stage1_core PUBLIC include)

# optional、variant、string_view 和结构化绑定均需要 C++17。PUBLIC 还会把这一
# 编译要求传递给依赖 stage1_core 的两个程序，避免它们使用了不同标准。
target_compile_features(stage1_core PUBLIC cxx_std_17)

# demo 的 main 只组织输入、输出和错误状态；真正的 parse_numbers 实现在核心库中，
# 所以必须链接 stage1_core 才能在链接阶段找到函数定义。
add_executable(stage1_demo src/main.cpp)
target_link_libraries(stage1_demo PRIVATE stage1_core)

# 测试是另一个独立程序，拥有自己的 main。它同样链接核心库，确保测试的是 demo
# 实际使用的同一份实现，而不是复制出的测试版本。
add_executable(stage1_test tests/runtime_test.cpp)
target_link_libraries(stage1_test PRIVATE stage1_core)

# AddressSanitizer 检查常见的主机内存错误，UndefinedBehaviorSanitizer 检查部分
# C++ 未定义行为。Sanitizer 的插桩发生在编译时，也需要在链接时加入运行库。
# 这里只为 GNU 和 Clang 添加这些参数，以避免把不兼容参数传给其他编译器。
if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target stage1_core stage1_demo stage1_test)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endforeach()
endif()

# enable_testing 打开 CTest 集成；下面把名为 stage1_test 的可执行程序登记为一项
# 测试。它返回 0 时 CTest 显示 Passed，非 0 时显示 Failed。
enable_testing()
add_test(NAME stage1_test COMMAND stage1_test)
