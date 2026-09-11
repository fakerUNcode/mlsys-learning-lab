# Stage 1 原始示例工程的唯一构建入口。
#
# 本文件与 reports/stage-01/02-build.md 一一对应：
# - stage1_core：保存解析与求和逻辑的静态库；
# - stage1_demo：读取命令行并展示结果；
# - stage1_test：直接调用核心库执行断言；
# - ENABLE_SANITIZERS：统一为三个目标开启 ASan/UBSan。
#
# 从仓库根目录运行：
#   cmake -S learning/stage-01/examples -B build/stage1-lifetime
#   cmake --build build/stage1-lifetime
#   ctest --test-dir build/stage1-lifetime --output-on-failure

# 要求 CMake 版本不低于 3.16。
# 这是 CMake 工具版本，不是 C++ 标准版本。
cmake_minimum_required(VERSION 3.16)

# 项目名是 stage1_runtime，并且本工程只启用 C++。
# 没有写 CUDA，因此这里不会调用 nvcc 或编译 .cu 文件。
project(stage1_runtime LANGUAGES CXX)

# 定义一个用户可配置开关，默认 OFF。
# 配置时传 -DENABLE_SANITIZERS=ON 才会启用 ASan 和 UBSan。
option(ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)

# 将 runtime_demo.cpp 编译成 stage1_core 库。
# 当前未设置 BUILD_SHARED_LIBS，实际构建会生成 libstage1_core.a 静态库。
add_library(stage1_core src/runtime_demo.cpp)

# 让 stage1_core 在 include/ 中查找 runtime_demo.hpp。
# PUBLIC 还会把该包含路径传播给链接 stage1_core 的 demo 和 test。
target_include_directories(stage1_core PUBLIC include)

# 要求核心库使用 C++17。
# PUBLIC 会把同一语言要求传播给依赖它的两个可执行目标。
target_compile_features(stage1_core PUBLIC cxx_std_17)

# 示例程序只负责命令行输入、输出和退出码。
add_executable(stage1_demo src/main.cpp)

# main.cpp 使用核心库声明的函数，因此必须链接 stage1_core。
# PRIVATE 表示该链接依赖不再由 stage1_demo 向外传播。
target_link_libraries(stage1_demo PRIVATE stage1_core)

# 测试程序拥有自己的 main()，与示例程序是两个不同进程。
add_executable(stage1_test tests/runtime_test.cpp)

# 测试直接调用 parse_numbers 和 sum_if_not_empty，
# 所以也需要链接同一份 stage1_core 实现。
target_link_libraries(stage1_test PRIVATE stage1_core)

# 只有用户打开开关，并且编译器是 GNU 或 Clang 时才添加这些参数。
# 这样不会把 GCC/Clang 专用选项强行传给其他工具链。
if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # 三个目标都要插桩：
  # - 库需要检查业务实现；
  # - demo/test 需要链接 Sanitizer 运行库并检查自己的代码。
  foreach(target stage1_core stage1_demo stage1_test)
    # address 检查 Host 内存错误；
    # undefined 检查部分 C++ 未定义行为；
    # frame pointer 帮助生成容易阅读的错误调用栈。
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)

    # 链接阶段也必须添加 Sanitizer 参数，否则可能缺少运行库符号。
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endforeach()
endif()

# 生成 CTest 所需的测试元数据。
enable_testing()

# 测试显示名和执行目标都叫 stage1_test。
# CTest 根据该程序的退出码判断 Passed 或 Failed。
add_test(NAME stage1_test COMMAND stage1_test)
