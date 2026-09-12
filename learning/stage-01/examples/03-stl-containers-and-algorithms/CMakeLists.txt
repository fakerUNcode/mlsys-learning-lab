# 把 src/main.cpp 编译为一个独立的 STL 学习程序。
add_executable(stage1_stl_demo src/main.cpp)

# 本示例使用结构化绑定等 C++17 语法。
target_compile_features(stage1_stl_demo PRIVATE cxx_std_17)

# 可选的运行时检查与同阶段其他示例保持一致；本目录不注册测试程序。
if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(stage1_stl_demo PRIVATE
                         -fsanitize=address,undefined
                         -fno-omit-frame-pointer)
  target_link_options(stage1_stl_demo PRIVATE -fsanitize=address,undefined)
endif()
