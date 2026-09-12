# C++ 运行时基础示例

这是阶段 1 的基础工程，用命令行数字解析来展示 C++ 函数、`vector`、错误结果、所有权和自动化测试。它不是每一节的唯一示例；新的主题代码放在对应编号目录中。

## 文件职责

| 路径 | 用途 |
| --- | --- |
| `include/runtime_demo.hpp` | 声明解析与求和接口 |
| `src/runtime_demo.cpp` | 实现解析、错误返回和求和 |
| `src/main.cpp` | 处理命令行参数并显示结果 |
| `tests/runtime_test.cpp` | 检查核心函数的正常与错误输入 |
| `CMakeLists.txt` | 定义 `stage1_core`、`stage1_demo`、`stage1_test` |

本工程由父目录 CMake 统一构建。历史实验报告仍引用既有目标名；当前构建入口见[示例总览](../README.md)。
