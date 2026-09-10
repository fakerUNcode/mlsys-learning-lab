# 阶段1报告

本目录保存阶段 1“C++ 必需基础”的实验报告。报告按章节拆分，避免把构建日志、源码解析和结论堆在同一个大文件中。

## 本次实验

主题：C++17 生命周期示例的构建、运行、测试与 Sanitizer 验证。

| 章节 | 内容 |
| --- | --- |
| [实验总览](01-overview.md) | 环境、命令、输出、结论与边界 |
| [构建解析](02-build.md) | CMake、编译、静态库、链接和 CTest |
| [源码解析](03-code.md) | 头文件、解析函数、入口和对象生命周期 |
| [测试检查](04-test.md) | 测试分支、ASan、UBSan、局限与排错 |
| [今日总结](05-daily-summary.md) | 今日证据、掌握边界与剩余计划 |

## 阅读顺序

第一次阅读建议按表格顺序进行。已经熟悉 CMake 的读者可以先读[源码解析](03-code.md)，再回看构建产物如何连接。

## 证据边界

本次程序只使用 CPU 和标准 C++17，没有调用 CUDA Runtime，也没有分配 Device Memory。报告中涉及 GPU buffer 的内容只用于说明知识迁移关系，不代表本实验已经验证 GPU 生命周期。

## 目录变更

报告记录的是重构前的 `stage1_demo/stage1_test`。同一份解析源码现已归入[现代类型示例](../../learning/stage-01/examples/03-modern-types/README.md)，目标改名为 `modern_types_demo/modern_types_test`。历史输出保持原样，新运行命令以示例 README 为准。

## 后续任务

完成现有[练习测评](../../learning/stage-01/exercises/README.md)后，再把纯 CPU 所有权模型扩展为可选 CUDA 环境下的 `GpuBuffer` RAII 包装器。
