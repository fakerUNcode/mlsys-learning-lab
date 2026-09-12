# 阶段1

主题：面向 CUDA Host 侧开发的 C++ 必需知识，建议用时 3～4 周。

当前进度：第 `00`～`03` 节已完成；下一步学习第 `04` 节“模板与类型特性”。总进度与每日证据见 [学习进度](../../daily-record/README.md)。

## 主线位置

本阶段不追求完整掌握现代 C++，只获得继续学习 GPU 并行、CUDA kernel 和 PyTorch 扩展所需的最低闭环：

```text
C++ 管理资源与并发
⇒（编写底层实现）CUDA kernel
⇒（接入框架）PyTorch C++/CUDA 扩展
⇒（组织任务）推理 Runtime
⇒（定位问题）Profiling 与性能优化
```

因此，每个知识点都要追问三个问题：它管理了哪种 CPU/GPU 资源？它位于 Host、Device 还是二者边界？它怎样影响正确性、性能或可部署性？

## 学习目标

- 能解释对象何时创建、销毁、拷贝和移动，并管理显存、Stream、Event 等 GPU 资源。
- 能根据所有权选择智能指针。
- 会用 STL、模板和 C++17 类型表达 Tensor 元数据、算子配置与执行结果。
- 初步认识 concept 与协程。
- 会在 C++ 内部、CUDA API 和 Python 扩展边界选择异常或错误码。
- 能解释 `.so` 如何让 Python/PyTorch 找到 C++/CUDA 算子。
- 会用 CMake 构建，用测试和 Sanitizer 找问题。

## 学习边界

allocator、concept、协程和复杂 ABI 只要求能读懂、能查询。通用线程池、RPC、网络服务治理和模板元编程技巧暂缓；达到本页目标后直接进入 GPU 并行基础。

## 学习顺序

1. [Stage 1 课程地图](lessons/README.md)
2. [语法地基](lessons/00-cpp-language-basics/README.md)
3. [生命周期](lessons/01-object-lifetime-and-move/README.md)
4. [智能指针](lessons/02-smart-pointer-ownership/README.md)
5. [STL基础](lessons/03-stl-containers-and-algorithms/README.md)
6. [模板特性](lessons/04-templates-and-type-traits/README.md)
7. [C++17](lessons/05-cpp17-types-and-features/README.md)
8. [C++20](lessons/06-cpp20-concepts-and-coroutines/README.md)
9. [错误处理](lessons/07-error-handling/README.md)
10. [ABI链接](lessons/08-abi-and-dynamic-linking/README.md)
11. [构建测试](lessons/09-build-test-sanitizers/README.md)
12. [按主题组织的示例程序](examples/README.md)
13. [练习测评](exercises/README.md)

练习页不含答案。提交你的答案后，再逐步批改与评分。
