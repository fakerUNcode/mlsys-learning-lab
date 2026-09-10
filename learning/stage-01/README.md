# 阶段1

主题：面向 CUDA Host 侧开发的 C++ 必需知识，建议用时 3～4 周。

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

1. [前置总览](before-learning/README.md)
2. [生命所有权](before-learning/01-lifetime.md)
3. [泛型与特性](before-learning/02-generic.md)
4. [运行与构建](before-learning/03-runtime.md)
5. [程序实例](examples/README.md)
6. [练习测评](exercises/README.md)

练习页不含答案。提交你的答案后，再逐步批改与评分。
