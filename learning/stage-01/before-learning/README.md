# 前置总览

## 前置基础

本阶段只依赖三个旧概念：

1. **变量**：有类型、有名字的一块可访问数据。
2. **作用域**：名字可以被使用的代码范围，常由一对花括号界定。
3. **栈与堆**：栈上局部对象通常随作用域自动结束；堆上对象需要明确的所有者负责释放。

## 核心关系

对象的完整生命过程是：

```text
取得存储
⇒（构造）成为对象
⇒（使用）保持有效
⇒（析构）结束生命
⇒（释放）归还存储
```

这里没有需要计算的数学公式。箭头右侧已经标明每次状态变化所依据的操作。

## 主线位置

后续编写 CUDA/PyTorch 扩展时，一次计算会同时涉及多种对象：Host 上的 Tensor 元数据、Device 上的显存、CUDA Stream、Event、动态库句柄和工作线程。阶段 1 要解决的是“这些资源由谁创建、传给谁、何时释放、失败后怎样收尾”。

```text
Python 发起算子
⇒（扩展边界）C++ 检查 Tensor
⇒（CUDA Runtime）申请资源并发射 kernel
⇒（异步执行）GPU 写回结果
⇒（同步或事件）Host 确认完成
```

## 符号说明

- `⇒`：状态转移。
- RAII：**Resource Acquisition Is Initialization**，资源获取即初始化。
- ABI：Application Binary Interface，应用二进制接口。

## 阅读地图

| 顺序 | 文档 | 解决的问题 |
| --- | --- | --- |
| 0 | [语法地基](00-cpp-basics/README.md) | 读懂类型、函数、引用、指针和现代语法 |
| 1 | [生命周期](01-lifetime/README.md) | RAII、构造、析构、拷贝和移动 |
| 2 | [智能指针](02-smart-pointers/README.md) | 独占、共享和观察所有权 |
| 3 | [STL基础](03-stl/README.md) | 容器、迭代器、算法和 allocator |
| 4 | [模板特性](04-templates/README.md) | 泛型与类型 traits |
| 5 | [C++17](05-cpp17/README.md) | optional、variant、view 和绑定 |
| 6 | [C++20](06-cpp20/README.md) | concept 与协程初识 |
| 7 | [错误处理](07-errors/README.md) | exception 与 error code |
| 8 | [ABI链接](08-abi/README.md) | 动态库、符号与链接 |
| 9 | [构建测试](09-build-test/README.md) | CMake、CTest 与 Sanitizer |

## 直观理解

把推理程序看成 GPU 工厂：C++ 对象是资源管理员，RAII 负责下班时归还显存和 Stream，类型系统检查原料规格，链接器把 Host 调度代码与 CUDA 算子组装起来。
