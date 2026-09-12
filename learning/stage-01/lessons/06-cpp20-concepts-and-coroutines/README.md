# C++20

## 前置基础

1. 模板错误常在实例化时暴露。
2. concept 为模板参数声明约束。
3. 协程允许函数暂停后恢复。

## 学习边界

```cpp
template <std::integral T>
T twice(T value) {
  return value + value;
}
```

concept 能让“不支持该类型”的接口更清楚。协程先认识 `co_await`、`co_yield` 和 `co_return`，暂不实现 promise type 或异步 Runtime。

实习前重点是能阅读使用 concept 的 CUDA/C++ 库代码。协程、复杂 executor 和网络异步框架暂缓。

## 程序实例

当前示例统一使用 C++17，因此本节没有独立工程。需要实验时再创建 `06-cpp20-concepts-and-coroutines` 示例，避免空项目占位。

## 直观理解

concept 像 kernel 的编译期入场条件；协程像 Host 调度任务保存现场后暂时让出线程。阶段 1 先能看懂标识和控制流，不建设完整调度系统。
