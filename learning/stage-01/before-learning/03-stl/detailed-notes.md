# 泛型与特性

## 前置基础

1. 容器保存元素，迭代器表示一个位置，算法处理一段位置范围。
2. 模板让编译器根据实际类型生成代码。
3. 类型 traits 在编译期回答“这个类型具有什么性质”。

## STL

先掌握三类角色：

| 角色 | 例子 | 职责 |
| --- | --- | --- |
| 容器 | `vector`、`map` | 保存元素 |
| 迭代器 | `begin()`、`end()` | 表示范围 |
| 算法 | `sort`、`find_if` | 操作范围 |

放到 GPU 主线上，`vector<Tensor>` 可以保存一批输入，迭代器界定要处理的任务区间，算法可在 Host 侧筛选 dtype、按 shape 排序或组织 batch。STL 代码仍在 CPU 上运行；它不会因为元素描述 GPU Tensor 就自动变成 CUDA kernel。

半开区间写作 \([b,e)\)：包含起点，不包含终点。元素数量为：

\[
N
\overset{\text{迭代器作差}}{=}
e-b
\]

## 符号说明

- \(b\)：begin，范围起始迭代器。
- \(e\)：end，尾后迭代器。
- \(N\)：范围内的元素数量。
- \([b,e)\)：左闭右开范围。

容器扩容、删除或移动元素后，旧迭代器可能失效。是否失效取决于具体容器和操作，使用前要查该容器的规则。

allocator 把“申请原始存储”和“在存储中构造对象”分开。初学时使用默认 allocator；只有在内存池、对齐、共享内存或性能测量证明有需要时才自定义。

这与推理系统的显存池思想相通，但不是同一个分配器：标准 allocator 默认管理 Host 内存；CUDA caching allocator 管理 Device 内存。二者共同目标是减少频繁申请，但服务的地址空间和 API 不同。

## 模板特性

```cpp
template <class T>
auto twice(T value) {
  static_assert(std::is_arithmetic_v<T>);
  return value + value;
}
```

`T` 是模板参数；`is_arithmetic_v<T>` 是布尔类型特性。编译器在调用处代入实际类型，然后检查生成的代码是否合法。

CUDA 和 PyTorch 扩展大量使用模板：同一份 elementwise kernel 可针对 `float`、`half` 等类型实例化；traits 可以在编译期选择累加类型或判断某种 dtype 是否支持某条实现路径。

C++20 concept 把约束写进函数接口：

```cpp
template <std::integral T>
T twice_integer(T value) {
  return value + value;
}
```

concept 的主要收益是约束更清楚、报错更靠近调用原因。协程则允许函数暂停并稍后恢复；阶段 1 只需认识 `co_await`、`co_yield`、`co_return`，无需自己实现 promise type。

## 现代类型

| 类型 | 表达的含义 | 注意点 |
| --- | --- | --- |
| `optional<T>` | 可能有一个 `T`，也可能没有 | 读取前检查 |
| `variant<A,B>` | 当前恰好是候选类型之一 | 用 `visit` 统一处理 |
| `string_view` | 只观察一段字符，不拥有字符 | 原字符串必须活得更久 |
| 结构化绑定 | 把组合值拆成多个名字 | `auto` 与 `auto&` 含义不同 |

对应到推理 Runtime：`optional<Device>` 表示用户可能未指定设备；`variant<CpuBuffer, GpuBuffer>` 表示数据当前位于一种后端；`string_view` 可读取算子名而不复制；结构化绑定可拆开 shape 与 stride。

```cpp
std::optional<int> parse_count(std::string_view text);

std::variant<int, std::string> value = 7;
std::visit([](const auto& item) { std::cout << item; }, value);

for (const auto& [key, count] : table) {
  std::cout << key << count;
}
```

`string_view` 最常见的错误是悬空：被观察的 `std::string` 已销毁，view 还在使用。

## 程序实例

实例工程使用 `vector`、算法、`optional`、`variant`、`string_view`、结构化绑定和 type traits，见[程序实例](../../examples/README.md)。

## 部署说明

实例以 C++17 为基线。concept 示例需要 C++20，可单独把 `CMAKE_CXX_STANDARD` 改为 `20` 后实验；协程在不同编译器上的库支持有差异，本阶段不作为构建要求。

## 直观理解

一个 batch 像待送入 GPU 的货架：容器保存任务，迭代器圈出本批范围，算法在 Host 侧分组。模板让同一 kernel 适配多种 dtype，traits 和 concept 在编译期拦下不支持的类型。
