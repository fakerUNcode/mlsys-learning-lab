# 智能指针

本节把前一节的对象生命周期与移动语义用于三种常见所有权关系。判断指针类型时，先问“谁负责让对象活着并最终销毁它”，再选表达这种责任的类型。

## 前置基础

1. 所有权表示谁保证对象存活并最终销毁它。
2. 裸指针只保存地址，不自动表达所有权。
3. 引用计数记录共享所有者数量。

## 使用边界

| 类型 | GPU/推理场景 | 边界 |
| --- | --- | --- |
| `unique_ptr` | 请求独占 workspace | 默认首选，可移动不可复制 |
| `shared_ptr` | 多请求共享模型 | 只在确需共享寿命时使用 |
| `weak_ptr` | 模型缓存与观察者 | 使用前必须 `lock()` |

## 三种所有权

### `unique_ptr`：独占所有权

同一时刻只有一个 `unique_ptr` 拥有对象。它不能复制；可以用 `std::move` 把所有权交给另一个 `unique_ptr`。移动后源指针为空，但仍可安全销毁或重新赋值。

```cpp
auto workspace = std::make_unique<Workspace>(8);
auto active_workspace = std::move(workspace);
// workspace 为空；active_workspace 独占 workspace 对象。
```

这适合请求私有的临时 workspace、独占句柄和具有单一生命周期的 GPU 资源包装器。

### `shared_ptr`：共享所有权

多个 `shared_ptr` 可以共同拥有同一对象。复制一个强引用会增加所有者数；销毁或 `reset()` 一个强引用会减少数量。最后一个强引用消失时，对象才析构。

```cpp
auto model = std::make_shared<Model>("demo-model");
auto request = model;  // 两者共同持有同一个 Model。
```

只在多个请求确实需要共同延长模型寿命时使用。不要仅为“方便传递”而使用共享所有权；互相持有的 `shared_ptr` 还可能形成循环，使对象永远不析构。

### `weak_ptr`：观察，不拥有

`weak_ptr` 从 `shared_ptr` 建立，但不会增加强引用数，也不会让对象多活一会儿。要使用对象时必须调用 `lock()`：对象还活着就获得一个新的 `shared_ptr`，对象已销毁就得到空指针。

```cpp
std::weak_ptr<Model> cache_entry = model;
if (auto cached_model = cache_entry.lock()) {
  // cached_model 在当前作用域内保证对象存活。
} else {
  // 缓存条目已经过期，可以重新加载模型。
}
```

检查并解引用应基于 `lock()` 返回的强引用；单独检查 `expired()` 后再假定对象仍存在，可能遇到并发释放。

## 运行时观察

运行[智能指针示例](../../examples/02-smart-pointer-ownership/README.md)，然后查看 [`main.cpp`](../../examples/02-smart-pointer-ownership/src/main.cpp) 和 [`smart_pointer_test.cpp`](../../examples/02-smart-pointer-ownership/tests/smart_pointer_test.cpp)。示例中的模型用 Host `vector` 模拟权重，重点是所有权语义；它没有申请真实 GPU 显存。

先预测测试中的 `use_count()`、`expired()` 和移动后源指针状态，再运行测试核对预测。不要把强引用计数当作多线程同步工具。

对象销毁条件：

$$
n_s
\overset{\text{减至零}}{=}
0
\Rightarrow_{\text{析构对象}}
\text{释放资源}
$$


## 符号说明

- \(n_s\)：强引用数量。
- 下标 \(s\)：strong，强引用。

## 程序实例

本节的可运行示例位于[智能指针专题目录](../../examples/02-smart-pointer-ownership/README.md)：它展示 `unique_ptr` 移交 workspace、多个请求共享模型，以及 `weak_ptr::lock()` 对存活和过期模型的检查。

## 部署说明

从仓库根目录构建并运行：

```bash
cmake -S learning/stage-01/examples -B build/stage1-examples -DCMAKE_BUILD_TYPE=Debug
cmake --build build/stage1-examples
ctest --test-dir build/stage1-examples --output-on-failure
./build/stage1-examples/02-smart-pointer-ownership/stage1_smart_pointer_demo
```

重点预测每次 `reset()` 后的强引用数，并确认弱引用不会阻止模型销毁。

## 直观理解

`unique_ptr` 是唯一门卡，`shared_ptr` 是多人共同续租，`weak_ptr` 是不续租的通讯录。模型缓存应使用通讯录，否则缓存本身可能让过期权重一直占用显存。
