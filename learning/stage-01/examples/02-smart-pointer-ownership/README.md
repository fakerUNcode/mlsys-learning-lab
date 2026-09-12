# 智能指针所有权示例

这个小工程把三种所有权关系放到同一条推理场景里：请求独占 workspace、多个请求共享模型、缓存用弱引用观察模型。

## 文件职责

| 路径 | 用途 |
| --- | --- |
| `include/smart_pointer_model.hpp` | 定义 workspace、模型和弱缓存查询接口 |
| `src/main.cpp` | 展示所有权转移、强引用计数和弱引用过期 |
| `tests/smart_pointer_test.cpp` | 验证移动、共享寿命和缓存过期行为 |
| `CMakeLists.txt` | 声明 demo、测试和 Sanitizer 配置 |

## 运行

在仓库根目录执行[Stage 1 示例构建说明](../README.md)。demo 的输出会显示 `unique_ptr` 移动后的空源状态、共享模型的强引用数，以及最后一个强引用释放后的弱引用状态。

## 观察问题

1. `std::move` 后哪个 `unique_ptr` 拥有 workspace？
2. 为什么缓存保存 `weak_ptr` 不会增加 `use_count()`？
3. 为什么读取缓存时要调用 `lock()`，而不能直接解引用？
4. 最后一个 `shared_ptr` 销毁后，模型何时析构？
