# STL 容器、迭代器与算法示例

这个小程序使用一个 Host 侧推理任务队列，集中演示：

- `std::vector` 保存连续的任务数据；
- 迭代器表示 `[begin, end)` 半开区间；
- `std::sort` 排序任务；
- `std::find_if` 查找第一个满足条件的任务；
- `std::count_if` 统计满足条件的任务；
- `std::accumulate` 归约总 batch size。

示例刻意只包含一个源码和一个可执行程序，不包含测试目标。所有 STL 操作都在
CPU/Host 侧执行，任务结构体只是对推理调度元数据的简化模拟。

## 构建和运行

在仓库根目录执行：

```bash
cmake -S learning/stage-01/examples -B build/stage1-examples -DCMAKE_BUILD_TYPE=Debug
cmake --build build/stage1-examples --target stage1_stl_demo
./build/stage1-examples/03-stl-containers-and-algorithms/stage1_stl_demo
```

阅读时建议先预测排序后的顺序、查找结果、统计值和 batch 总和，再运行程序核对。
