# 今日总结

日期：2026-09-10。

## 今日目标

运行阶段 1 的首个 C++ 工程，理解 CMake 构建、CTest、正常与错误路径、退出码及 Sanitizer，并把实验过程沉淀为可复现报告。

## 完成内容

- 使用 GNU C++ 13.3.0 完成 Debug 构建；
- 生成静态库、示例程序和测试程序；
- CTest 的 `1/1` 测试通过；
- 正常输入 `1,2,3,4` 得到数量 `4`、和 `10`；
- 非法输入 `1,x,3` 得到错误文本和退出码 `1`；
- ASan 与 UBSan 构建及测试通过；
- 输出四章、共千行以上的实验解析报告；
- 保存 CUDA Vector Add 实测结果；
- 将阶段 1 讲义按九个主题拆分；
- 将示例拆成生命周期、智能指针、现代类型三个独立工程；
- 使用新顶层 CMake 构建三项测试，结果 `3/3` 通过。

## 今日证据

新目录的验证结果：

```text
lifetime_test      Passed
smart_pointer_test Passed
modern_types_test  Passed

100% tests passed
0 tests failed out of 3
```

生命周期示例输出：

```text
acquire 1024 elements
move ownership
source_size=0
target_size=1024
release 1024 elements
```

它表明模拟 buffer 在构造时获取资源，移动后源对象为空，最终只有目标对象释放资源。

智能指针示例输出：

```text
load demo-model
owners=2
unload demo-model
cache_expired=true
```

它表明两个请求共同拥有模型；请求结束后模型卸载；只持有 `weak_ptr` 的缓存没有延长模型寿命。

## 掌握边界

今天已经通过程序验证：

- CMake 的配置、构建目录与目标关系；
- CTest 根据退出状态判断测试结果；
- `vector` 和 `unique_ptr` 的自动清理；
- 移动后所有权由源对象转给目标对象；
- `shared_ptr` 强引用数量与 `weak_ptr` 观察关系；
- `variant` 成功/错误分支；
- `optional` 的有值/无值表达；
- shell 退出码；
- ASan、UBSan 的基本使用。

今天尚未验证：

- 真实 `cudaMalloc/cudaFree`；
- CUDA Stream 上的异步资源寿命；
- 自定义拷贝赋值和移动赋值；
- `shared_ptr` 循环引用；
- 迭代器失效；
- 多 dtype 模板；
- 动态库 `.so` 和符号故障；
- TSan 与多线程竞争。

## 剩余计划

按阶段 1 新导航依次推进：

1. 完成[生命周期](../../learning/stage-01/before-learning/01-lifetime/README.md)练习，能手画构造、移动与析构顺序；
2. 运行[智能指针](../../learning/stage-01/before-learning/02-smart-pointers/README.md)示例，修改作用域并预测引用计数；
3. 学习[STL基础](../../learning/stage-01/before-learning/03-stl/README.md)，重点验证 vector 扩容后的迭代器失效；
4. 学习[模板特性](../../learning/stage-01/before-learning/04-templates/README.md)，为后续多 dtype kernel 做准备；
5. 复盘[C++17](../../learning/stage-01/before-learning/05-cpp17/README.md)并补足边界测试；
6. 快速浏览[C++20](../../learning/stage-01/before-learning/06-cpp20/README.md)，不深入协程实现；
7. 对照[错误处理](../../learning/stage-01/before-learning/07-errors/README.md)改进解析错误上下文；
8. 完成[ABI链接](../../learning/stage-01/before-learning/08-abi/README.md)的 `nm/ldd/readelf` 实验；
9. 使用[构建测试](../../learning/stage-01/before-learning/09-build-test/README.md)统一复查 ASan/UBSan；
10. 完成阶段练习后进入 GPU 并行基础。

## 下次任务

下一小节只做智能指针，不同时展开 STL 或并发：

1. 运行 `smart_pointer_demo`；
2. 画出两个 `shared_ptr` 与一个 `weak_ptr` 的关系；
3. 在每个作用域边界记录 `use_count()`；
4. 删除最后一个强引用后检查 `expired()`；
5. 完成一份不含猜测的运行记录。

## 直观理解

今天不是“学完了 C++ 生命周期”，而是第一次把所有权画成可运行证据：资源何时取得、交给谁、何时释放，都能由输出和测试验证。下一步逐个扩大场景，不同时铺开多个主题。
