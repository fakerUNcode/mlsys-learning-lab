# 历史总结

> 本页是 2026-09-10 Stage 1 实验总结的旧入口。每日学习记录现已统一迁移到根目录 [daily-record](../../daily-record/README.md)；整理后的当天记录见 [2026-09-10](../../daily-record/2026-09-10.md)。下文保留作为原始实验复盘。

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
- 曾尝试把示例拆成三个工程，随后发现这会让既有报告、目标名和命令失配；
- 最终恢复原始单工程，并把详细注释放在与报告一致的源码和 CMake 中。

## 今日证据

最终目录的验证结果：

```text
stage1_test Passed

100% tests passed
0 tests failed out of 1
```

当前可执行程序输出 `count=4 sum=10` 和 `status=ok`。它验证解析、vector、variant、optional 与 unique_ptr 路径；模拟 GPU buffer 和 shared/weak 模型缓存仍属于讲义内容，不能计为已经运行验证。

## 掌握边界

今天已经通过程序验证：

- CMake 的配置、构建目录与目标关系；
- CTest 根据退出状态判断测试结果；
- `vector` 和 `unique_ptr` 的自动清理；
- 移动后所有权由源对象转给目标对象；
- `variant` 成功/错误分支；
- `optional` 的有值/无值表达；
- shell 退出码；
- ASan、UBSan 的基本使用。

今天尚未验证：

- 真实 `cudaMalloc/cudaFree`；
- CUDA Stream 上的异步资源寿命；
- 自定义拷贝赋值和移动赋值；
- `shared_ptr` 引用计数、weak_ptr 观察和循环引用；
- 迭代器失效；
- 多 dtype 模板；
- 动态库 `.so` 和符号故障；
- TSan 与多线程竞争。

## 剩余计划

按阶段 1 新导航依次推进：

1. 完成[生命周期](../../learning/stage-01/lessons/01-object-lifetime-and-move/README.md)练习，能手画构造、移动与析构顺序；
2. 运行[智能指针](../../learning/stage-01/lessons/02-smart-pointer-ownership/README.md)示例，修改作用域并预测引用计数；
3. 学习[STL基础](../../learning/stage-01/lessons/03-stl-containers-and-algorithms/README.md)，重点验证 vector 扩容后的迭代器失效；
4. 学习[模板特性](../../learning/stage-01/lessons/04-templates-and-type-traits/README.md)，为后续多 dtype kernel 做准备；
5. 复盘[C++17](../../learning/stage-01/lessons/05-cpp17-types-and-features/README.md)并补足边界测试；
6. 快速浏览[C++20](../../learning/stage-01/lessons/06-cpp20-concepts-and-coroutines/README.md)，不深入协程实现；
7. 对照[错误处理](../../learning/stage-01/lessons/07-error-handling/README.md)改进解析错误上下文；
8. 完成[ABI链接](../../learning/stage-01/lessons/08-abi-and-dynamic-linking/README.md)的 `nm/ldd/readelf` 实验；
9. 使用[构建测试](../../learning/stage-01/lessons/09-build-test-sanitizers/README.md)统一复查 ASan/UBSan；
10. 完成阶段练习后进入 GPU 并行基础。

## 下次任务

下一小节只做智能指针，不同时展开 STL 或并发：

1. 在现有单工程中新增智能指针代码前，先确定它对应哪篇报告；
2. 画出两个 `shared_ptr` 与一个 `weak_ptr` 的关系；
3. 设计每个作用域边界应记录的 `use_count()`；
4. 设计最后一个强引用删除后的 `expired()` 检查；
5. 同时更新源码、CMake、测试、README 和报告，再运行验证。

## 直观理解

今天不是“学完了 C++ 生命周期”，而是第一次把所有权画成可运行证据：资源何时取得、交给谁、何时释放，都能由输出和测试验证。下一步逐个扩大场景，不同时铺开多个主题。
