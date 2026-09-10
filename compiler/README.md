# Compiler Experiments

研究机器学习编译器中的图捕获、IR、合法变换、算子融合、内存规划、调度与代码生成。

## 当前状态

该模块处于规划阶段，尚无端到端编译实验。计划从 PyTorch eager/FX/`torch.compile` 的可观察案例开始，再进入 Triton、LLVM 或 MLIR。

## 目录约定

```text
compiler/
├── README.md
├── fx/                     # 图捕获与变换
├── torch_compile/          # 编译前后对照
├── triton/                 # kernel 与调度实验
├── mlir/                   # dialect、pass、lowering
├── graphs/                 # 可审阅的图与 IR 快照
└── tests/
```

目录将在首个实验加入时按需创建，不预先放置空目录。

## 实验记录

- eager baseline 与输入约束；
- 捕获的图或 IR；
- 变换前后代码；
- 等价性测试与失败样例；
- trace/compile 时间与 steady-state 性能；
- 动态 shape、控制流、alias 和数值精度假设。

编译时间与运行收益必须分开。一次输入上的输出相同不足以证明变换普遍正确，需要覆盖边界和约束失效情况。

## 完成标准

至少完成一个端到端图或 IR 优化实验；变换前后结果由自动化测试验证；保存可阅读的中间表示，并通过统一 benchmark 复现性能结论。
