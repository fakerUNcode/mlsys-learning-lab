# compiler/

## 职责

研究 ML 编译器的图捕获、IR 表示、合法变换、算子融合、布局/内存规划、调度和代码生成。这里的实验重点是“变换为何正确、何时有效、边界在哪里”。

## 推荐布局

```text
compiler/
├── README.md
├── fx/ 或 torch_compile/
├── triton/ 或 mlir/
├── graphs/                 # 捕获的图和 IR 快照
├── passes/                 # 独立变换实验
└── tests/
```

## 记录内容

保存 eager baseline、捕获图/IR、优化前后代码、编译时间、运行指标和失败样例。区分 trace/compile 开销与 steady-state 执行开销，说明动态 shape、控制流、别名和数值精度假设。

## 完成标准

至少完成一个端到端图优化实验，并能用小型单元测试证明变换前后输出一致、性能结论可复现。
