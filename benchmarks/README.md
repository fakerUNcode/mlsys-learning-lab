# benchmarks/

## 职责

存放可重复的性能基准：统一输入生成、warmup、计时、重复实验、结果序列化和可视化。它回答“哪个实现更快、快在哪里、代价是什么”，不负责承载被测实现本身。

## 推荐布局

```text
benchmarks/
├── README.md
├── common.py              # timer、同步、结果 schema
├── benchmark_<topic>.py   # 一个主题一个入口
└── results/               # gitignore 大型原始结果，保留摘要/示例
```

## 运行与规范

CPU/GPU 计时必须区分 wall-clock 与 device time；CUDA 测量前后同步，先 warmup 再采样，并报告 median、p90/p99、吞吐、显存峰值和输入规模。每个 benchmark 应支持 `--device`、`--batch-size`、`--warmup`、`--iters`、`--seed`、`--output`。

## 完成标准

至少有一个 baseline、一个候选优化实现、自动正确性检查和可复现结果文件；报告应解释瓶颈，而不是只给单个平均耗时。
