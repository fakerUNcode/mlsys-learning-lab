# inference/

## 职责

研究模型加载、预处理、执行、后处理和服务化的端到端推理路径，关注延迟、吞吐、显存、并发、batch、精度与稳定性之间的权衡。

## 推荐布局

```text
inference/
├── README.md
├── models/                 # 模型配置/下载说明，不提交大权重
├── runners/                # eager、compile、runtime 等执行器
├── benchmarks/             # 端到端 latency/throughput 测试
└── configs/                # 可复现实验参数
```

## 指标与实验

预热后分别报告 cold start、steady-state latency（p50/p95/p99）、throughput、峰值显存、模型加载时间和精度差异。固定输入分布、batch、并发、线程、功耗/时钟条件；避免把数据加载时间和 GPU 执行时间混为一谈。

## 完成标准

至少有一个 baseline runner 和一个优化 runner，支持命令行配置，并能用同一输入集做准确性与性能对照。
