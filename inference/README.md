# Inference

研究从模型加载、请求排队、预处理、GPU 执行到后处理的端到端推理链路，以及延迟、吞吐、显存、并发和精度之间的取舍。

## 当前状态

该模块处于规划阶段，尚未提供模型 runner。后续会在完成 C++ Runtime、CUDA 和 PyTorch 扩展基础后建立可复现 baseline。

## 目录约定

```text
inference/
├── README.md
├── models/                 # 配置与下载说明，不提交大权重
├── runners/                # eager、compile、runtime 执行器
├── benchmarks/             # 端到端性能测试
└── configs/                # 固定实验参数
```

## 指标约定

| 指标 | 含义 |
| --- | --- |
| cold start | 模型加载、初始化或编译后的首次延迟 |
| p50/p95/p99 | 稳态延迟与尾延迟 |
| throughput | 单位时间完成的请求数或 token 数 |
| peak memory | 测量区间内的峰值显存 |
| queue time | 请求进入系统到开始执行的等待时间 |
| execution time | 实际 CPU/GPU 执行时间 |

实验必须记录输入分布、batch、并发、dtype、线程数、设备和软件版本。数据加载、排队、Host 调度、H2D/D2H 与 GPU 执行应尽可能分开报告。

## 数据约定

不要提交模型权重、访问令牌或私有请求数据。为公开模型记录来源、版本和下载命令，为合成输入记录生成方式与随机种子。

## 完成标准

至少包含一个 baseline runner 和一个有明确假设的优化 runner；二者使用同一输入集完成准确性与性能对照，并可通过命令行配置复现。
