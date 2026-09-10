# Benchmarks

统一保存可复现的性能基准。这里负责输入生成、正确性校验、warmup、计时、统计与结果输出；被测实现应放在所属模块中。

## 当前状态

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 向量加法 | 可运行 | 使用 PyTorch 比较 CPU 或 CUDA 路径 |
| 公共计时器 | 待实现 | 后续抽取同步和统计逻辑 |
| 结果格式 | 待实现 | 后续增加 JSON/CSV schema |

## 快速运行

从仓库根目录执行：

```bash
python benchmarks/benchmark_vector_add.py \
  --device cpu --n 1048576 --warmup 5 --iters 20
```

GPU 示例：

```bash
python benchmarks/benchmark_vector_add.py \
  --device cuda --n 16777216 --warmup 20 --iters 100 \
  --dtype float32
```

该脚本测量 PyTorch 已有向量加法，不是自定义 `.cu` kernel。GPU 路径使用 CUDA Event 并进行必要同步。

## 指标解释

- `p50_ms`：中位延迟，描述典型运行。
- `p95_ms`、`p99_ms`：尾延迟，帮助观察抖动。
- `std_ms`：采样离散程度。
- `effective_read_write_bandwidth_GB_s`：按两次读取、一次写入估算的有效带宽，不等于硬件峰值。

比较两个实现时，设备、dtype、输入规模、warmup、迭代次数和同步方式必须一致。

## 目录约定

```text
benchmarks/
├── README.md
├── common.py              # 规划：计时与结果 schema
├── benchmark_<topic>.py   # 每个主题一个入口
└── results/               # 规划：摘要可提交，大文件忽略
```

新增脚本应支持适用的 `--device`、`--warmup`、`--iters`、`--seed` 和 `--output` 参数，并在计时前完成正确性检查。

## 完成标准

每个主题至少包含 baseline、候选实现、自动正确性检查、重复测量和可复现结果。报告需要解释性能变化，不能只展示单个最快数字。
