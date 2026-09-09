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

## 当前可运行示例

`benchmark_vector_add.py` 测量 PyTorch 已有的向量加法 CUDA 算子。它不是自定义 `.cu` 编译实验，而是用来练习 benchmark 方法：先做正确性检查，再预热；GPU 路径使用 CUDA Event 和同步；最后报告 min/mean/p50/p95/p99/std。

```bash
source .venv/bin/activate
python benchmarks/benchmark_vector_add.py \
  --device cuda --n $((1 << 24)) --warmup 20 --iters 100 --dtype float32
```

如果想先用较小数据快速试跑：

```bash
python benchmarks/benchmark_vector_add.py \
  --device cuda --n $((1 << 20)) --warmup 5 --iters 20
```

`p50_ms` 是稳定中位数，`p95_ms` 反映尾延迟；`effective_read_write_bandwidth_GB_s` 只是根据读取 A/B、写入 C 的字节数估算的有效带宽，不等于硬件理论峰值。比较优化版本时，必须保持设备、dtype、元素数量、warmup 和迭代次数一致。

## 完成标准

至少有一个 baseline、一个候选优化实现、自动正确性检查和可复现结果文件；报告应解释瓶颈，而不是只给单个平均耗时。
