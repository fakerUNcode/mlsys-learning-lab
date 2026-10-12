# T10 数值稳定性与 Attention 成本验证

日期：2026-10-12。范围：稳定 softmax、全 mask 行约定、Attention 中间矩阵资源估算与 FlashAttention IO 边界。

## 环境与方法

- 环境：仓库 `.venv`；PyTorch 2.11.0+cu128；CPU float64 与 NVIDIA GeForce RTX 5080 Laptop GPU（CUDA 12.8）float64。GPU 属性查询报告显存总量约 15.9 GiB；本实验只分配小张量。
- 脚本：`.venv/bin/python learning/batches/b03-attention/t10-numerical-stability-and-attention-cost/attention_cost.py`
- 自动检查：`.venv/bin/python learning/batches/b03-attention/t10-numerical-stability-and-attention-cost/test_attention_cost.py`
- GPU 脚本：`.venv/bin/python learning/batches/b03-attention/t10-numerical-stability-and-attention-cost/attention_cost_gpu.py`
- GPU 自动检查：`.venv/bin/python learning/batches/b03-attention/t10-numerical-stability-and-attention-cost/test_attention_cost_gpu.py`
- 随机种子：不适用，输入均为固定显式值；资源账使用整数公式，不分配大矩阵。
- 资料：R04《Attention Is All You Need》§3.2.1；R18《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》§3.1～3.2；R21 服务讲义 04～05。R18 论文于 2026-10-12 回查 [arXiv 页面](https://arxiv.org/abs/2205.14135)，R04 原文见 [arXiv](https://arxiv.org/abs/1706.03762)。

## 实际输出

```text
torch=2.11.0+cu128 device=cpu dtype=float64
overflow_input=[[1000.0, 999.0, 998.0]]
naive=[[nan, nan, nan]] finite=False
stable=[[0.6652409557748218, 0.24472847105479764, 0.09003057317038046]] finite=True
masked_probabilities=[[1.0, 0.0], [0.0, 0.0]]
masked_row_sums=[1.0, 0.0]

B=1 H=16 head_dim=64 dtype=fp16; one attention layer
S | scores(each) | probabilities(each) | explicit bool mask | Q,K,V,O | K+V cache | 2 matmuls
4096 | 512.0 MiB | 512.0 MiB | 256.0 MiB | 32.0 MiB | 16.0 MiB | 68,719,476,736
8192 | 2.0 GiB | 2.0 GiB | 1.0 GiB | 64.0 MiB | 32.0 MiB | 274,877,906,944
doubling_ratios=score:4x qkvo:2x attention_flops:4x
```

自动检查输出：5 项 unittest 全通过，退出码 0。覆盖大正数溢出对照、稳定结果归一化、全 mask 行返回零、可广播 mask 和资源翻倍关系。

GPU 程序输出（同一数值输入在 CUDA Tensor 上运行）：

```text
torch=2.11.0+cu128 device=NVIDIA GeForce RTX 5080 Laptop GPU cuda=12.8
memory_before_free=14.6 GiB total=15.9 GiB allocated=0.0 MiB reserved=0.0 MiB
naive=[[nan, nan, nan]] finite=False
stable=[[0.6652409557748218, 0.24472847105479764, 0.09003057317038046]] finite=True
masked_probabilities=[[1.0, 0.0], [0.0, 0.0]]
masked_row_sums=[1.0, 0.0]
memory_after_allocated=0.0 MiB reserved=2.0 MiB
doubling ledger: score/probability 512 MiB -> 2.0 GiB; QKVO 32 -> 64 MiB
GPU T10 checks passed (correctness only; no performance benchmark)
```

CUDA unittest 2 项通过，退出码 0：稳定 softmax 输入/输出均为 CUDA Tensor 且有限；全 mask 行精确返回 GPU 上的 `[0,0]`。资源表仍由 CPU 整数运算生成，没有在 GPU 上申请 score/probability 大矩阵。

## 结论

| 检查点 | 证据与结论 |
| --- | --- |
| 减最大值 | `[1000,999,998]` 朴素 softmax 得到 NaN；减最大值后结果有限，约为 `[0.665241,0.244728,0.090031]`，行和为 1。 |
| 全 mask 行 | 采用明确约定：注意力概率为全零，乘 V 后注意力输出为零向量；不把有效行的概率和为 1 断言套到全 mask 行。 |
| 序列翻倍 | S 从 4096 到 8192，score、probability、完全展开的 bool mask、两次 Attention 矩阵乘法 FLOPs 均增至 4 倍；Q/K/V/O 和每层 K/V cache 增至 2 倍。 |
| FlashAttention 的读写 | 分块、在线 softmax 和融合减少完整 S×S score/probability 中间矩阵对 HBM 的写出与读回；输入 Q/K/V 仍要读取、输出仍要写出。稠密 Attention 的 query-key 配对算术量仍为二次增长。 |

资料复核中，R21 讲义 04 的“数学结果逐位相同”不宜作为浮点实现保证：分块改变归约顺序，浮点舍入结果可能略有差别。本章只称数学定义精确等价，并保留浮点舍入边界。内存 IO 复杂度需结合片上 SRAM 容量、head dimension 与算法循环假设解释，不能据此说总读写字节在任意配置下都线性。

## 边界

- 数值实验分别在 CPU 与 GPU 使用 float64 小输入；没有测 fp16/bf16 softmax、反向传播或 PyTorch 实际选择的融合后端。
- 资源表是理论张量字节数，不含投影、反向激活、分配器碎片和临时工作区；bool mask 按完整 `[B,H,S,S]` 显式物化计数，广播或隐式 mask 会更小。
- GPU 程序使用 PyTorch CUDA 张量算子，不是自定义 CUDA/Triton kernel，也没有基准测试 FlashAttention；不能由账本推断具体设备峰值显存或速度提升。
- 独立复写/口述理解尚未验收，自动测试不代表学习者独立掌握。

## 产物

- [T10 章节笔记](../../learning/batches/b03-attention/t10-numerical-stability-and-attention-cost/README.md)
- [实验实现](../../learning/batches/b03-attention/t10-numerical-stability-and-attention-cost/attention_cost.py)
- [GPU 实验实现](../../learning/batches/b03-attention/t10-numerical-stability-and-attention-cost/attention_cost_gpu.py)
- [自动检查](../../learning/batches/b03-attention/t10-numerical-stability-and-attention-cost/test_attention_cost.py)
- [GPU 自动检查](../../learning/batches/b03-attention/t10-numerical-stability-and-attention-cost/test_attention_cost_gpu.py)
- [完成记录](../../plan/completed/T10.md)
