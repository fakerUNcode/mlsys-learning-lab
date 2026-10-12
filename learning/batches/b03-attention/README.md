# B03：Attention

B03 属于 TASKS 阶段 1 的第 3 周，依赖 B02，包含 T08～T10。

| 任务 | 主题 | 章节与证据 |
| --- | --- | --- |
| T08 | 单头缩放点积 Attention | [章节与 reference](t08-scaled-dot-product-attention/README.md)、[验证报告](../../../reports/stage-02/2026-10-11-t08-validation.md) |
| T09 | 因果 mask 与多头 | [章节与实现](t09-causal-mask-and-multihead/README.md)、[完成记录](../../../plan/completed/T09.md)、[验证报告](../../../reports/stage-02/2026-10-11-t09-validation.md) |
| T10 | 数值稳定性与 Attention 成本 | [实验与笔记](t10-numerical-stability-and-attention-cost/README.md)、[完成记录](../../../plan/completed/T10.md)、[验证报告](../../../reports/stage-02/2026-10-12-t10-validation.md) |

T08 的程序可从仓库根目录运行：

```bash
.venv/bin/python learning/batches/b03-attention/t08-scaled-dot-product-attention/attention_reference.py
```

T08 的 CPU float64 对照、T09 的 CPU 多头因果语义检查、T10 的 CPU/GPU float64 数值与资源估算检查通过；T09 学习者脱离讲义独立复写已验收，T08 独立复写状态仍见各自记录，T10 独立复写/口述未验收。T10 未覆盖低精度与性能；任务范围与阶段门槛以[任务清单](../../../plan/TASKS.md)为准。
